"""Check curated measurements and reproduce reported aggregates without a GPU."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.layer_study.configuration import ROOT, VARIANTS
from scripts.layer_study.metrics import aggregate
from scripts.layer_study.prepare import split_payloads
from scripts.layer_study.result_io import read_json, read_rows
from scripts.layer_study.spectral import perturbation_summary, scene_summary

DEFAULT = ROOT / "results/nyu_layer_study"


def equal(actual, expected, location="result"):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Different keys at {location}")
        for key, value in expected.items():
            equal(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, (list, float, int)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, err_msg=location)
    elif actual != expected:
        raise ValueError(f"Different values at {location}: {actual!r} != {expected!r}")


def verify_checksums(directory):
    checksums = read_json(directory / "checksums.json")
    files = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file()
        and (p.name.endswith(".json") or p.name.endswith(".jsonl.gz"))
        and p.name != "checksums.json"
    }
    if files != set(checksums):
        raise ValueError("Curated measurement inventory differs from checksums.json")
    for name, expected in checksums.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(checksums)


def verify(directory=DEFAULT):
    directory = Path(directory)
    checked = verify_checksums(directory)
    splits, _ = split_payloads()
    expected_test = [line.split()[0].lstrip("/") for line in splits["test"].decode().splitlines()]
    sample = read_json(directory / "sample_manifest.json")
    expected_sample = [line.split()[0] for line in sample["lines"]]
    if (
        hashlib.sha256(("\n".join(sample["lines"]) + "\n").encode()).hexdigest()
        != sample["selection_sha256"]
    ):
        raise ValueError("Fourier sample identity mismatch")
    initializations = []
    for variant in VARIANTS:
        run = directory / "runs" / variant
        metrics = read_json(run / "test_metrics.json")
        rows = read_rows(run / "test_per_image.jsonl")
        if [r["image_path"] for r in rows] != expected_test or len(rows) != 654:
            raise ValueError("Official test images or ordering differ")
        equal(aggregate(rows), {k: metrics[k] for k in ("n_images", "regions")}, variant)
        if metrics["median_scaling"] or not metrics["flip_tta"] or metrics["crop"] != "eigen":
            raise ValueError("Unexpected evaluation protocol")
        init = read_json(run / "initialization.json")
        initializations.append(init)
        if init["structural_layers"] != VARIANTS[variant]:
            raise ValueError("Layer assignment mismatch")
        complete = read_json(run / "training_complete.json")
        if (
            not complete["completed"]
            or not complete["backbone_unchanged"]
            or complete["official_test_used_for_selection"]
        ):
            raise ValueError("Training provenance mismatch")
        history = read_rows(run / "history.jsonl")
        if [r["epoch"] for r in history] != list(range(1, 21)):
            raise ValueError("Incomplete training history")
        candidates = [
            (row[f"validation_{source}"]["regions"]["all"]["abs_rel"], row["epoch"], source)
            for row in history
            for source in ("raw", "ema")
        ]
        score, epoch, source = min(candidates)
        if (epoch, source) != (metrics["checkpoint_epoch"], metrics["checkpoint_source"]):
            raise ValueError("Selected checkpoint is not the validation minimum")
        equal(score, complete["best_validation_abs_rel"])
        fourier = directory / "fourier" / variant
        summary = read_json(fourier / "summary.json")
        perturbations = read_rows(fourier / "perturbation_per_image.jsonl")
        check_sample(perturbations, expected_sample)
        if any(r["sham_max_abs_depth_difference"] != 0.0 for r in perturbations):
            raise ValueError("Nonzero sham prediction difference")
        equal(
            perturbation_summary(perturbations), summary["perturbation"], f"{variant}.perturbation"
        )
        for window in ("hann", "rectangular"):
            projected = read_rows(fourier / f"projected_per_image_{window}.jsonl")
            check_sample(projected, expected_sample)
            equal(
                scene_summary(projected),
                summary["projected_spectra"][window],
                f"{variant}.{window}",
            )
    for key in (
        "trainable_parameters",
        "total_parameters",
        "initial_trainable_sha256",
        "backbone_sha256",
    ):
        if len({init[key] for init in initializations}) != 1:
            raise ValueError(f"Uncontrolled initialization: {key}")
    raw_summary = read_json(directory / "fourier/raw/summary.json")
    for window in ("hann", "rectangular"):
        raw = read_rows(directory / f"fourier/raw/per_image_{window}.jsonl")
        check_sample(raw, expected_sample)
        equal(scene_summary(raw), raw_summary["windows"][window], f"raw.{window}")
    return {
        "verified_files": checked,
        "test_images_per_variant": 654,
        "training_epochs_per_variant": 20,
        "fourier_images_per_variant": 200,
        "fourier_scenes": 25,
        "zero_difference_sham_checks": 600,
    }


def check_sample(rows, expected):
    if [r["image_path"] for r in rows] != expected or len(rows) != 200:
        raise ValueError("Fourier sample/order mismatch")
    counts = Counter(r["scene"] for r in rows)
    if len(counts) != 25 or set(counts.values()) != {8}:
        raise ValueError("Expected eight images from each of 25 validation scenes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
