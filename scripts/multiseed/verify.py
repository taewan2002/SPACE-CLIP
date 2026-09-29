"""Check hashes and recompute five-seed results from the per-image records."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.layer_study.result_io import read_rows
from scripts.layer_study.configuration import VARIANTS
from scripts.layer_study.prepare import split_payloads
from scripts.layer_study.verify_results import verify_checksums
from scripts.multiseed import aggregate, spectra


def same(actual, expected, path="root"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{path}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: lengths differ")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{path}/{i}")
    elif isinstance(expected, (int, float)):
        if not np.isclose(actual, expected, atol=1e-12, rtol=1e-10):
            raise ValueError(f"{path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"{path}: values differ")


def verify(data=aggregate.DATA):
    data = Path(data).resolve()
    count = verify_checksums(data)
    splits, _ = split_payloads()
    test_order = [line.split()[0].lstrip("/") for line in splits["test"].decode().splitlines()]
    sample = json.loads(
        (aggregate.ROOT / "results/nyu_layer_study/sample_manifest.json").read_text()
    )
    frequency_order = [line.split()[0] for line in sample["lines"]]
    sham_checks = 0
    initializations = {}
    for seed in aggregate.SEEDS:
        for variant in aggregate.VARIANTS:
            run = data / str(seed) / variant
            records = read_rows(run / "test_per_image.jsonl")
            order = [r["image_path"] for r in records]
            if len(order) != 654 or len(set(order)) != 654:
                raise ValueError("Expected 654 distinct official test images per run")
            if order != test_order:
                raise ValueError("Different test order across seeds/variants")
            meta = json.loads((run / "test_metrics.json").read_text())
            init = json.loads((run / "initialization.json").read_text())
            complete = json.loads((run / "training_complete.json").read_text())
            initializations[seed, variant] = init
            if init["structural_layers"] != VARIANTS[variant] or init["smoke_only"]:
                raise ValueError("Layer assignment or initialization differs")
            if init["config_sha256"] != meta["config_sha256"]:
                raise ValueError("Checkpoint configuration identity differs")
            if (
                not complete["completed"]
                or not complete["backbone_unchanged"]
                or complete["official_test_used_for_selection"]
            ):
                raise ValueError("Training provenance differs")
            history = read_rows(run / "history.jsonl")
            if [r["epoch"] for r in history] != list(range(1, 21)):
                raise ValueError("Expected 20 complete training epochs")
            candidates = [
                (r[f"validation_{source}"]["regions"]["all"]["abs_rel"], r["epoch"], source)
                for r in history
                for source in ("raw", "ema")
            ]
            score, epoch, source = min(candidates, key=lambda x: (x[0], x[1], x[2] != "raw"))
            if (epoch, source) != (meta["checkpoint_epoch"], meta["checkpoint_source"]):
                raise ValueError("Checkpoint is not the validation minimum")
            same(score, complete["best_validation_abs_rel"])
            if meta["median_scaling"] or meta["crop"] != "eigen" or not meta["flip_tta"]:
                raise ValueError("Evaluation protocol differs")
            for region, metrics in meta["regions"].items():
                for metric in aggregate.METRICS:
                    values = [
                        r["metrics"][region][metric]
                        for r in records
                        if r["metrics"][region]["pixels"] > 0
                    ]
                    same(
                        float(np.mean(values)),
                        metrics[metric],
                        f"{seed}/{variant}/{region}/{metric}",
                    )
            rows = read_rows(run / "perturbation_per_image.jsonl")
            order = [r["image_path"] for r in rows]
            if len(order) != 200 or len(set(order)) != 200:
                raise ValueError("Expected 200 distinct frequency samples")
            if order != frequency_order:
                raise ValueError("Frequency samples differ")
            groups = {r["scene"] for r in rows}
            if len(groups) != 25 or any(sum(r["scene"] == g for r in rows) != 8 for g in groups):
                raise ValueError("Expected 25 scenes, eight frames per scene")
            summary = json.loads((run / "frequency_summary.json").read_text())
            if summary["max_sham_depth_difference"] != 0 or any(
                r["sham_max_abs_depth_difference"] != 0 for r in rows
            ):
                raise ValueError("Nonzero FFT round trip")
            sham_checks += len(rows)
    for seed in aggregate.SEEDS:
        for field in (
            "initial_trainable_sha256",
            "backbone_sha256",
            "trainable_parameters",
            "total_parameters",
        ):
            if len({initializations[seed, v][field] for v in aggregate.VARIANTS}) != 1:
                raise ValueError(f"Unmatched initialization: seed {seed}, {field}")
    if len({i["backbone_sha256"] for i in initializations.values()}) != 1:
        raise ValueError("Frozen backbone differs across seeds")
    if len({initializations[s, "early"]["initial_trainable_sha256"] for s in aggregate.SEEDS}) != 5:
        raise ValueError("Expected independent decoder initializations across five seeds")
    with tempfile.TemporaryDirectory() as temp:
        aggregate.generate(data, Path(temp))
        spectra.generate(data, Path(temp))
        for name in ["manuscript_numbers.json", "projected_pooled.json"]:
            same(
                json.loads((Path(temp) / name).read_text()),
                json.loads((data / name).read_text()),
                name,
            )
    return {
        "passed": True,
        "numeric_files": count,
        "runs": 15,
        "test_images_per_run": 654,
        "zero_difference_sham_checks": sham_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=aggregate.DATA)
    args = parser.parse_args()
    print(json.dumps(verify(args.data), indent=2))


if __name__ == "__main__":
    main()
