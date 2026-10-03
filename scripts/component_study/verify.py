"""Verify all control records and recompute the declared paired comparisons on CPU."""

import argparse
import json
from pathlib import Path
import statistics

from scripts.layer_study.configuration import ROOT
from scripts.layer_study.prepare import split_payloads
from scripts.layer_study.result_io import read_json, read_rows
from scripts.layer_study.verify_results import verify_checksums
from scripts.multiseed.verify import same

DATA = ROOT / "results/nyu_component_ablation"
REFERENCE = ROOT / "results/nyu_multiseed"
SEEDS = tuple(range(42, 47))
VARIANTS = ("main_only", "early", "middle", "late")
METRICS = ("abs_rel", "sq_rel", "rmse", "mae", "rmse_log", "a1", "a2", "a3")


def stats(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "per_seed": dict(zip(map(str, SEEDS), values)),
    }


def validate_run(folder, test_order):
    meta = read_json(folder / "test_metrics.json")
    initial = read_json(folder / "initialization.json")
    complete = read_json(folder / "training_complete.json")
    history = read_rows(folder / "history.jsonl")
    rows = read_rows(folder / "test_per_image.jsonl")
    if [row["image_path"] for row in rows] != test_order or len(set(test_order)) != 654:
        raise ValueError("Official test identities or order differ")
    if meta["n_images"] != 654 or [row["epoch"] for row in history] != list(range(1, 21)):
        raise ValueError("Expected 654 test images and 20 complete epochs")
    if initial["smoke_only"] or initial["config_sha256"] != meta["config_sha256"]:
        raise ValueError("Training/test configuration identities differ")
    if not complete["completed"] or not complete["backbone_unchanged"]:
        raise ValueError("Training did not complete with an unchanged backbone")
    if complete["official_test_used_for_selection"]:
        raise ValueError("Test data were used for checkpoint selection")
    candidates = [
        (row[f"validation_{source}"]["regions"]["all"]["abs_rel"], row["epoch"], source)
        for row in history
        for source in ("raw", "ema")
    ]
    score, epoch, source = min(candidates, key=lambda x: (x[0], x[1], x[2] != "raw"))
    if (epoch, source) != (meta["checkpoint_epoch"], meta["checkpoint_source"]):
        raise ValueError("Checkpoint differs from the validation minimum")
    same(score, complete["best_validation_abs_rel"])
    if meta["median_scaling"] or not meta["flip_tta"] or meta["crop"] != "eigen":
        raise ValueError("Test evaluation protocol differs")
    for region, summary in meta["regions"].items():
        samples = [row["metrics"][region] for row in rows if row["metrics"][region]["pixels"] > 0]
        same(len(samples), summary["n_images"])
        same(sum(row["pixels"] for row in samples), summary["n_pixels"])
        for metric in METRICS:
            same(statistics.mean(row[metric] for row in samples), summary[metric])
    return meta, initial


def recompute(data=DATA, reference=REFERENCE):
    data, reference = Path(data), Path(reference)
    splits, _ = split_payloads()
    order = [line.split()[0].lstrip("/") for line in splits["test"].decode().splitlines()]
    runs = {variant: [] for variant in VARIANTS}
    protocol = read_json(data / "protocol.json")
    for seed in SEEDS:
        folder = data / str(seed)
        meta, initial = validate_run(folder, order)
        audit = read_json(folder / "initialization_audit.json")
        smoke = read_json(folder / "smoke_audit.json")
        expected = read_json(reference / str(seed) / "early/initialization.json")
        if (
            initial["trainable_parameters"] != 7570932
            or initial["variant"] != "main_only"
            or audit["seed"] != seed
            or not audit["reference_initialization_matched"]
            or audit["parameter_matched"]
            or not smoke["dual_path_patch_forward_bitwise_identical"]
        ):
            raise ValueError("Control initialization provenance differs")
        same(audit["full_trainable_parameters"], expected["trainable_parameters"])
        same(audit["control_trainable_parameters"], initial["trainable_parameters"])
        same(audit["control_initial_trainable_sha256"], initial["initial_trainable_sha256"])
        same(initial["config_sha256"], protocol["original_config_sha256"][str(seed)])
        same(initial["backbone_sha256"], expected["backbone_sha256"])
        runs["main_only"].append(meta)
        for variant in VARIANTS[1:]:
            values, _ = validate_run(reference / str(seed) / variant, order)
            runs[variant].append(values)
    summaries, differences = {}, {}
    for region in runs["main_only"][0]["regions"]:
        summaries[region], differences[region] = {}, {}
        for metric in METRICS:
            summaries[region][metric] = {
                variant: stats([row["regions"][region][metric] for row in values])
                for variant, values in runs.items()
            }
            differences[region][metric] = {}
            for variant in VARIANTS[1:]:
                values = [
                    full["regions"][region][metric] - control["regions"][region][metric]
                    for full, control in zip(runs[variant], runs["main_only"])
                ]
                result = stats(values)
                result["dual_path_better_seeds"] = sum(
                    x > 0 if metric in ("a1", "a2", "a3") else x < 0 for x in values
                )
                differences[region][metric][variant] = result
    return {"summaries": summaries, "paired_differences": differences}


def verify(data=DATA, reference=REFERENCE):
    data = Path(data)
    count = verify_checksums(data)
    actual = recompute(data, reference)
    expected = read_json(data / "component_comparison.json")
    for key, values in actual.items():
        same(values, expected[key], key)
    return {
        "passed": True,
        "archive_files": count,
        "control_runs": 5,
        "total_runs": 20,
        "test_images_per_run": 654,
        "validation_selected_checkpoints": 20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.reference), indent=2))


if __name__ == "__main__":
    main()
