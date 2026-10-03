"""Derive every manuscript number from the five per-seed per-image records."""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from scripts.layer_study.result_io import read_rows as rows

SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ["early", "middle", "late"]
REGIONS = [
    "all",
    "boundary_0.03",
    "interior_0.03",
    "boundary_0.05",
    "interior_0.05",
    "boundary_0.10",
    "interior_0.10",
]
METRICS = ["abs_rel", "sq_rel", "rmse", "mae", "rmse_log", "a1", "a2", "a3"]
CONDITIONS = ["lowpass", "lowpass_rms", "highpass_rms"]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/nyu_multiseed"
OUT = ROOT / "study/multiseed-figures"


def mean_sd(values):
    return [float(st.mean(values)), float(st.stdev(values))]


def load_test(data=DATA):
    per = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            per[seed, variant] = {
                r["image_path"]: r["metrics"]
                for r in rows(data / str(seed) / variant / "test_per_image.jsonl")
            }
    order = [r["image_path"] for r in rows(data / "42" / "early" / "test_per_image.jsonl")]
    table = {}
    per_seed = {}
    for variant in VARIANTS:
        table[variant] = {}
        per_seed[variant] = {}
        for region in REGIONS:
            seed_summaries = []
            detail = {}
            for seed in SEEDS:
                records = [
                    per[seed, variant][path][region]
                    for path in order
                    if per[seed, variant][path][region]["pixels"] > 0
                ]
                summary = {
                    metric: st.mean([record[metric] for record in records]) for metric in METRICS
                }
                seed_summaries.append(summary)
                detail[str(seed)] = summary
            table[variant][region] = {
                metric: mean_sd([s[metric] for s in seed_summaries]) for metric in METRICS
            }
            per_seed[variant][region] = detail
    return table, per_seed, order


def scene_index(images, scenes, groups):
    return {g: [i for i, p in enumerate(images) if scenes[p] == g] for g in groups}


def bootstrap_ci(scene_values, seed):
    values = np.asarray(scene_values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def load_perturbation(data=DATA):
    per = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            records = rows(data / str(seed) / variant / "perturbation_per_image.jsonl")
            per[seed, variant] = {r["image_path"]: r for r in records}
    reference = rows(data / "42" / "early" / "perturbation_per_image.jsonl")
    images = [r["image_path"] for r in reference]
    scenes = {r["image_path"]: r["scene"] for r in reference}
    groups = sorted(set(scenes.values()))
    assert len(images) == 200 and len(groups) == 25
    for seed in SEEDS:
        for variant in VARIANTS:
            again = [
                r["image_path"]
                for r in rows(data / str(seed) / variant / "perturbation_per_image.jsonl")
            ]
            assert again == images
    index = scene_index(images, scenes, groups)
    table = {}
    for variant in VARIANTS:
        table[variant] = {}
        for condition in ["baseline"] + CONDITIONS:
            table[variant][condition] = {}
            for region in ["all", "boundary", "interior"]:
                seed_means = []
                seed_deltas = []
                for seed in SEEDS:
                    values = [
                        per[seed, variant][p]["conditions"][condition][region]["abs_rel"]
                        for p in images
                    ]
                    base = [
                        per[seed, variant][p]["conditions"]["baseline"][region]["abs_rel"]
                        for p in images
                    ]
                    seed_means.append(st.mean(values))
                    seed_deltas.append([v - b for v, b in zip(values, base)])
                delta_seeds = [st.mean(d) for d in seed_deltas]
                pooled = [sum(col) / len(SEEDS) for col in zip(*seed_deltas)]
                scene_means = [st.mean([pooled[i] for i in index[g]]) for g in groups]
                entry = {
                    "delta_mean_sd": mean_sd(delta_seeds),
                    "delta_per_seed": [round(d, 6) for d in delta_seeds],
                    "delta_ci95_scene_bootstrap": bootstrap_ci(scene_means, 20260926),
                }
                if condition == "baseline":
                    entry["baseline"] = mean_sd(seed_means)
                table[variant][condition][region] = entry
    return table, per, images, groups, scenes


def region_contrasts(per, images, groups, scenes):
    index = scene_index(images, scenes, groups)
    out = []
    for variant in VARIANTS:
        for condition in CONDITIONS:
            seed_values = []
            pooled = None
            for seed in SEEDS:
                deltas = []
                for path in images:
                    record = per[seed, variant][path]["conditions"]
                    deltas.append(
                        (
                            record[condition]["boundary"]["abs_rel"]
                            - record["baseline"]["boundary"]["abs_rel"]
                        )
                        - (
                            record[condition]["interior"]["abs_rel"]
                            - record["baseline"]["interior"]["abs_rel"]
                        )
                    )
                seed_values.append(st.mean(deltas))
                pooled = deltas if pooled is None else [a + d for a, d in zip(pooled, deltas)]
            pooled = [value / len(SEEDS) for value in pooled]
            scene_means = [st.mean([pooled[i] for i in index[g]]) for g in groups]
            out.append(
                {
                    "variant": variant,
                    "condition": condition,
                    "mean_sd": mean_sd(seed_values),
                    "per_seed": [round(v, 6) for v in seed_values],
                    "seed_averaged_mean": st.mean(seed_values),
                    "ci95_scene_bootstrap_on_seed_averaged": bootstrap_ci(scene_means, 20260927),
                }
            )
    return out


def generate(data=DATA, output=OUT):
    data, output = Path(data), Path(output)
    test, per_seed, order = load_test(data)
    pert, per, images, groups, scenes = load_perturbation(data)
    contrast = region_contrasts(per, images, groups, scenes)
    payload = {
        "seeds": SEEDS,
        "test": test,
        "test_per_seed": per_seed,
        "perturbation": pert,
        "region_contrasts": contrast,
        "definitions": {
            "test": "mean over 654 test images within a seed, then mean and sample SD (ddof=1) over five seeds",
            "perturbation": (
                "paired per-image change against the same seed unmodified model; "
                "seed means give mean and SD, and pooled seed-averaged per-image changes "
                "are scene-weighted with 2,000 scene-bootstrap resamples"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manuscript_numbers.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    generate(args.data, args.output)
    print("Saved:", args.output / "manuscript_numbers.json")
