"""Pool projected-feature spectra over the five training seeds with scene bootstrap intervals."""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from scripts.layer_study.result_io import read_rows as rows

SEEDS = [42, 43, 44, 45, 46]
VARIANTS = ["early", "middle", "late"]
WINDOWS = ["hann", "rectangular"]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/nyu_multiseed"
OUT = ROOT / "study/multiseed-figures"


def bootstrap_ci(scene_values, seed):
    values = np.asarray(scene_values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def generate(data=DATA, output=OUT):
    data, output = Path(data), Path(output)
    out = {}
    for window in WINDOWS:
        out[window] = {}
        for variant in VARIANTS:
            per_seed = {}
            scenes = None
            for seed in SEEDS:
                records = rows(data / str(seed) / variant / f"projected_per_image_{window}.jsonl")
                scenes = {r["image_path"]: r["scene"] for r in records}
                for record in records:
                    for feature, spectrum in record["spectra"].items():
                        key = (seed, feature, record["image_path"])
                        per_seed[key] = spectrum["bands"]["high"]
            images = sorted(scenes)
            features = sorted({k[1] for k in per_seed})
            groups = sorted(set(scenes.values()))
            index = {g: [i for i, p in enumerate(images) if scenes[p] == g] for g in groups}
            table = {}
            for feature in features:
                seed_means = []
                pooled = None
                for seed in SEEDS:
                    values = [per_seed[(seed, feature, p)] for p in images]
                    seed_means.append(st.mean(values))
                    pooled = values if pooled is None else [a + b for a, b in zip(pooled, values)]
                pooled = [v / len(SEEDS) for v in pooled]
                scene_means = [st.mean([pooled[i] for i in index[g]]) for g in groups]
                table[feature] = {
                    "mean_sd": [float(st.mean(seed_means)), float(st.stdev(seed_means))],
                    "pooled_mean": float(st.mean(pooled)),
                    "ci95_scene_bootstrap": bootstrap_ci(scene_means, 20260926),
                }
            out[window][variant] = {
                "n_images": len(images),
                "n_scenes": len(groups),
                "features": table,
            }
    output.mkdir(parents=True, exist_ok=True)
    (output / "projected_pooled.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    generate(args.data, args.output)
    print("Saved:", args.output / "projected_pooled.json")
