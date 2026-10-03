"""Produce a descriptive comparison; one seed is not a stability estimate."""

import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for variant in ("early", "middle", "late"):
        metrics = json.loads((ROOT / "study/runs" / variant / "test_metrics.json").read_text())
        init = json.loads((ROOT / "study/runs" / variant / "initialization.json").read_text())
        regions = metrics["regions"]
        rows.append(
            {
                "variant": variant,
                "layers": str(init["structural_layers"]),
                "trainable_parameters": init["trainable_parameters"],
                "n_images": metrics["n_images"],
                "best_epoch": metrics["checkpoint_epoch"],
                "source": metrics["checkpoint_source"],
                "abs_rel": regions["all"]["abs_rel"],
                "rmse": regions["all"]["rmse"],
                "boundary_abs_rel": regions["boundary_0.05"].get("abs_rel"),
                "boundary_rmse": regions["boundary_0.05"].get("rmse"),
                "interior_abs_rel": regions["interior_0.05"].get("abs_rel"),
                "boundary_n_images": regions["boundary_0.05"]["n_images"],
            }
        )
    with (ROOT / "study/comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "study/comparison.json").write_text(json.dumps(rows, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for axis, key, title in zip(
        axes, ("abs_rel", "boundary_abs_rel"), ("Overall depth error", "GT depth boundary error")
    ):
        values = [r[key] for r in rows]
        if any(v is None for v in values):
            raise RuntimeError("No evaluable depth boundaries")
        bars = axis.bar(
            [r["variant"] for r in rows], values, color=["#31688e", "#35b779", "#f2b134"]
        )
        axis.bar_label(bars, fmt="%.4f", padding=3)
        axis.set_ylabel("AbsRel (lower is better)")
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.18)
    fig.suptitle("NYU structural layer selection · seed 42 · single-run comparison")
    fig.savefig(ROOT / "study/layer_comparison.png", dpi=200)
    fig.savefig(ROOT / "study/layer_comparison.pdf")
    plt.close(fig)
    lines = [
        "# SPACE-CLIP layer selection results",
        "",
        "Single training seed (42); variability across training seeds was not measured.",
        "Checkpoints were selected on held-out validation scenes and evaluated on all 654 official test images.",
        "These results use a different protocol from the historical release scores.",
        "",
        "|Variant|Layers|AbsRel|RMSE|Boundary AbsRel|Selected epoch|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"|{r['variant']}|{r['layers']}|{r['abs_rel']:.5f}|{r['rmse']:.5f}|{r['boundary_abs_rel']:.5f}|{r['best_epoch']}|"
        )
    lines += [
        "",
        "Boundaries use adjacent GT log-depth differences > 0.05 and a 3-pixel dilation radius.",
        "Invalid-depth neighborhoods and evaluation-crop borders are excluded from regional metrics.",
        "Threshold sensitivities (0.03, 0.10) and per-image metrics are stored in each run directory.",
    ]
    (ROOT / "study/RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
