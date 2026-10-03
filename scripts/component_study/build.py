"""Rebuild component-ablation tables and paired statistics without model weights."""

import argparse
import csv
import json
from pathlib import Path

from scripts.component_study.verify import DATA, REFERENCE, ROOT, SEEDS, VARIANTS, recompute

LABELS = {"main_only": "Semantic only + FiLM", "early": "Early", "middle": "Middle", "late": "Late"}


def build(data=DATA, reference=REFERENCE, output=ROOT / "study/component-tables"):
    data, reference, output = (Path(p).resolve() for p in (data, reference, output))
    if any(output == source or source in output.parents for source in (data, reference)):
        raise ValueError("Generated outputs must not overwrite measurements")
    output.mkdir(parents=True, exist_ok=True)
    numbers = recompute(data, reference)
    (output / "component_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    all_metrics = numbers["summaries"]["all"]
    boundary = numbers["summaries"]["boundary_0.05"]["abs_rel"]
    lines = [
        "# Five-seed component ablation",
        "",
        "Mean ± sample SD; all five seeds retained.",
        "",
        "| Model | AbsRel | RMSE (m) | δ₁ | Boundary AbsRel |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    tex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & AbsRel & RMSE (m) & $\delta_1$ & Boundary AbsRel \\",
        r"\midrule",
    ]
    for variant in VARIANTS:
        values = [all_metrics[m][variant] for m in ("abs_rel", "rmse", "a1")] + [boundary[variant]]
        md = [f"{v['mean']:.4f} ± {v['sample_sd']:.4f}" for v in values]
        cells = [f"{v['mean']:.4f} $\\pm$ {v['sample_sd']:.4f}" for v in values]
        lines.append("| " + " | ".join([LABELS[variant], *md]) + " |")
        tex.append(" & ".join([LABELS[variant], *cells]) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    lines.extend(
        ["", "Paired differences are full minus control; negative errors favor the full model.", ""]
    )
    for variant in VARIANTS[1:]:
        paired = numbers["paired_differences"]["all"]["abs_rel"][variant]
        reduction = -paired["mean"] / all_metrics["abs_rel"]["main_only"]["mean"] * 100
        lines.append(
            f"- {LABELS[variant]}: {paired['mean']:+.4f} ± {paired['sample_sd']:.4f}; "
            f"lower error in {paired['dual_path_better_seeds']}/5 seeds; "
            f"{reduction:.1f}% reduction in the five-seed mean."
        )
    (output / "component_summary.md").write_text("\n".join(lines) + "\n")
    (output / "component_table.tex").write_text("\n".join(tex) + "\n")
    with (output / "component_per_seed.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "variant", "abs_rel", "boundary_abs_rel", "paired_abs_rel"])
        for seed in SEEDS:
            for variant in VARIANTS:
                score = all_metrics["abs_rel"][variant]["per_seed"][str(seed)]
                base = all_metrics["abs_rel"]["main_only"]["per_seed"][str(seed)]
                writer.writerow(
                    [seed, variant, score, boundary[variant]["per_seed"][str(seed)], score - base]
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "study/component-tables")
    args = parser.parse_args()
    print("Generated:", build(args.data, args.reference, args.output))


if __name__ == "__main__":
    main()
