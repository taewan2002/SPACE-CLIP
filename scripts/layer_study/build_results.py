from pathlib import Path
import csv
import json
import shutil
import gzip
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Reaggregate archived measurements and regenerate tables and spectral figures; no GPU or model download needed."
    )
    parser.add_argument(
        "--study",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results/nyu_layer_study",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[2] / "study/paper-figures"
    )
    args = parser.parse_args()
    SRC = args.study.resolve()
    REPO = args.output.resolve()
    REPO.mkdir(parents=True, exist_ok=True)
    WORK = REPO
    DATA = REPO / "source-data"
    DATA.mkdir(exist_ok=True)
    TABLES = REPO / "result-tables"
    TABLES.mkdir(exist_ok=True)
    VARIANTS = ["early", "middle", "late"]
    COLORS = {"boundary": "#C44E26", "interior": "#0072B2"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.1,
        }
    )

    def load(p):
        return json.loads(p.read_text())

    def rows(p):
        content = (
            p.read_text()
            if p.exists()
            else gzip.decompress(p.with_suffix(p.suffix + ".gz").read_bytes()).decode()
        )
        return [json.loads(x) for x in content.splitlines() if x.strip()]

    def csvwrite(name, rs):
        with (DATA / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rs)

    def save(fig, name):
        fig.savefig(REPO / (name + ".pdf"), bbox_inches="tight")
        fig.savefig(REPO / (name + ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = {v: load(SRC / "fourier" / v / "summary.json") for v in VARIANTS}
    metrics = {v: load(SRC / "runs" / v / "test_metrics.json") for v in VARIANTS}
    # Flatten archived raw spectra to the original source-data schema.
    raw_records = []
    for window in ["hann", "rectangular"]:
        for row in rows(SRC / "fourier/raw" / f"per_image_{window}.jsonl"):
            for layer, spec in row["spectra"].items():
                raw_records.append(
                    {
                        "image_path": row["image_path"],
                        "scene": row["scene"],
                        "window": window,
                        "layer": layer,
                        "channels": spec["channels"],
                        "height": spec["grid"][0],
                        "width": spec["grid"][1],
                        "ac_energy": spec["ac_energy"],
                        **{k + "_fraction": val for k, val in spec["bands"].items()},
                        **{f"radial_bin_{i}": val for i, val in enumerate(spec["profile"])},
                    }
                )
    csvwrite("frozen_layer_spectra.csv", raw_records)

    # Main comparison table: identical test examples and protocol.
    lines = []
    for v in VARIANTS:
        m = metrics[v]["regions"]
        a = m["all"]
        layers = {"early": "2, 1, 0", "middle": "7, 5, 4", "late": "11, 10, 8"}[v]
        lines.append(
            f"{v.capitalize()} & {layers} & {a['abs_rel']:.4f} & {a['rmse']:.4f} & {a['a1']:.4f} & {m['boundary_0.05']['abs_rel']:.4f} & {m['interior_0.05']['abs_rel']:.4f} \\\\"
        )
    (TABLES / "main-comparison.tex").write_text("\n".join(lines) + "\n")
    # Additional accuracy and checkpoint-selection tables.
    lines = []
    for v in VARIANTS:
        a = metrics[v]["regions"]["all"]
        m = metrics[v]
        lines.append(
            f"{v.capitalize()} & {m['checkpoint_epoch']} & {m['checkpoint_source'].upper()} & {a['sq_rel']:.4f} & {a['mae']:.4f} & {a['rmse_log']:.4f} & {a['a2']:.4f} & {a['a3']:.4f} \\\\"
        )
    (TABLES / "additional-metrics.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for t in ["0.03", "0.05", "0.10"]:
        for v in VARIANTS:
            b = metrics[v]["regions"]["boundary_" + t]
            i = metrics[v]["regions"]["interior_" + t]
            lines.append(
                f"{t} & {v.capitalize()} & {b['abs_rel']:.4f} & {b['rmse']:.4f} & {i['abs_rel']:.4f} & {i['rmse']:.4f} \\\\"
            )
    (TABLES / "boundary-sensitivity.tex").write_text("\n".join(lines) + "\n")
    # Full test source data, long-form region records.
    records = []
    for v in VARIANTS:
        for row in rows(SRC / "runs" / v / "test_per_image.jsonl"):
            for region, m in row["metrics"].items():
                records.append(
                    {"variant": v, "image_path": row["image_path"], "region": region, **m}
                )
    csvwrite("test_depth_metrics.csv", records)
    # Perturbation source data and exploratory paired contrast across regions.
    records = []
    contrasts = []
    for v in VARIANTS:
        rr = rows(SRC / "fourier" / v / "perturbation_per_image.jsonl")
        groups = sorted({r["scene"] for r in rr})
        for row in rr:
            for cond, regions in row["conditions"].items():
                for region, m in regions.items():
                    records.append(
                        {
                            "variant": v,
                            "image_path": row["image_path"],
                            "scene": row["scene"],
                            "condition": cond,
                            "region": region,
                            **m,
                            "sham_max_depth_difference": row["sham_max_abs_depth_difference"],
                        }
                    )
        for cond in ["lowpass", "lowpass_rms", "highpass_rms"]:
            delta = np.array(
                [
                    np.mean(
                        [
                            (
                                r["conditions"][cond]["boundary"]["abs_rel"]
                                - r["conditions"]["baseline"]["boundary"]["abs_rel"]
                            )
                            - (
                                r["conditions"][cond]["interior"]["abs_rel"]
                                - r["conditions"]["baseline"]["interior"]["abs_rel"]
                            )
                            for r in rr
                            if r["scene"] == g
                        ]
                    )
                    for g in groups
                ]
            )
            rng = np.random.default_rng(20260927)
            draws = rng.integers(0, 25, size=(2000, 25))
            ci = np.quantile(delta[draws].mean(1), [0.025, 0.975])
            contrasts.append(
                {
                    "variant": v,
                    "condition": cond,
                    "boundary_minus_interior_delta": float(delta.mean()),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "n_scenes": 25,
                    "bootstrap_seed": 20260927,
                    "analysis": "exploratory post-hoc region contrast; intervals are not multiplicity-adjusted",
                }
            )
    csvwrite("frequency_perturbation_metrics.csv", records)
    csvwrite("exploratory_region_contrasts.csv", contrasts)
    (WORK / "region-contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    # Projected feature source data.
    records = []
    for v in VARIANTS:
        for window in ["hann", "rectangular"]:
            for row in rows(SRC / "fourier" / v / f"projected_per_image_{window}.jsonl"):
                for feature, s in row["spectra"].items():
                    records.append(
                        {
                            "variant": v,
                            "window": window,
                            "image_path": row["image_path"],
                            "scene": row["scene"],
                            "feature": feature,
                            "channels": s["channels"],
                            "height": s["grid"][0],
                            "width": s["grid"][1],
                            "ac_energy": s["ac_energy"],
                            **{k + "_fraction": val for k, val in s["bands"].items()},
                            **{f"radial_bin_{i}": val for i, val in enumerate(s["profile"])},
                        }
                    )
    csvwrite("projected_feature_spectra.csv", records)
    # Summaries for independent regeneration and control energies.
    shutil.copy2(SRC / "fourier/raw/summary.json", DATA / "frozen_feature_summary.json")
    for v in VARIANTS:
        shutil.copy2(SRC / "fourier" / v / "summary.json", DATA / f"{v}_frequency_summary.json")
    # Raw spectra, shown without assigning semantic/structural functions.
    r = load(SRC / "fourier/raw/summary.json")
    names = [f"L{i:02d}" for i in range(13)]
    fig, axs = plt.subplots(
        1, 2, figsize=(7.0, 3.05), layout="constrained", gridspec_kw={"width_ratios": [0.92, 1.08]}
    )
    mat = np.array([r["windows"]["hann"]["features"][n]["profile"] for n in names])
    im = axs[0].imshow(
        mat, origin="lower", aspect="auto", extent=(0, 10, -0.5, 12.5), cmap="magma", vmin=0
    )
    axs[0].set(
        xlabel="Radial frequency (cycles / input field)",
        ylabel="CLIP layer",
        yticks=[0, 3, 6, 9, 12],
        title="a  Frozen patch-feature spectra",
    )
    fig.colorbar(im, ax=axs[0], label="Fraction of AC energy", shrink=0.85)
    for window, color in [("hann", "#0072B2"), ("rectangular", "#D55E00")]:
        f = r["windows"][window]["features"]
        y = np.array([f[n]["bands"]["high"]["mean"] for n in names])
        ci = np.array([f[n]["bands"]["high"]["ci95_scene_bootstrap"] for n in names])
        axs[1].plot(range(13), y, "o-", ms=3, label=window.capitalize(), color=color)
        axs[1].fill_between(range(13), ci[:, 0], ci[:, 1], color=color, alpha=0.13)
    axs[1].plot(
        range(13),
        [r["uniform_mean_color_control"]["hann"][n]["bands"]["high"] for n in names],
        "--",
        color="#666666",
        label="Uniform input (Hann)",
    )
    axs[1].set(
        xlabel="CLIP layer (L0: embedding output)",
        ylabel="High-frequency AC fraction",
        xticks=[0, 3, 6, 9, 12],
        ylim=(0, 1),
        title="b  Frequency content across layers",
    )
    axs[1].legend(loc="lower right", frameon=False)
    axs[1].grid(alpha=0.15)
    save(fig, "sr-frozen-layer-spectrum")
    # Native common-axis perturbation plot with scene-bootstrap intervals.
    conditions = ["lowpass", "lowpass_rms", "highpass_rms"]
    labels = ["Low-pass", "Low-pass\nenergy\nmatched", "High-pass\nenergy\nmatched"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True, layout="constrained")
    upper = max(
        summary[v]["perturbation"][c][region]["delta_ci95_scene_bootstrap"][1]
        for v in VARIANTS
        for c in conditions
        for region in ["boundary", "interior"]
    )
    for ax, v, letter in zip(axs, VARIANTS, "abc"):
        for offset, region in [(-0.12, "boundary"), (0.12, "interior")]:
            y = np.array(
                [
                    summary[v]["perturbation"][c][region]["delta_abs_rel_scene_mean"]
                    for c in conditions
                ]
            )
            ci = np.array(
                [
                    summary[v]["perturbation"][c][region]["delta_ci95_scene_bootstrap"]
                    for c in conditions
                ]
            )
            x = np.arange(3) + offset
            ax.vlines(x, ci[:, 0], ci[:, 1], color=COLORS[region], lw=1.2)
            ax.plot(x, y, "o", ms=4, color=COLORS[region], label=region.capitalize())
            ax.scatter(x, ci[:, 0], marker="_", color=COLORS[region])
            ax.scatter(x, ci[:, 1], marker="_", color=COLORS[region])
        ax.axhline(0, color="#777777", lw=1)
        ax.set(
            title=f"{letter}  {v.capitalize()} layers",
            xticks=range(3),
            xticklabels=labels,
            ylim=(-0.004, upper * 1.1),
        )
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="y", alpha=0.15)
    axs[0].set_ylabel("Change in AbsRel")
    axs[0].legend(frameon=False, loc="upper left")
    save(fig, "sr-frequency-perturbation")
    # Projected-feature spectra (each feature normalized separately).
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.4), sharex=True, layout="constrained")
    for ax, v, letter in zip(axs, VARIANTS, "abc"):
        f = summary[v]["projected_spectra"]["hann"]["features"]
        ordered = sorted(f)
        y = np.arange(len(ordered))
        m = [f[n]["bands"]["high"]["mean"] for n in ordered]
        ci = np.array([f[n]["bands"]["high"]["ci95_scene_bootstrap"] for n in ordered])
        for j, n in enumerate(ordered):
            color = "#0072B2" if n.startswith("semantic") else "#C44E26"
            ax.hlines(j, *ci[j], color=color, lw=1.2)
            ax.plot(m[j], j, "o", color=color, ms=4)
        labels2 = [
            n.replace("semantic_", "Main ").replace("structural_", "Additional ") for n in ordered
        ]
        ax.set(
            yticks=y,
            yticklabels=labels2,
            xlim=(0, 1),
            xlabel="High-frequency AC fraction",
            title=f"{letter}  {v.capitalize()} decoder",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    save(fig, "sr-projected-spectra")
    # Complete perturbation tables (all conditions and regions).
    for v in VARIANTS:
        lines = []
        for cond, label in [
            ("baseline", "Baseline"),
            ("sham", "FFT round trip"),
            ("lowpass", "Low-pass"),
            ("lowpass_rms", "Low-pass, energy matched"),
            ("highpass_rms", "High-pass, energy matched"),
        ]:
            for region in ["all", "boundary", "interior"]:
                a = summary[v]["perturbation"][cond][region]
                lo, hi = a["delta_ci95_scene_bootstrap"]
                lines.append(
                    f"{label} & {region.capitalize()} & {a['abs_rel_scene_mean']:.4f} & {a['delta_abs_rel_scene_mean']:+.4f} & [{lo:+.4f}, {hi:+.4f}] \\\\"
                )
        (TABLES / f"{v}-perturbation.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for r in contrasts:
        label = {
            "lowpass": "Low-pass",
            "lowpass_rms": "Low-pass, energy matched",
            "highpass_rms": "High-pass, energy matched",
        }[r["condition"]]
        lines.append(
            f"{r['variant'].capitalize()} & {label} & {r['boundary_minus_interior_delta']:+.5f} & [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] \\\\"
        )
    (TABLES / "region-contrasts.tex").write_text("\n".join(lines) + "\n")
    counts = {f.name: sum(1 for _ in f.open()) - 1 for f in DATA.glob("*.csv")}
    (WORK / "source-data-counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))

    # Integrated manuscript figure and complete archival table regeneration.
    r = load(SRC / "fourier/raw/summary.json")
    names = [f"L{i:02d}" for i in range(13)]
    fig = plt.figure(figsize=(7.2, 6.7), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 0.98])
    axs = subfigs[0].subplots(1, 2, gridspec_kw={"width_ratios": [0.92, 1.08]})
    mat = np.array([r["windows"]["hann"]["features"][n]["profile"] for n in names])
    im = axs[0].imshow(
        mat, origin="lower", aspect="auto", extent=(0, 10, -0.5, 12.5), cmap="magma", vmin=0
    )
    axs[0].set(
        xlabel="Radial frequency (cycles / input field)",
        ylabel="CLIP layer",
        yticks=[0, 3, 6, 9, 12],
        title="a  Frozen patch-feature spectra",
    )
    subfigs[0].colorbar(im, ax=axs[0], label="Fraction of AC energy", shrink=0.85)
    for window, color in [("hann", "#0072B2"), ("rectangular", "#D55E00")]:
        f = r["windows"][window]["features"]
        y = np.array([f[n]["bands"]["high"]["mean"] for n in names])
        ci = np.array([f[n]["bands"]["high"]["ci95_scene_bootstrap"] for n in names])
        axs[1].plot(range(13), y, "o-", ms=3, label=window.capitalize(), color=color)
        axs[1].fill_between(range(13), ci[:, 0], ci[:, 1], color=color, alpha=0.13)
    axs[1].plot(
        range(13),
        [r["uniform_mean_color_control"]["hann"][n]["bands"]["high"] for n in names],
        "--",
        color="#666666",
        label="Uniform input (Hann)",
    )
    axs[1].set(
        xlabel="CLIP layer (L0: embedding output)",
        ylabel="High-frequency AC fraction",
        xticks=[0, 3, 6, 9, 12],
        ylim=(0, 1),
        title="b  Frequency content across layers",
    )
    axs[1].legend(loc="lower right", frameon=False)
    axs[1].grid(alpha=0.15)

    axs = subfigs[1].subplots(1, 3, sharex=True)
    for ax, v, letter in zip(axs, VARIANTS, "cde"):
        f = summary[v]["projected_spectra"]["hann"]["features"]
        ordered = sorted(f)
        y = np.arange(len(ordered))
        m = [f[n]["bands"]["high"]["mean"] for n in ordered]
        ci = np.array([f[n]["bands"]["high"]["ci95_scene_bootstrap"] for n in ordered])
        for j, n in enumerate(ordered):
            color = "#0072B2" if n.startswith("semantic") else "#C44E26"
            ax.hlines(j, *ci[j], color=color, lw=1.2)
            ax.plot(m[j], j, "o", color=color, ms=4)
        labels2 = [
            n.replace("semantic_", "Main ").replace("structural_", "Additional ") for n in ordered
        ]
        ax.set(
            yticks=y,
            yticklabels=labels2,
            xlim=(0, 1),
            xlabel="High-frequency AC fraction",
            title=f"{letter}  {v.capitalize()} decoder",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)

    save(fig, "sr-feature-spectra-combined")

    energy = load(SRC / "fourier/raw/summary.json")["uniform_absolute_energy_control"]["hann"]
    lines = []
    for name in ["L00", "L05", "L08", "L12"]:
        e = energy[name]
        lines.append(
            f"L{int(name[1:])} & {e['natural_scene_mean_ac_energy']:.2f} & {e['uniform_ac_energy']:.2f} & {e['uniform_over_natural_energy']:.4f} "
            + r"\\"
        )
    (TABLES / "uniform-energy.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
