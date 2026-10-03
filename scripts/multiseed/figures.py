"""Figures for the five-seed revision: perturbation effects and seed repeatability."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["early", "middle", "late"]
CONDITIONS = ["lowpass", "lowpass_rms", "highpass_rms"]
LABELS = ["Low-pass", "Low-pass\nenergy\nmatched", "High-pass\nenergy\nmatched"]
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


def save(fig, name, output):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def perturbation_figure(N, output):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True, layout="constrained")
    upper = max(
        N["perturbation"][v][c][r]["delta_ci95_scene_bootstrap"][1]
        for v in VARIANTS
        for c in CONDITIONS
        for r in ["boundary", "interior"]
    )
    for ax, variant, letter in zip(axs, VARIANTS, "abc"):
        for offset, region in [(-0.12, "boundary"), (0.12, "interior")]:
            mean = np.array(
                [N["perturbation"][variant][c][region]["delta_mean_sd"][0] for c in CONDITIONS]
            )
            ci = np.array(
                [
                    N["perturbation"][variant][c][region]["delta_ci95_scene_bootstrap"]
                    for c in CONDITIONS
                ]
            )
            per_seed = np.array(
                [N["perturbation"][variant][c][region]["delta_per_seed"] for c in CONDITIONS]
            )
            x = np.arange(3) + offset
            ax.vlines(x, ci[:, 0], ci[:, 1], color=COLORS[region], lw=1.2)
            jitter = np.linspace(-0.045, 0.045, per_seed.shape[1])
            ax.scatter(
                np.repeat(x, per_seed.shape[1]) + np.tile(jitter, per_seed.shape[0]),
                per_seed.ravel(),
                s=6,
                color=COLORS[region],
                alpha=0.35,
                lw=0,
                zorder=2,
                label="_nolegend_",
            )
            ax.plot(x, mean, "o", ms=4.5, color=COLORS[region], label=region.capitalize(), zorder=3)
            ax.scatter(x, ci[:, 0], marker="_", color=COLORS[region], s=25)
            ax.scatter(x, ci[:, 1], marker="_", color=COLORS[region], s=25)
        ax.axhline(0, color="#777777", lw=1)
        ax.set(
            title=f"{letter}  {variant.capitalize()} layers",
            xticks=range(3),
            xticklabels=LABELS,
            ylim=(-0.008, upper * 1.08),
        )
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="y", alpha=0.15)
    axs[0].set_ylabel("Change in AbsRel")
    axs[0].legend(frameon=False, loc="upper left", title="Region", title_fontsize=7)
    axs[2].text(
        0.98,
        0.03,
        "small marks: five training seeds",
        transform=axs[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    save(fig, "frequency_perturbation", output)


def repeatability_figure(N, output):
    SEEDS = N["seeds"]
    fig, axs = plt.subplots(
        1, 2, figsize=(7.2, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    panel_a = axs[0]
    for index, variant in enumerate(VARIANTS):
        values = [N["test_per_seed"][variant]["all"][str(seed)]["abs_rel"] for seed in SEEDS]
        mean = N["test"][variant]["all"]["abs_rel"][0]
        sd = N["test"][variant]["all"]["abs_rel"][1]
        shade = {"early": "#C44E26", "middle": "#0072B2", "late": "#009E73"}[variant]
        xj = np.linspace(-0.16, 0.16, len(SEEDS))
        panel_a.scatter(index + xj, values, s=22, color=shade, alpha=0.75, zorder=3, lw=0)
        panel_a.hlines(mean, index - 0.26, index + 0.26, color=shade, lw=1.6, zorder=4)
        panel_a.vlines(index, mean - sd, mean + sd, color=shade, lw=6, alpha=0.22, zorder=2)
        worst = int(np.argmax(values))
        if values[worst] - mean > 2 * sd:
            panel_a.annotate(
                "one run did not\nconverge",
                xy=(index + xj[worst], values[worst]),
                xytext=(index + 0.36, values[worst] + 0.001),
                fontsize=6.2,
                color="#444444",
                ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"),
            )
    panel_a.set(
        xticks=range(3),
        xticklabels=[v.capitalize() for v in VARIANTS],
        xlabel="Structural-pathway layers",
        ylabel="Test AbsRel",
        title="a  Five independent training runs",
        ylim=(0.115, 0.155),
    )
    panel_a.grid(axis="y", alpha=0.15)

    panel_b = axs[1]
    offsets = {"early_minus_middle": -0.18, "early_minus_late": 0.0, "middle_minus_late": 0.18}
    marks = {"early_minus_middle": "o", "early_minus_late": "s", "middle_minus_late": "^"}
    for key, offset in offsets.items():
        first, second = key.replace("_minus_", "|").split("|")
        values = [
            N["test_per_seed"][first]["all"][str(seed)]["abs_rel"]
            - N["test_per_seed"][second]["all"][str(seed)]["abs_rel"]
            for seed in SEEDS
        ]
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        color = "#444444"
        panel_b.scatter(
            [offset] * len(SEEDS),
            values,
            marker=marks[key],
            s=22,
            color=color,
            alpha=0.6,
            lw=0,
            zorder=3,
        )
        panel_b.hlines(mean, offset - 0.13, offset + 0.13, color=color, lw=1.6, zorder=4)
        panel_b.vlines(offset, mean - sd, mean + sd, color=color, lw=6, alpha=0.18, zorder=2)
    panel_b.axhline(0, color="#777777", lw=1)
    panel_b.set(
        xticks=list(offsets.values()),
        xticklabels=["Early\n−\nMiddle", "Early\n−\nLate", "Middle\n−\nLate"],
        ylabel="Paired AbsRel difference",
        title="b  Same-seed differences",
        xlabel="Variant pair",
    )
    panel_b.tick_params(axis="x", labelsize=6.5)
    panel_b.grid(axis="y", alpha=0.15)
    save(fig, "seed_repeatability", output)


def spectra_figure(frozen, pooled, output):
    names = [f"L{i:02d}" for i in range(13)]
    fig = plt.figure(figsize=(7.2, 6.7), layout="constrained")
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 0.98])

    axs = subfigs[0].subplots(1, 2, gridspec_kw={"width_ratios": [0.92, 1.08]})
    mat = np.array([frozen["windows"]["hann"]["features"][n]["profile"] for n in names])
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
        f = frozen["windows"][window]["features"]
        y = np.array([f[n]["bands"]["high"]["mean"] for n in names])
        ci = np.array([f[n]["bands"]["high"]["ci95_scene_bootstrap"] for n in names])
        axs[1].plot(range(13), y, "o-", ms=3, label=window.capitalize(), color=color)
        axs[1].fill_between(range(13), ci[:, 0], ci[:, 1], color=color, alpha=0.13)
    axs[1].plot(
        range(13),
        [frozen["uniform_mean_color_control"]["hann"][n]["bands"]["high"] for n in names],
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
    for ax, variant, letter in zip(axs, VARIANTS, "cde"):
        f = pooled["hann"][variant]["features"]
        ordered = sorted(f)
        y = np.arange(len(ordered))
        means = np.array([f[n]["pooled_mean"] for n in ordered])
        ci = np.array([f[n]["ci95_scene_bootstrap"] for n in ordered])
        for j, name in enumerate(ordered):
            color = "#0072B2" if name.startswith("semantic") else "#C44E26"
            ax.hlines(j, *ci[j], color=color, lw=1.2)
            ax.plot(means[j], j, "o", color=color, ms=4)
        labels = [
            n.replace("semantic_", "Semantic ").replace("structural_", "Structural ")
            for n in ordered
        ]
        ax.set(
            yticks=y,
            yticklabels=labels,
            xlim=(0, 1),
            xlabel="High-frequency AC fraction",
            title=f"{letter}  {variant.capitalize()} decoder",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    subfigs[1].suptitle(
        "Projected features, pooled over five training seeds", fontsize=7.5, y=1.02, color="#444444"
    )

    save(fig, "feature_spectra", output)
