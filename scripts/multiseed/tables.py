"""Generate complete five-seed manuscript table rows from derived measurements."""

VARIANTS = ["early", "middle", "late"]
LAYERS = {"early": "2, 1, 0", "middle": "7, 5, 4", "late": "11, 10, 8"}
CONDITION_LABEL = {
    "lowpass": "Low-pass",
    "lowpass_rms": "Low-pass, energy matched",
    "highpass_rms": "High-pass, energy matched",
}


def ms(pair):
    return "%.4f $\\pm$ %.4f" % (pair[0], pair[1])


def write_tables(N, selection, OUT):
    SELECTION = {
        v: {
            seed: (r["checkpoint_epoch"], r["checkpoint_source"].upper())
            for seed, r in seeds.items()
        }
        for v, seeds in selection.items()
    }
    test = N["test"]
    lines = []
    for variant in VARIANTS:
        a = test[variant]["all"]
        lines.append(
            "%s & %s & %s & %s & %s & %s & %s \\\\"
            % (
                variant.capitalize(),
                LAYERS[variant],
                ms(a["abs_rel"]),
                ms(a["rmse"]),
                ms(a["a1"]),
                ms(test[variant]["boundary_0.05"]["abs_rel"]),
                ms(test[variant]["interior_0.05"]["abs_rel"]),
            )
        )
    (OUT / "table-main.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for variant in VARIANTS:
        a = test[variant]["all"]
        epochs = [str(SELECTION[variant][seed][0]) for seed in N["seeds"]]
        weights = sorted({SELECTION[variant][seed][1] for seed in N["seeds"]})
        lines.append(
            "%s & %s & %s & %s & %s & %s \\\\"
            % (
                variant.capitalize(),
                ", ".join(epochs),
                "/".join(weights),
                ms(a["sq_rel"]),
                ms(a["mae"]),
                ms(a["rmse_log"]),
            )
        )
    (OUT / "table-additional.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for threshold in ["0.03", "0.05", "0.10"]:
        for variant in VARIANTS:
            b = test[variant]["boundary_" + threshold]
            i = test[variant]["interior_" + threshold]
            lines.append(
                "%s & %s & %s & %s & %s & %s \\\\"
                % (
                    threshold,
                    variant.capitalize(),
                    ms(b["abs_rel"]),
                    ms(b["rmse"]),
                    ms(i["abs_rel"]),
                    ms(i["rmse"]),
                )
            )
    (OUT / "table-boundary.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for variant in VARIANTS:
        lines.append("\\multicolumn{5}{l}{\\textbf{%s variant}} \\\\" % variant.capitalize())
        for region, label in [("all", "All"), ("boundary", "Boundary"), ("interior", "Interior")]:
            base = N["perturbation"][variant]["baseline"][region]["baseline"]
            lines.append(
                "Baseline / FFT round trip & %s & %s & +0.0000 $\\pm$ 0.0000 & -- \\\\"
                % (label, ms(base))
            )
        for condition in ["lowpass", "lowpass_rms", "highpass_rms"]:
            for region, label in [
                ("all", "All"),
                ("boundary", "Boundary"),
                ("interior", "Interior"),
            ]:
                entry = N["perturbation"][variant][condition][region]
                low, high = entry["delta_ci95_scene_bootstrap"]
                lines.append(
                    "%s & %s & -- & %s & [%.4f, %.4f] \\\\"
                    % (CONDITION_LABEL[condition], label, ms(entry["delta_mean_sd"]), low, high)
                )
        lines.append("\\midrule")
    lines.pop()
    (OUT / "table-perturbations.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for item in N["region_contrasts"]:
        low, high = item["ci95_scene_bootstrap_on_seed_averaged"]
        positives = sum(1 for v in item["per_seed"] if v > 0)
        lines.append(
            "%s & %s & %s & [%.5f, %.5f] & %d/5 \\\\"
            % (
                item["variant"].capitalize(),
                CONDITION_LABEL[item["condition"]],
                "%+.5f $\\pm$ %.5f" % (item["mean_sd"][0], item["mean_sd"][1]),
                low,
                high,
                positives,
            )
        )
    (OUT / "table-contrasts.tex").write_text("\n".join(lines) + "\n")
