from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("study/qualitative/predictions.npz"))
    parser.add_argument("--output", type=Path, default=Path("study/paper-figures"))
    args = parser.parse_args()
    R = args.output.resolve()
    R.mkdir(parents=True, exist_ok=True)
    z = np.load(args.input)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axs = plt.subplots(
        5,
        5,
        figsize=(7.2, 6.05),
        layout="constrained",
        gridspec_kw={"wspace": 0.015, "hspace": 0.025},
    )
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#d0d0d0")
    for i in range(5):
        axs[i, 0].imshow(z[f"rgb_{i}"])
        for j, name in enumerate(["gt", "early", "middle", "late"], 1):
            dep = np.ma.array(z[f"{name}_{i}"], mask=~z[f"mask_{i}"].astype(bool))
            im = axs[i, j].imshow(
                dep, cmap=cmap, norm=Normalize(0.001, 10), interpolation="nearest"
            )
        for ax in axs[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        axs[i, 0].set_ylabel(f"Example {i + 1}", fontsize=8)
    for ax, t in zip(axs[0], ["RGB", "Reference", "Early", "Middle", "Late"]):
        ax.set_title(t, fontsize=9)
    cb = fig.colorbar(
        im,
        ax=axs[:, 1:],
        location="bottom",
        shrink=0.65,
        pad=0.015,
        aspect=45,
        ticks=[0, 2, 4, 6, 8, 10],
    )
    cb.set_label("Depth (m); gray: outside the valid evaluation mask")
    fig.savefig(R / "sr-qualitative-controlled.png", dpi=300, bbox_inches="tight")
    fig.savefig(R / "sr-qualitative-controlled.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
