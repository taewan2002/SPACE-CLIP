"""Exploratory validation-only spectra and inference-time frequency perturbation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import CLIPVisionModel
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.layer_study.configuration import load_config
from scripts.layer_study.run import (
    StrictDepthDataset,
    atomic_json,
    load_decoder,
    seed_everything,
    state_hash,
    config_hash,
    decoder_state,
)
from utils.dataloader import preprocessing_transforms
from scripts.layer_study.prepare import scene_group
from scripts.layer_study.metrics import valid_depth, boundary_band, region_metrics
from scripts.layer_study.spectral import (
    feature_spectrum,
    filter_patch_tokens,
    scene_summary,
    perturbation_summary,
)

OUT = ROOT / "study/fourier"
WINDOWS = ("hann", "rectangular")
CONDITIONS = ("baseline", "sham", "lowpass", "lowpass_rms", "highpass_rms")


def analysis_loader(config, limit=None):
    lines = Path(config["filenames_file_eval"]).read_text().splitlines()
    groups = {}
    for index, line in enumerate(lines):
        groups.setdefault(scene_group(line), []).append(index)
    selected = []
    for group in sorted(groups):
        indices = sorted(
            groups[group],
            key=lambda i: hashlib.sha256(("20260926|" + lines[i]).encode()).hexdigest(),
        )
        selected.extend(indices[:8])
    if limit is not None:
        selected = selected[:limit]
    names = [lines[i] for i in selected]
    selection = {
        "source_partition": "scene-held-out validation",
        "selection_seed": 20260926,
        "max_images_per_scene": 8,
        "n_images": len(selected),
        "n_scenes": len({scene_group(x) for x in names}),
        "selection_sha256": hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest(),
        "lines": names,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUT / ("smoke_sample_manifest.json" if limit else "sample_manifest.json"), selection
    )
    args = argparse.Namespace(**config)
    dataset = StrictDepthDataset(args, "online_eval", preprocessing_transforms(args, "online_eval"))
    return DataLoader(
        Subset(dataset, selected), batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    ), selection


def patch_map(tokens):
    patch = tokens[0, 1:, :]
    side = int(round(patch.shape[0] ** 0.5))
    return patch.transpose(0, 1).reshape(patch.shape[-1], side, side).detach().cpu().numpy()


def absolute_energy_control(rows, blank):
    result = {}
    for window in WINDOWS:
        result[window] = {}
        groups = sorted({r["scene"] for r in rows[window]})
        for name in rows[window][0]["spectra"]:
            natural = float(
                np.mean(
                    [
                        np.mean(
                            [
                                r["spectra"][name]["ac_energy"]
                                for r in rows[window]
                                if r["scene"] == group
                            ]
                        )
                        for group in groups
                    ]
                )
            )
            uniform = blank[window][name]["ac_energy"]
            result[window][name] = {
                "natural_scene_mean_ac_energy": natural,
                "uniform_ac_energy": uniform,
                "uniform_over_natural_energy": uniform / max(natural, 1e-20),
            }
    return result


def plot_raw(summaries, blank, directory):
    names = [f"L{i:02d}" for i in range(13)]
    primary = summaries["hann"]["features"]
    matrix = np.array([primary[name]["profile"] for name in names])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    im = axes[0].imshow(
        matrix, origin="lower", aspect="auto", cmap="magma", extent=(0, 10, -0.5, 12.5), vmin=0
    )
    axes[0].set(
        xlabel="Radial frequency (cycles / input field)",
        ylabel="CLIP layer",
        title="AC energy in radial bands · Hann window",
        yticks=range(13),
    )
    fig.colorbar(im, ax=axes[0], label="Fraction of AC energy")
    for window, color in (("hann", "#0072B2"), ("rectangular", "#D55E00")):
        features = summaries[window]["features"]
        means = [features[n]["bands"]["high"]["mean"] for n in names]
        bounds = np.array([features[n]["bands"]["high"]["ci95_scene_bootstrap"] for n in names])
        axes[1].plot(range(13), means, "o-", label=window, color=color)
        axes[1].fill_between(range(13), bounds[:, 0], bounds[:, 1], alpha=0.15, color=color)
    axes[1].plot(
        range(13),
        [blank["hann"][n]["bands"]["high"] for n in names],
        "--",
        color="#777777",
        label="Uniform control (Hann; separately normalized)",
    )
    axes[1].set(
        xlabel="CLIP layer (L0 = embedding output)",
        ylabel="High-frequency AC energy fraction (r >= 4)",
        title="Frozen representations · scene bootstrap 95% CI",
        ylim=(0, 1),
        xticks=range(13),
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        f"Frozen CLIP ViT-B/16 · {summaries['hann']['n_images']} validation images / "
        f"{summaries['hann']['n_scenes']} scenes · native 14×14 grids"
    )
    fig.supxlabel(
        "Each spectrum is independently normalized by its AC energy; uniform and natural inputs have different absolute energies.",
        fontsize=8,
    )
    fig.savefig(directory / "frozen_layer_spectrum.png", dpi=180)
    fig.savefig(directory / "frozen_layer_spectrum.pdf")
    plt.close(fig)


@torch.no_grad()
def raw_analysis(config, device, limit):
    directory = OUT / ("smoke_raw" if limit else "raw")
    directory.mkdir(parents=True, exist_ok=True)
    model = CLIPVisionModel.from_pretrained(config["clip_model_name"]).to(device).eval()
    expected = json.loads(
        (ROOT / "results/nyu_layer_study/runs/early/initialization.json").read_text()
    )["backbone_sha256"]
    if state_hash(model.state_dict()) != expected:
        raise RuntimeError("Frozen backbone differs from layer-selection experiment")
    loader, selection = analysis_loader(config, limit)
    blank_out = model(torch.zeros(1, 3, 224, 224, device=device), output_hidden_states=True)
    blank = {
        window: {
            f"L{i:02d}": feature_spectrum(patch_map(tokens), window)
            for i, tokens in enumerate(blank_out.hidden_states)
        }
        for window in WINDOWS
    }
    rows = {window: [] for window in WINDOWS}
    started = time.monotonic()
    for index, batch in enumerate(loader):
        output = model(batch["image_clip"].to(device), output_hidden_states=True)
        image_path = batch["image_path"][0]
        maps = [patch_map(tokens) for tokens in output.hidden_states]
        for window in WINDOWS:
            row = {
                "image_path": image_path,
                "scene": scene_group(image_path),
                "spectra": {
                    f"L{i:02d}": feature_spectrum(feature, window) for i, feature in enumerate(maps)
                },
            }
            rows[window].append(row)
        if (index + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "phase": "raw_fourier",
                        "images": index + 1,
                        "total": len(loader),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    summaries = {window: scene_summary(rows[window]) for window in WINDOWS}
    for window in WINDOWS:
        with (directory / f"per_image_{window}.jsonl").open("w") as stream:
            for row in rows[window]:
                stream.write(json.dumps(row) + "\n")
    atomic_json(
        directory / "summary.json",
        {
            "partition": "validation",
            "selection_sha256": selection["selection_sha256"],
            "backbone_sha256": expected,
            "frequency_unit": "cycles per input field",
            "bands": {"low": "0 < r < 2", "mid": "2 <= r < 4", "high": "r >= 4"},
            "windows": summaries,
            "uniform_mean_color_control": blank,
            "uniform_absolute_energy_control": absolute_energy_control(rows, blank),
            "interpretation": "Spatial variation is not a measure of geometric usefulness.",
        },
    )
    plot_raw(summaries, blank, directory)
    atomic_json(
        directory / "complete.json",
        {"completed": True, "n_images": len(loader), "n_scenes": selection["n_scenes"]},
    )
    print(json.dumps({"phase": "raw_complete", "directory": str(directory)}), flush=True)


def plot_perturbation(summary, directory, variant):
    names = ["lowpass", "lowpass_rms", "highpass_rms"]
    labels = ["Low-pass", "Low-pass\nAC energy matched", "High-pass\nAC energy matched"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = np.arange(len(names))
    for offset, region, color in [(-0.18, "boundary", "#D55E00"), (0.18, "interior", "#0072B2")]:
        values = np.array([summary[c][region]["delta_abs_rel_scene_mean"] for c in names])
        bounds = np.array([summary[c][region]["delta_ci95_scene_bootstrap"] for c in names])
        # Bootstrap intervals need not bracket the point estimate exactly.
        ax.bar(x + offset, values, width=0.34, label=region, color=color, alpha=0.85)
        ax.vlines(x + offset, bounds[:, 0], bounds[:, 1], color="#222222", linewidth=1)
        ax.scatter(x + offset, bounds[:, 0], marker="_", color="#222222", s=35)
        ax.scatter(x + offset, bounds[:, 1], marker="_", color="#222222", s=35)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Change in AbsRel versus unmodified model",
        title=f"{variant}: structural-feature frequency perturbation\nValidation scenes · 95% scene bootstrap CI",
    )
    ax.legend()
    fig.savefig(directory / "frequency_perturbation.png", dpi=180)
    fig.savefig(directory / "frequency_perturbation.pdf")
    plt.close(fig)


@torch.no_grad()
def trained_analysis(config, device, limit, smoke_checkpoint):
    variant = config["variant"]
    directory = OUT / ("smoke_trained" if smoke_checkpoint else variant)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "study" / ("smoke" if smoke_checkpoint else "runs") / variant / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from space_clip import SPACECLIP

    model = SPACECLIP(config).to(device).eval()
    before = state_hash(model.clip_vision_model.state_dict())
    if before != saved["backbone_sha256"]:
        raise RuntimeError("Backbone checksum mismatch")
    if set(config["structural_path_indices"]) & set(config["main_path_indices"]):
        raise ValueError("Frequency intervention requires disjoint pathway layer indices")
    if not smoke_checkpoint and saved["config_sha256"] != config_hash(config):
        raise ValueError("Trained-analysis configuration differs from checkpoint")
    load_decoder(model, saved["model"])
    decoder_before = state_hash(decoder_state(model))
    loader, selection = analysis_loader(config, limit)
    all_rows = []
    spectral_rows = {window: [] for window in WINDOWS}
    for index, batch in enumerate(loader):
        inputs = batch["image_clip"].to(device)
        gt = batch["depth"].numpy()[0, 0]
        masks = {"all": valid_depth(gt, crop=config["eval_crop"])}
        masks["boundary"], masks["interior"] = boundary_band(gt, masks["all"], 0.05, 3)
        _, base, intermediates = model(inputs, output_size=gt.shape, return_intermediates=True)
        image_path = batch["image_path"][0]
        scene = scene_group(image_path)
        features = {
            f"semantic_L{layer:02d}": feat[0].detach().cpu().numpy()
            for layer, feat in zip(config["main_path_indices"], intermediates["semantic_projected"])
        }
        features.update(
            {
                f"structural_L{layer:02d}": feat[0].detach().cpu().numpy()
                for layer, feat in zip(
                    config["structural_path_indices"], intermediates["structural_projected"]
                )
            }
        )
        for window in WINDOWS:
            spectral_rows[window].append(
                {
                    "image_path": image_path,
                    "scene": scene,
                    "spectra": {
                        name: feature_spectrum(feat, window) for name, feat in features.items()
                    },
                }
            )
        row = {"image_path": image_path, "scene": scene, "conditions": {}}
        for condition in CONDITIONS:
            if condition == "baseline":
                pred = base
            else:
                mode = (
                    "sham"
                    if condition == "sham"
                    else ("lowpass" if condition.startswith("lowpass") else "highpass")
                )
                match = condition.endswith("_rms")

                def hook(module, args, output):
                    states = list(output.hidden_states)
                    for layer in config["structural_path_indices"]:
                        states[layer] = filter_patch_tokens(states[layer], mode, 4.0, match)
                    output.hidden_states = tuple(states)
                    return output

                handle = model.clip_vision_model.register_forward_hook(hook)
                try:
                    _, pred = model(inputs, output_size=gt.shape)
                finally:
                    handle.remove()
                if condition == "sham":
                    if not torch.allclose(pred, base, atol=1e-5, rtol=1e-4):
                        raise RuntimeError(
                            "FFT round-trip sham altered model prediction beyond tolerance"
                        )
                    row["sham_max_abs_depth_difference"] = float((pred - base).abs().max())
            pred_np = pred.cpu().numpy()[0, 0]
            if not np.isfinite(pred_np).all():
                raise RuntimeError("Nonfinite frequency-perturbed prediction")
            pred_np = np.clip(pred_np, 0.001, 10.0)
            row["conditions"][condition] = {
                name: region_metrics(gt, pred_np, mask) for name, mask in masks.items()
            }
        all_rows.append(row)
        if (index + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "phase": "trained_fourier",
                        "variant": variant,
                        "images": index + 1,
                        "total": len(loader),
                    }
                ),
                flush=True,
            )
    if state_hash(model.clip_vision_model.state_dict()) != before:
        raise RuntimeError("Analysis modified frozen weights")
    if state_hash(decoder_state(model)) != decoder_before:
        raise RuntimeError("Analysis modified decoder state")
    summary = perturbation_summary(all_rows)
    for window in WINDOWS:
        with (directory / f"projected_per_image_{window}.jsonl").open("w") as stream:
            for row in spectral_rows[window]:
                stream.write(json.dumps(row) + "\n")
    with (directory / "perturbation_per_image.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + "\n")
    atomic_json(
        directory / "summary.json",
        {
            "variant": variant,
            "smoke_only": smoke_checkpoint,
            "n_images": len(loader),
            "n_scenes": selection["n_scenes"],
            "selection_sha256": selection["selection_sha256"],
            "partition": "validation",
            "checkpoint_epoch": saved["epoch"] + 1,
            "checkpoint_source": saved["source"],
            "decoder_sha256": decoder_before,
            "backbone_sha256": before,
            "flip_tta": False,
            "frequency_cutoff_cycles_per_input_field": 4,
            "fft_work_dtype": "float64",
            "feature_return_dtype": "float32",
            "sham_atol": 1e-5,
            "sham_rtol": 1e-4,
            "projected_spectra": {w: scene_summary(spectral_rows[w]) for w in WINDOWS},
            "perturbation": summary,
            "max_sham_depth_difference": max(r["sham_max_abs_depth_difference"] for r in all_rows),
            "limitations": [
                "Inference perturbations can be out of distribution.",
                "Hard FFT cutoffs assume periodic boundaries and can ring.",
                "Spatial frequency is not equivalent to geometric information.",
                "Validation scenes also support checkpoint selection; this is exploratory mechanism analysis.",
            ],
        },
    )
    plot_perturbation(summary, directory, variant)
    atomic_json(
        directory / "complete.json",
        {"completed": True, "variant": variant, "smoke_only": smoke_checkpoint},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["raw", "trained"], required=True)
    parser.add_argument("--variant", choices=["early", "middle", "late"], default="early")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--smoke-checkpoint", action="store_true")
    cli = parser.parse_args()
    if cli.max_images is not None and cli.max_images < 1:
        parser.error("--max-images must be positive")
    if cli.phase == "trained" and cli.max_images is not None and not cli.smoke_checkpoint:
        parser.error(
            "Partial trained analysis requires --smoke-checkpoint to protect final results"
        )
    if cli.smoke_checkpoint and (cli.phase != "trained" or cli.max_images is None):
        raise ValueError("Smoke requires a bounded trained-analysis invocation")
    torch.set_num_threads(2)
    seed_everything(42)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if cli.device == "cuda"
        else torch.device("cpu")
    )
    if cli.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.15, device)
    config = load_config(cli.variant)
    if cli.phase == "raw":
        raw_analysis(config, device, cli.max_images)
    else:
        if (
            not cli.smoke_checkpoint
            and not (ROOT / "study/runs" / cli.variant / "training_complete.json").exists()
        ):
            raise RuntimeError("Training has not completed")
        trained_analysis(config, device, cli.max_images, cli.smoke_checkpoint)


if __name__ == "__main__":
    main()
