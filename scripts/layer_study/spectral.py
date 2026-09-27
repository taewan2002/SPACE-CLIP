"""Spatial feature spectra on native 14x14 grids, with exact FFT frequencies."""

import numpy as np


def frequency_radius(height, width):
    """Cycles across the input field of view; diagonal frequencies are retained."""
    fy = np.fft.fftfreq(height, d=1.0 / height)
    fx = np.fft.fftfreq(width, d=1.0 / width)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def feature_spectrum(feature, window="hann"):
    """Channel-summed AC energy, normalized separately for each image/feature.

    Do not average signed channels before FFT. Annular energies are sums, not
    sums of annular means. Each returned profile sums to one for nonzero AC.
    """
    feature = np.asarray(feature, dtype=np.float64)
    if feature.ndim != 3:
        raise ValueError("Expected C,H,W for one image")
    channels, height, width = feature.shape
    if min(height, width) < 3 or not np.isfinite(feature).all():
        raise ValueError("Feature is too small or contains nonfinite values")
    if window == "hann":
        taper = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    elif window == "rectangular":
        taper = np.ones((height, width))
    else:
        raise ValueError(window)
    # Weighted mean removes DC even after application of the taper.
    means = (feature * taper).sum(axis=(-2, -1), keepdims=True) / taper.sum()
    centered = (feature - means) * taper
    coefficients = np.fft.fft2(centered, axes=(-2, -1), norm="ortho")
    power = (np.abs(coefficients) ** 2).sum(axis=0)
    power[0, 0] = 0.0
    radius = frequency_radius(height, width)
    edges = np.arange(0, np.ceil(radius.max()) + 1, dtype=float)
    if edges[-1] <= radius.max():
        edges = np.append(edges, edges[-1] + 1)
    profile = np.histogram(radius, bins=edges, weights=power)[0]
    mode_counts = np.histogram(radius[radius > 0], bins=edges)[0]
    total = float(power.sum())
    if total <= 1e-20:
        return {
            "active": False,
            "ac_energy": total,
            "edges": edges.tolist(),
            "mode_counts": mode_counts.tolist(),
            "profile": np.zeros_like(profile).tolist(),
            "bands": {"low": 0.0, "mid": 0.0, "high": 0.0},
        }
    masks = {
        "low": (radius > 0) & (radius < 2),
        "mid": (radius >= 2) & (radius < 4),
        "high": radius >= 4,
    }
    return {
        "active": True,
        "channels": channels,
        "grid": [height, width],
        "ac_energy": total,
        "windowed_spatial_energy": float((centered**2).sum()),
        "edges": edges.tolist(),
        "mode_counts": mode_counts.tolist(),
        "profile": (profile / total).tolist(),
        "bands": {name: float(power[mask].sum() / total) for name, mask in masks.items()},
    }


def filter_patch_tokens(tokens, mode, cutoff=4.0, match_rms=False):
    """Temporary inference perturbation; CLS and channel means are preserved.

    Hard radial masks assume periodic boundaries and can cause ringing; results
    are sensitivity diagnostics, not evidence of an intrinsic causal mechanism.
    """
    import torch

    patch = tokens[:, 1:, :]
    batch, count, channels = patch.shape
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError("Expected a square patch grid plus one CLS token")
    # Use float64 for the transform, then restore the original tensor dtype.
    # Float32 FFT roundoff can be amplified by a trained nonlinear decoder.
    feature = patch.transpose(1, 2).reshape(batch, channels, side, side).double()
    means = feature.mean(dim=(-2, -1), keepdim=True)
    residual = feature - means
    spectrum = torch.fft.fft2(residual, norm="ortho")
    radius = torch.as_tensor(frequency_radius(side, side), device=tokens.device)
    if mode == "sham":
        mask = torch.ones_like(radius, dtype=torch.bool)
    elif mode == "lowpass":
        mask = radius < cutoff
    elif mode == "highpass":
        mask = radius >= cutoff
    else:
        raise ValueError(mode)
    filtered = torch.fft.ifft2(spectrum * mask, norm="ortho").real
    # Correct numerical mean drift so channel means stay fixed.
    filtered = filtered - filtered.mean(dim=(-2, -1), keepdim=True)
    if match_rms:
        original_energy = residual.square().sum(dim=(1, 2, 3), keepdim=True)
        retained_energy = filtered.square().sum(dim=(1, 2, 3), keepdim=True)
        scale = torch.sqrt(original_energy / retained_energy.clamp_min(1e-20))
        scale = torch.where(retained_energy > 1e-20, scale, torch.zeros_like(scale))
        filtered = filtered * scale
    filtered = (filtered + means).reshape(batch, channels, count).transpose(1, 2)
    return torch.cat([tokens[:, :1, :], filtered.to(tokens.dtype)], dim=1)


def scene_summary(rows, feature_key="spectra", bootstrap_seed=20260926):
    """Average frames within scene, then average equally weighted scene means."""
    names = sorted(rows[0][feature_key])
    groups = sorted({row["scene"] for row in rows})
    result = {"n_images": len(rows), "n_scenes": len(groups), "features": {}}
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    for name in names:
        by_scene = []
        profiles = []
        active_images = 0
        for group in groups:
            spectra = [
                r[feature_key][name]
                for r in rows
                if r["scene"] == group and r[feature_key][name]["active"]
            ]
            if not spectra:
                continue
            active_images += len(spectra)
            by_scene.append(
                [np.mean([s["bands"][b] for s in spectra]) for b in ("low", "mid", "high")]
            )
            profiles.append(np.mean([s["profile"] for s in spectra], axis=0))
        if not by_scene:
            result["features"][name] = {"n_active_images": 0}
            continue
        array = np.asarray(by_scene)
        local_draws = (
            draws
            if len(array) == len(groups)
            else rng.integers(0, len(array), size=(2000, len(array)))
        )
        boot = array[local_draws].mean(axis=1)
        result["features"][name] = {
            "n_active_images": active_images,
            "n_active_scenes": len(array),
            "profile": np.mean(profiles, axis=0).tolist(),
            "bands": {
                band: {
                    "mean": float(array[:, i].mean()),
                    "ci95_scene_bootstrap": np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
                }
                for i, band in enumerate(("low", "mid", "high"))
            },
        }
    return result


def perturbation_summary(rows):
    groups = sorted({r["scene"] for r in rows})
    result = {}
    for condition in ("baseline", "sham", "lowpass", "lowpass_rms", "highpass_rms"):
        result[condition] = {}
        for region in ("all", "boundary", "interior"):
            paired = []
            absolute = []
            for group in groups:
                samples = [
                    r
                    for r in rows
                    if r["scene"] == group and r["conditions"][condition][region]["pixels"] > 0
                ]
                if not samples:
                    continue
                absolute.append(
                    np.mean([r["conditions"][condition][region]["abs_rel"] for r in samples])
                )
                paired.append(
                    np.mean(
                        [
                            r["conditions"][condition][region]["abs_rel"]
                            - r["conditions"]["baseline"][region]["abs_rel"]
                            for r in samples
                        ]
                    )
                )
            if not paired:
                result[condition][region] = {"n_scenes": 0}
                continue
            array = np.asarray(paired)
            rng = np.random.default_rng(20260926)
            draws = rng.integers(0, len(array), size=(2000, len(array)))
            result[condition][region] = {
                "n_scenes": len(array),
                "abs_rel_scene_mean": float(np.mean(absolute)),
                "delta_abs_rel_scene_mean": float(array.mean()),
                "delta_ci95_scene_bootstrap": np.quantile(
                    array[draws].mean(1), [0.025, 0.975]
                ).tolist(),
            }
    return result
