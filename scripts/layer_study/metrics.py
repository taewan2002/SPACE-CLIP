"""Depth metrics conditioned on ground-truth discontinuities; no RGB edges."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def valid_depth(gt, minimum=0.001, maximum=10.0, crop="eigen"):
    valid = np.isfinite(gt) & (gt > minimum) & (gt < maximum)
    if crop == "eigen":
        region = np.zeros_like(valid)
        region[45:471, 41:601] = True
        valid &= region
    elif crop != "none":
        raise ValueError(crop)
    return valid


def boundary_band(gt, valid, threshold=0.05, radius=3):
    """Mark both sides of valid adjacent log-depth jumps, then dilate.

    The complete radius-neighborhood must be valid, excluding missing-depth
    holes and crop/image borders. threshold=0.05 means a depth ratio > exp(.05).
    """
    gt = np.asarray(gt, dtype=np.float64)
    log_gt = np.log(np.maximum(gt, 1e-12))
    edge = np.zeros_like(valid, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1]
    horizontal &= np.abs(log_gt[:, 1:] - log_gt[:, :-1]) > threshold
    vertical = valid[1:, :] & valid[:-1, :]
    vertical &= np.abs(log_gt[1:, :] - log_gt[:-1, :]) > threshold
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    safe = binary_erosion(valid, structure=structure, border_value=0)
    band = binary_dilation(edge, structure=structure) & safe
    return band, safe & ~band


def region_metrics(gt, pred, mask):
    if not mask.any():
        return {"pixels": 0}
    g = np.asarray(gt[mask], dtype=np.float64)
    p = np.asarray(pred[mask], dtype=np.float64)
    error = p - g
    log_error = np.log(p) - np.log(g)
    ratio = np.maximum(g / p, p / g)
    return {
        "pixels": int(mask.sum()),
        "abs_rel": float(np.mean(np.abs(error) / g)),
        "sq_rel": float(np.mean(error**2 / g)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "a1": float(np.mean(ratio < 1.25)),
        "a2": float(np.mean(ratio < 1.25**2)),
        "a3": float(np.mean(ratio < 1.25**3)),
    }


def evaluate_image(gt, pred, crop="eigen"):
    gt, pred = np.asarray(gt).squeeze(), np.asarray(pred).squeeze()
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError(f"Expected matching 2D arrays, got {gt.shape}/{pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite prediction")
    valid = valid_depth(gt, crop=crop)
    if not valid.any():
        raise ValueError("No valid ground-truth depth")
    pred = np.clip(pred, 0.001, 10.0)  # No ground-truth median scaling.
    result = {"all": region_metrics(gt, pred, valid)}
    for threshold in (0.03, 0.05, 0.10):
        band, interior = boundary_band(gt, valid, threshold=threshold)
        suffix = f"{threshold:.2f}"
        result[f"boundary_{suffix}"] = region_metrics(gt, pred, band)
        result[f"interior_{suffix}"] = region_metrics(gt, pred, interior)
    return result


def aggregate(rows):
    """Macro-average images; disclose separate denominators for empty regions."""
    if not rows:
        raise ValueError("No evaluated images")
    result = {"n_images": len(rows), "regions": {}}
    for region in rows[0]["metrics"]:
        available = [r["metrics"][region] for r in rows if r["metrics"][region]["pixels"]]
        values = {"n_images": len(available), "n_pixels": sum(v["pixels"] for v in available)}
        if available:
            for metric in available[0]:
                if metric != "pixels":
                    values[metric] = float(np.mean([v[metric] for v in available]))
        result["regions"][region] = values
    return result
