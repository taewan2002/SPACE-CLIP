import numpy as np
from scripts.layer_study.metrics import aggregate, boundary_band, evaluate_image
from scripts.layer_study.prepare import scene_group, split_scenes


def test_missing_depth_does_not_create_boundary():
    gt = np.full((32, 32), 2.0)
    gt[12:20, 12:20] = 0
    valid = gt > 0
    band, interior = boundary_band(gt, valid)
    assert not band.any()
    assert not interior[12:20, 12:20].any()


def test_boundary_detects_depth_jump_and_excludes_borders():
    gt = np.ones((32, 32))
    gt[:, 16:] = 2
    band, interior = boundary_band(gt, np.ones_like(gt, dtype=bool))
    assert band[16, 15] and band[16, 16]
    assert not band[:3].any()
    assert interior[16, 5]


def test_boundary_error_distinguishes_shifted_edge():
    gt = np.ones((32, 32))
    gt[:, 16:] = 2
    pred = gt.copy()
    pred[:, 16:19] = 1
    values = evaluate_image(gt, pred, crop="none")
    assert values["boundary_0.05"]["mae"] > 0
    assert values["interior_0.05"]["mae"] == 0


def test_perfect_prediction_and_no_boundary_denominator():
    gt = np.ones((32, 32))
    values = evaluate_image(gt, gt, crop="none")
    result = aggregate([{"metrics": values}])
    assert result["regions"]["all"]["rmse"] == 0
    assert result["regions"]["boundary_0.05"]["n_images"] == 0
    assert "rmse" not in result["regions"]["boundary_0.05"]


def test_scene_suffixes_stay_in_same_partition():
    lines = [
        f"/kitchen_{i:04d}{suffix}/rgb.jpg /x.png 1" for i in range(20) for suffix in ("a", "b")
    ]
    train, val = split_scenes(lines)
    assert set(map(scene_group, train)).isdisjoint(set(map(scene_group, val)))
    assert train + val != []
    assert set(train + val) == set(lines)
    assert split_scenes(lines) == (train, val)


def test_nonfinite_predictions_fail():
    import pytest

    gt = np.ones((16, 16))
    pred = gt.copy()
    pred[5, 5] = np.nan
    with pytest.raises(ValueError):
        evaluate_image(gt, pred, crop="none")
