"""Offline regressions for the reduced decoder and controlled initialization."""

from copy import deepcopy
import json

import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from space_clip import SPACECLIP
from scripts.component_study.initialization import copy_shared_initialization
from scripts.component_study.run import require_completed_controls


@pytest.fixture
def models(monkeypatch):
    torch.manual_seed(42)
    backbone = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=12,
            num_attention_heads=4,
            image_size=4,
            patch_size=2,
        )
    )
    monkeypatch.setattr(SPACECLIP, "_load_vision_backbone", lambda *_: deepcopy(backbone))
    config = {
        "decoder_channels": [16, 8, 4, 2],
        "film_param_hidden_dim": 8,
        "use_multiscale_supervision": True,
        "decoder_dropout": 0.0,
    }
    return SPACECLIP(config), SPACECLIP(dict(config, use_structural_pathway=False))


def test_semantic_only_forward_backward_with_auxiliary_heads(models):
    _, control = models
    auxiliary, depth, features = control(
        torch.randn(2, 3, 4, 4), output_size=(8, 10), return_intermediates=True
    )
    assert depth.shape == (2, 1, 8, 10)
    assert len(auxiliary) == 3
    assert features["structural_decoder"] == ()
    assert features["structural_projected"] == ()
    assert len(features["semantic_decoder"]) == 4
    predictions = [*auxiliary, depth]
    assert all(torch.isfinite(prediction).all() for prediction in predictions)
    sum((prediction - 1).square().mean() for prediction in predictions).backward()
    assert control.main_path_projections[0].weight.grad is not None
    assert all(p.grad is None for p in control.clip_vision_model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in control.parameters() if p.grad is not None)


def test_shared_initialization_copies_weights_and_slices_only_fusion_inputs(models):
    full, control = models
    audit = copy_shared_initialization(full, control)
    source = full.state_dict()
    for name, value in control.state_dict().items():
        expected = source[name]
        if name in audit["fusion_input_slices"]:
            expected = expected[:, : value.shape[1]]
        assert torch.equal(value, expected), name
        assert value.data_ptr() != source[name].data_ptr(), name
    assert len(audit["fusion_input_slices"]) == 3
    assert sum(p.numel() for p in control.parameters()) < sum(p.numel() for p in full.parameters())


def test_shared_initialization_rejects_unexpected_architecture(models):
    full, control = models
    control.depth_prediction_head[3] = torch.nn.Conv2d(1, 2, 3, padding=1)
    with pytest.raises(ValueError, match="Unexpected control shape"):
        copy_shared_initialization(full, control)


@pytest.fixture
def completed_controls(tmp_path):
    roots = [tmp_path / f"seed-{seed}" for seed in range(42, 47)]
    for seed, root in zip(range(42, 47), roots):
        output = root / "study/runs/main_only"
        output.mkdir(parents=True)
        (output / "effective_config.json").write_text(
            json.dumps(
                dict(
                    variant="main_only",
                    use_structural_pathway=False,
                    use_film=True,
                    epochs=20,
                    random_seed=seed,
                )
            )
        )
        (output / "training_complete.json").write_text(
            json.dumps(
                dict(
                    completed=True, backbone_unchanged=True, official_test_used_for_selection=False
                )
            )
        )
        (output / "history.jsonl").write_text(
            "\n".join(json.dumps({"epoch": epoch}) for epoch in range(1, 21))
        )
        (output / "best.pt").touch()
    return roots


def test_test_gate_accepts_all_completed_seeds(completed_controls):
    require_completed_controls(completed_controls, completed_controls[0])


@pytest.mark.parametrize(
    "fault", ["missing", "partial", "duplicate_seed", "smoke", "no_checkpoint"]
)
def test_test_gate_rejects_incomplete_groups(completed_controls, fault):
    output = completed_controls[-1] / "study/runs/main_only"
    if fault == "missing":
        (output / "training_complete.json").unlink()
    elif fault == "partial":
        (output / "history.jsonl").write_text('{"epoch": 1}\n')
    elif fault == "no_checkpoint":
        (output / "best.pt").unlink()
    else:
        path = output / "effective_config.json"
        config = json.loads(path.read_text())
        config["random_seed" if fault == "duplicate_seed" else "epochs"] = (
            42 if fault == "duplicate_seed" else 1
        )
        path.write_text(json.dumps(config))
    with pytest.raises((ValueError, FileNotFoundError)):
        require_completed_controls(completed_controls, completed_controls[0])


def test_test_gate_rejects_duplicate_checkout(completed_controls):
    with pytest.raises(ValueError, match="distinct"):
        require_completed_controls([completed_controls[0]] * 5, completed_controls[0])
