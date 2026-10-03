"""Validate the component comparison against per-image and selection records."""

import json
import shutil

import pytest

from scripts.component_study.build import build
from scripts.component_study.verify import DATA, REFERENCE, recompute, verify


def test_completed_component_archive():
    result = verify()
    assert result["total_runs"] == 20
    assert result["test_images_per_run"] == 654


def test_test_checkpoint_must_match_validation_history(tmp_path):
    data = tmp_path / "archive"
    shutil.copytree(DATA, data)
    path = data / "42/test_metrics.json"
    meta = json.loads(path.read_text())
    meta["checkpoint_epoch"] = 20
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="validation minimum"):
        recompute(data)


def test_component_builder_preserves_input_archives():
    for output in (DATA, DATA / "generated", REFERENCE / "generated"):
        with pytest.raises(ValueError, match="must not overwrite"):
            build(output=output)
