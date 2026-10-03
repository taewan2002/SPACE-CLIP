"""Check the complete archive and protect independent training runs."""

import pytest

from scripts.layer_study.configuration import load_config
from scripts.layer_study.prepare import prepare
from scripts.layer_study.test_release import setup_checkout
from scripts.multiseed.aggregate import DATA
from scripts.multiseed.build import build
from scripts.multiseed.verify import verify


def test_five_seed_archive_reproduces_manuscript():
    report = verify()
    assert report["runs"] == 15
    assert report["test_images_per_run"] == 654
    assert report["zero_difference_sham_checks"] == 3000


def test_seed_configuration_cannot_replace_an_existing_run(tmp_path):
    root = setup_checkout(tmp_path)
    prepare(root, seed=43)
    for variant in ("early", "middle", "late"):
        assert load_config(variant, root)["random_seed"] == 43
    checkpoint = root / "study/runs/early/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing weights")
    with pytest.raises(ValueError, match="Cannot change setup"):
        prepare(root, seed=44)
    assert load_config("early", root)["random_seed"] == 43
    assert checkpoint.read_bytes() == b"existing weights"


def test_builder_cannot_overwrite_measurements():
    for output in (DATA, DATA / "generated"):
        with pytest.raises(ValueError, match="must not overwrite"):
            build(output=output)
