"""Regression checks for portable setup and archived scientific measurements."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.layer_study.configuration import ROOT, load_config
from scripts.layer_study.prepare import prepare, split_payloads, write_prepared_files
from scripts.layer_study.result_io import read_rows
from scripts.layer_study.verify_results import DEFAULT, verify, verify_checksums


def setup_checkout(tmp_path):
    for directory in ("configs/layer_study", "train_test_inputs"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    target = tmp_path / "results/nyu_layer_study"
    target.mkdir(parents=True)
    shutil.copy2(DEFAULT / "split_manifest.json", target / "split_manifest.json")
    return tmp_path


def test_split_bytes_match_archived_protocol():
    payloads, manifest = split_payloads()
    assert {name: len(content.splitlines()) for name, content in payloads.items()} == {
        "train": 21974,
        "validation": 2257,
        "test": 654,
    }
    assert {
        name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()
    } == manifest["split_sha256"]


def test_portable_setup_is_idempotent_and_preserves_templates(tmp_path):
    root = setup_checkout(tmp_path)
    template = root / "configs/layer_study/early.yaml"
    original = template.read_bytes()
    prepare(root, tmp_path / "my-data/train", tmp_path / "my-data/test")
    config = load_config("early", root)
    assert config["data_path"] == str(tmp_path / "my-data/train")
    assert (
        Path(config["filenames_file_test"]).read_bytes()
        == (root / "train_test_inputs/nyudepthv2_test_files_with_gt.txt").read_bytes()
    )
    prepare(root, tmp_path / "my-data/train", tmp_path / "my-data/test")
    assert template.read_bytes() == original


def test_missing_dataset_audit_does_not_write_partial_setup(tmp_path):
    root = setup_checkout(tmp_path)
    with pytest.raises(FileNotFoundError, match="dataset entries missing"):
        prepare(root, audit=True)
    assert not (root / "study").exists()


def test_changed_setup_cannot_overwrite_started_run(tmp_path):
    study = tmp_path / "study"
    original = {"configs/early.yaml": b"original", "splits/test.txt": b"test"}
    write_prepared_files(study, original)
    checkpoint = study / "runs/early/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing checkpoint")
    write_prepared_files(study, original)
    with pytest.raises(ValueError, match="Cannot change setup"):
        write_prepared_files(study, {**original, "configs/early.yaml": b"changed"})
    assert (study / "configs/early.yaml").read_bytes() == b"original"
    assert checkpoint.read_bytes() == b"existing checkpoint"


def test_source_list_corruption_fails_before_writing(tmp_path):
    root = setup_checkout(tmp_path)
    (root / "train_test_inputs/nyudepthv2_test_files_with_gt.txt").write_text("incorrect\n")
    with pytest.raises(ValueError, match="Official test list differs"):
        prepare(root)
    assert not (root / "study").exists()


def test_missing_configuration_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts.layer_study.prepare"):
        load_config("early", tmp_path)


def test_plain_and_compressed_results_are_equivalent(tmp_path):
    path = tmp_path / "rows.jsonl"
    content = b'{"a": 1}\n\n{"a": 2}\n'
    path.write_bytes(content)
    plain = read_rows(path)
    path.unlink()
    path.with_suffix(".jsonl.gz").write_bytes(gzip.compress(content, mtime=0))
    assert read_rows(path) == plain == [{"a": 1}, {"a": 2}]


def test_checksum_corruption_is_detected(tmp_path):
    path = tmp_path / "sample.json"
    path.write_bytes(b"{}")
    (tmp_path / "checksums.json").write_text(
        json.dumps({"sample.json": hashlib.sha256(b"{}").hexdigest()})
    )
    assert verify_checksums(tmp_path) == 1
    path.write_bytes(b"{ }")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path)


def test_archived_measurements_reproduce_reported_aggregates():
    result = verify()
    assert result["zero_difference_sham_checks"] == 600
