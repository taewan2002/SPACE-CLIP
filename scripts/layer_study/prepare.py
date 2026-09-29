"""Recreate the fixed scene split and prepare machine-local configurations."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

from scripts.layer_study.configuration import ROOT, VARIANTS


def scene_group(line):
    scene = line.split()[0].lstrip("/").split("/")[0]
    return re.sub(r"[a-z]$", "", scene)


def split_scenes(lines, seed=20260926, fraction=0.1):
    groups = sorted({scene_group(line) for line in lines})
    if len(groups) < 2:
        raise ValueError("At least two scene groups required")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    held_out = set(groups[:count])
    return (
        [x for x in lines if scene_group(x) not in held_out],
        [x for x in lines if scene_group(x) in held_out],
    )


def digest(content):
    return hashlib.sha256(content).hexdigest()


def split_payloads(root=ROOT):
    """Check source identity, preserve official test bytes, and verify all splits."""
    root = Path(root)
    manifest = json.loads((root / "results/nyu_layer_study/split_manifest.json").read_text())
    sources = {}
    for partition in ("train", "test"):
        source = root / f"train_test_inputs/nyudepthv2_{partition}_files_with_gt.txt"
        sources[partition] = source.read_bytes()
        if digest(sources[partition]) != manifest[f"source_{partition}_sha256"]:
            raise ValueError(f"Official {partition} list differs from the archived study")
    lines = [x.strip() for x in sources["train"].decode().splitlines() if x.strip()]
    if len({x.split()[0] for x in lines}) != len(lines):
        raise ValueError("Duplicate training image paths")
    train, validation = split_scenes(lines)
    payloads = {
        "train": ("\n".join(train) + "\n").encode(),
        "validation": ("\n".join(validation) + "\n").encode(),
        "test": sources["test"],
    }
    for name, content in payloads.items():
        if digest(content) != manifest["split_sha256"][name]:
            raise ValueError(f"{name} split differs from the archived study")
    return payloads, manifest


def write_prepared_files(study, payloads):
    """Reject protocol changes before writing anything once a run has started."""
    study = Path(study)
    started = any(
        p.is_file() for name in ("smoke", "runs", "fourier") for p in (study / name).rglob("*")
    )
    changed = [
        name
        for name, content in payloads.items()
        if not (study / name).is_file() or (study / name).read_bytes() != content
    ]
    if started and changed:
        raise ValueError(
            "Cannot change setup after a run has started. Use a fresh checkout; "
            f"changed files: {', '.join(changed)}"
        )
    for name in changed:
        path = study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payloads[name])
        temporary.replace(path)


def prepare(root=ROOT, train_root=None, test_root=None, audit=False, seed=42):
    if seed not in range(42, 47):
        raise ValueError("The released five-seed protocol uses seeds 42 through 46")
    root = Path(root).resolve()
    default = root / "datasets/kitti_nyu/nyu_depth_v2"
    train_root = Path(train_root or default / "sync").expanduser().resolve()
    test_root = Path(test_root or default / "official_splits/test").expanduser().resolve()
    splits, manifest = split_payloads(root)
    payloads = {f"splits/{name}.txt": content for name, content in splits.items()}
    payloads["split_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for variant in VARIANTS:
        config = yaml.safe_load((root / f"configs/layer_study/{variant}.yaml").read_text())
        config["random_seed"] = seed
        config["root"] = str(root)
        for key in ("data_path", "gt_path", "data_path_eval", "gt_path_eval"):
            config[key] = str(train_root)
        for key in ("data_path_test", "gt_path_test"):
            config[key] = str(test_root)
        for key, name in (
            ("filenames_file", "train"),
            ("filenames_file_eval", "validation"),
            ("filenames_file_test", "test"),
        ):
            config[key] = str(root / f"study/splits/{name}.txt")
        payloads[f"configs/{variant}.yaml"] = yaml.safe_dump(config, sort_keys=False).encode()
    if audit:
        missing = []
        for partition, content in splits.items():
            base = test_root if partition == "test" else train_root
            for line in content.decode().splitlines():
                for rel in line.split()[:2]:
                    path = base / rel.lstrip("/")
                    if not path.is_file():
                        missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} dataset entries missing; first examples: {missing[:5]}"
            )
    write_prepared_files(root / "study", payloads)
    return manifest["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, help="NYU sync directory for train/validation")
    parser.add_argument("--test-root", type=Path, help="Official 654-image test directory")
    parser.add_argument("--audit", action="store_true", help="Check RGB/depth files before writing")
    parser.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                train_root=args.train_root,
                test_root=args.test_root,
                audit=args.audit,
                seed=args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()
