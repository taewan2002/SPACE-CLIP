"""Load machine-local study configurations produced by prepare.py."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"early": [2, 1, 0], "middle": [7, 5, 4], "late": [11, 10, 8]}
PATH_KEYS = (
    "data_path",
    "gt_path",
    "data_path_eval",
    "gt_path_eval",
    "data_path_test",
    "gt_path_test",
    "filenames_file",
    "filenames_file_eval",
    "filenames_file_test",
)


def load_config(variant, root=ROOT):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    path = Path(root) / "study/configs" / f"{variant}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            "Run python -m scripts.layer_study.prepare --train-root PATH "
            "--test-root PATH --audit before running experiments."
        )
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("variant") != variant:
        raise ValueError(f"Invalid study configuration: {path}")
    if config.get("structural_path_indices") != VARIANTS[variant]:
        raise ValueError("Layer indices differ from the declared comparison")
    if config.get("main_path_indices") != [12, 9, 6, 3]:
        raise ValueError("Main-path layers must remain fixed across arms")
    for key in PATH_KEYS:
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError(f"Expected an absolute local path for {key}; rerun prepare")
    return config
