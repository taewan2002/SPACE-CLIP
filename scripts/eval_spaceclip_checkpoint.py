#!/usr/bin/env python3
"""Evaluate a SPACE-CLIP checkpoint with the same validation logic used in train.py."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train import load_config_from_yaml, safe_collate, setup_device, validate
from utils.dataloader import DepthDataLoader
from utils.loss import GradientLoss, SILogLoss, SSIMLoss
from utils import model_io
from space_clip import SPACECLIP


def _to_json_number(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SPACE-CLIP checkpoint")
    parser.add_argument("--config", required=True, help="YAML config used for the run")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    parser.add_argument(
        "--flip-tta",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override eval_flip_tta at evaluation time (default: auto uses config value).",
    )
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    cfg.distributed = False
    cfg.mode = "train"
    if not hasattr(cfg, "root"):
        cfg.root = "."
    if args.flip_tta == "true":
        cfg.eval_flip_tta = True
    elif args.flip_tta == "false":
        cfg.eval_flip_tta = False

    device = setup_device(cfg)
    model = SPACECLIP(config_model=cfg).to(device)
    model, _, _ = model_io.load_checkpoint(args.checkpoint, model)
    model.eval()

    val_loader = DepthDataLoader(cfg, "online_eval", collate_fn=safe_collate).data
    criterions = (SILogLoss().to(device), SSIMLoss().to(device), GradientLoss().to(device))
    metrics, val_loss = validate(
        model,
        val_loader,
        criterions,
        getattr(cfg, "w_ssim", 0.5),
        getattr(cfg, "w_grad", 0.0),
        cfg,
        device,
        global_step=0,
        fixed_samples=[],
    )

    report = {"val_loss": _to_json_number(val_loss)}
    for key, value in metrics.items():
        report[key] = _to_json_number(value)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
