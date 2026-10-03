"""Export predictions for samples fixed before the final test results."""

from pathlib import Path
import sys
import json
import os
import hashlib
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from space_clip import SPACECLIP
from scripts.layer_study.configuration import load_config
from scripts.layer_study.run import (
    config_hash,
    load_decoder,
    make_loader,
    seed_everything,
    state_hash,
)
from scripts.layer_study.metrics import evaluate_image, valid_depth
from torch.utils.data import Subset, DataLoader
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    policy = json.loads(
        (ROOT / "results/nyu_layer_study/qualitative_sample_policy.json").read_text()
    )
    out = ROOT / "study/qualitative"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.15, device)
    torch.set_num_threads(2)
    seed_everything(42)
    data = {}
    audit = {"selection": policy, "variants": {}}
    for variant in ["early", "middle", "late"]:
        cfg = load_config(variant)
        lines = Path(cfg["filenames_file_test"]).read_text().splitlines()
        assert (
            hashlib.sha256(Path(cfg["filenames_file_test"]).read_bytes()).hexdigest()
            == policy["test_split_sha256"]
        )
        chosen = [lines.index(row) for row in policy["selected_rows"]]
        dataset = make_loader(cfg, "test").dataset
        loader = DataLoader(Subset(dataset, chosen), batch_size=1, shuffle=False, num_workers=0)
        model = SPACECLIP(cfg).to(device).eval()
        saved = torch.load(
            ROOT / f"study/runs/{variant}/best.pt", map_location="cpu", weights_only=False
        )
        assert saved["config_sha256"] == config_hash(cfg)
        load_decoder(model, saved["model"])
        assert state_hash(model.clip_vision_model.state_dict()) == saved["backbone_sha256"]
        expected = {
            r["image_path"]: r
            for r in [
                json.loads(x)
                for x in (ROOT / f"study/runs/{variant}/test_per_image.jsonl")
                .read_text()
                .splitlines()
            ]
        }
        audit["variants"][variant] = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["image_clip"].to(device)
                gt = batch["depth"].numpy()[0, 0]
                _, pred = model(x, output_size=gt.shape)
                _, flipped = model(torch.flip(x, [-1]), output_size=gt.shape)
                pred = ((pred + torch.flip(flipped, [-1])) / 2).cpu().numpy()[0, 0]
                metrics = evaluate_image(gt, pred, crop=cfg["eval_crop"])
                path = batch["image_path"][0]
                difference = abs(
                    metrics["all"]["abs_rel"] - expected[path]["metrics"]["all"]["abs_rel"]
                )
                assert difference < 1e-6, (variant, path, difference)
                data[f"{variant}_{i}"] = np.clip(pred, 0.001, 10).astype(np.float32)
                if variant == "early":
                    data[f"rgb_{i}"] = np.asarray(
                        Image.open(Path(cfg["data_path_test"]) / path).convert("RGB")
                    )
                    data[f"gt_{i}"] = gt.astype(np.float32)
                    data[f"mask_{i}"] = valid_depth(gt, crop=cfg["eval_crop"])
                audit["variants"][variant].append(
                    {"image_path": path, "abs_rel_reproduction_difference": difference}
                )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(out / "predictions.npz", **data)
    (out / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"completed": True, "selected_images": 5, "variants": 3}))


if __name__ == "__main__":
    main()
