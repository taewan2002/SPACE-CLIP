"""Controlled SPACE-CLIP layer study with validation-only checkpoint selection."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from space_clip import SPACECLIP
from utils.dataloader import DataLoadPreprocess, preprocessing_transforms
from utils.loss import SILogLoss, SSIMLoss
from scripts.layer_study.configuration import load_config
from scripts.layer_study.metrics import aggregate, evaluate_image, region_metrics, valid_depth

BACKBONE_PREFIX = "clip_vision_model."


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class StrictDepthDataset(DataLoadPreprocess):
    def _load_images(self, img_rel_path, depth_rel_path):
        image, depth = super()._load_images(img_rel_path, depth_rel_path)
        if image is None or depth is None:
            raise FileNotFoundError(
                f"Missing or unreadable RGB/depth: {img_rel_path}, {depth_rel_path}"
            )
        return image, depth

    def _preprocess_eval_test(self, image, depth):
        # Keep native ground-truth resolution. Model prediction is resized to GT.
        image_np, depth_np = self._common_pil_to_numpy_and_scale_depth(image, depth)
        return image_np, depth_np, True

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if sample is None or not (sample["depth"] > 0).any():
            raise ValueError(f"Invalid sample at index {index}")
        return sample


def make_loader(config, partition, epoch=0, limit=None):
    cfg = dict(config)
    mode = "train" if partition == "train" else "online_eval"
    if partition == "test":
        cfg["data_path_eval"] = cfg["data_path_test"]
        cfg["gt_path_eval"] = cfg["gt_path_test"]
        cfg["filenames_file_eval"] = cfg["filenames_file_test"]
    args = argparse.Namespace(**cfg)
    dataset = StrictDepthDataset(args, mode, preprocessing_transforms(args, mode))
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    train = partition == "train"
    generator = torch.Generator().manual_seed(config["random_seed"] + epoch)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"] if train else 1,
        shuffle=train,
        num_workers=config["workers"] if train else config["eval_workers"],
        pin_memory=True,
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def decoder_state(model, cpu=False):
    return {
        key: (value.detach().cpu().clone() if cpu else value.detach().clone())
        for key, value in model.state_dict().items()
        if not key.startswith(BACKBONE_PREFIX)
    }


def load_decoder(model, state):
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        not key.startswith(BACKBONE_PREFIX) for key in result.missing_keys
    ):
        raise RuntimeError(f"Incompatible decoder checkpoint: {result}")


def state_hash(state):
    result = hashlib.sha256()
    for key in sorted(state):
        result.update(key.encode())
        result.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def trainable_hash(model):
    return state_hash({k: v for k, v in model.named_parameters() if v.requires_grad})


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class DecoderEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = decoder_state(model)

    @torch.no_grad()
    def update(self, model):
        for key, tensor in model.state_dict().items():
            if key.startswith(BACKBONE_PREFIX):
                continue
            if tensor.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(tensor, alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(tensor)


@torch.no_grad()
def evaluate(model, loader, config, device, detailed=False, flip=False):
    model.eval()
    rows = []
    started = time.monotonic()
    for index, batch in enumerate(loader):
        inputs = batch["image_clip"].to(device)
        gt_tensor = batch["depth"]
        _, pred = model(inputs, output_size=gt_tensor.shape[-2:])
        if flip:
            _, flipped = model(torch.flip(inputs, [-1]), output_size=gt_tensor.shape[-2:])
            pred = (pred + torch.flip(flipped, [-1])) / 2
        pred_np = pred.float().cpu().numpy()[0, 0]
        gt_np = gt_tensor.numpy()[0, 0]
        if detailed:
            values = evaluate_image(gt_np, pred_np, crop=config["eval_crop"])
        else:
            if not np.isfinite(pred_np).all():
                raise ValueError("Nonfinite validation prediction")
            mask = valid_depth(gt_np, crop=config["eval_crop"])
            if not mask.any():
                raise ValueError("Validation sample has no valid GT")
            values = {"all": region_metrics(gt_np, np.clip(pred_np, 0.001, 10.0), mask)}
        rows.append({"image_path": batch["image_path"][0], "metrics": values})
        if (index + 1) % 200 == 0:
            print(
                json.dumps(
                    {
                        "event": "evaluation",
                        "images": index + 1,
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
    return aggregate(rows), rows


def train(config, output, device, smoke=False):
    seed_everything(config["random_seed"])
    model = SPACECLIP(config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    initial_hash = trainable_hash(model)
    backbone_before = state_hash(model.clip_vision_model.state_dict())
    initial = {
        "config_sha256": config_hash(config),
        "variant": config["variant"],
        "structural_layers": config["structural_path_indices"],
        "main_layers": config["main_path_indices"],
        "trainable_parameters": sum(p.numel() for p in params),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "initial_trainable_sha256": initial_hash,
        "backbone_sha256": backbone_before,
        "smoke_only": smoke,
        "device": torch.cuda.get_device_name(device),
        "precision": "float32",
        "micro_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "batchnorm_batch_size": config["batch_size"],
    }
    atomic_json(output / "initialization.json", initial)
    optimizer = torch.optim.AdamW(
        params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    loss_silog, loss_ssim = SILogLoss(), SSIMLoss()
    ema = DecoderEMA(model, config["ema_decay"])
    accumulation = config["gradient_accumulation_steps"]
    max_train = config["batch_size"] * accumulation if smoke else None
    max_val = 4 if smoke else None
    first_loader = make_loader(config, "train", limit=max_train)
    updates_per_epoch = math.ceil(len(first_loader) / accumulation)
    total_updates = updates_per_epoch * config["epochs"]
    warmup = max(1, int(total_updates * config["warmup_ratio"]))
    minimum = config["min_lr"] / config["learning_rate"]

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_updates - warmup)))
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, best, updates = 0, float("inf"), 0
    last = output / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["config_sha256"] != config_hash(config):
            raise ValueError("Cannot resume with changed protocol")
        load_decoder(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        ema.shadow = {k: v.to(device) for k, v in saved["ema"].items()}
        start_epoch, best, updates = saved["epoch"] + 1, saved["best_abs_rel"], saved["updates"]
        print(json.dumps({"event": "resume", "next_epoch": start_epoch + 1}), flush=True)
    started = time.monotonic()
    val_loader = make_loader(config, "validation", limit=max_val)
    for epoch in range(start_epoch, config["epochs"]):
        seed_everything(config["random_seed"] + 10000 * (epoch + 1))
        loader = first_loader if epoch == 0 else make_loader(config, "train", epoch, max_train)
        model.train()
        model.clip_vision_model.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        epoch_started = time.monotonic()
        for index, batch in enumerate(loader):
            inputs = batch["image_clip"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            auxiliary, pred = model(inputs, output_size=gt.shape[-2:])
            mask = (gt > config["min_depth"]) & (gt < config["max_depth"])
            if not mask.any():
                raise ValueError("Training batch has no valid depths")
            silog = loss_silog(pred, gt, mask)
            ssim = loss_ssim(pred, gt, mask)
            loss = (1 - config["w_ssim"]) * silog + config["w_ssim"] * ssim
            for aux, weight in zip(auxiliary or [], config["aux_loss_weights"]):
                if weight > 0:
                    loss = loss + weight * loss_silog(aux, gt, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {index}")
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            loss_sum += loss.detach().item()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                updates += 1
            if (index + 1) % 100 == 0 or index + 1 == len(loader):
                status = {
                    "phase": "training",
                    "variant": config["variant"],
                    "epoch": epoch + 1,
                    "epochs": config["epochs"],
                    "batches": index + 1,
                    "total_batches": len(loader),
                    "mean_loss": loss_sum / (index + 1),
                    "optimizer_updates": updates,
                    "last_gradient_norm": float(norm),
                    "epoch_seconds": time.monotonic() - epoch_started,
                    "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
                atomic_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
        raw_metrics, _ = evaluate(model, val_loader, config, device)
        raw_state = decoder_state(model)
        load_decoder(model, ema.shadow)
        try:
            ema_metrics, _ = evaluate(model, val_loader, config, device)
            selected = (
                "ema"
                if (
                    ema_metrics["regions"]["all"]["abs_rel"]
                    < raw_metrics["regions"]["all"]["abs_rel"]
                )
                else "raw"
            )
            chosen = ema_metrics if selected == "ema" else raw_metrics
            score = chosen["regions"]["all"]["abs_rel"]
            if score < best:
                best = score
                selected_state = (
                    decoder_state(model, cpu=True)
                    if selected == "ema"
                    else {k: v.cpu() for k, v in raw_state.items()}
                )
                atomic_save(
                    output / "best.pt",
                    {
                        "model": selected_state,
                        "epoch": epoch,
                        "source": selected,
                        "validation": chosen,
                        "config": config,
                        "config_sha256": config_hash(config),
                        "backbone_sha256": backbone_before,
                    },
                )
        finally:
            load_decoder(model, raw_state)
        row = {
            "epoch": epoch + 1,
            "mean_train_loss": loss_sum / len(loader),
            "validation_raw": raw_metrics,
            "validation_ema": ema_metrics,
            "selected": selected,
            "best_abs_rel": best,
            "epoch_seconds": time.monotonic() - epoch_started,
            "optimizer_updates": updates,
        }
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        atomic_save(
            last,
            {
                "model": decoder_state(model, cpu=True),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": {k: v.cpu() for k, v in ema.shadow.items()},
                "epoch": epoch,
                "best_abs_rel": best,
                "updates": updates,
                "config_sha256": config_hash(config),
            },
        )
        atomic_json(output / "status.json", {"phase": "epoch_complete", **row})
        print(json.dumps({"event": "epoch_complete", **row}), flush=True)
    backbone_after = state_hash(model.clip_vision_model.state_dict())
    if backbone_before != backbone_after:
        raise RuntimeError("Frozen backbone changed")
    atomic_json(
        output / "training_complete.json",
        {
            "completed": True,
            "best_validation_abs_rel": best,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "backbone_unchanged": True,
            "official_test_used_for_selection": False,
            "final_trainable_sha256": trainable_hash(model),
        },
    )


def test(config, output, device):
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if saved["config_sha256"] != config_hash(config):
        raise ValueError("Test config differs from training protocol")
    seed_everything(config["random_seed"])
    model = SPACECLIP(config).to(device)
    if state_hash(model.clip_vision_model.state_dict()) != saved["backbone_sha256"]:
        raise ValueError("Backbone differs from trained decoder")
    load_decoder(model, saved["model"])
    metrics, rows = evaluate(
        model,
        make_loader(config, "test"),
        config,
        device,
        detailed=True,
        flip=config["final_eval_flip_tta"],
    )
    atomic_json(
        output / "test_metrics.json",
        {
            "checkpoint_epoch": saved["epoch"] + 1,
            "checkpoint_source": saved["source"],
            "config_sha256": config_hash(config),
            "crop": config["eval_crop"],
            "median_scaling": False,
            "flip_tta": config["final_eval_flip_tta"],
            **metrics,
        },
    )
    with (output / "test_per_image.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["early", "middle", "late"], required=True)
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    if cli.smoke and cli.phase == "test":
        raise ValueError("Smoke validation must not use official test")
    config = load_config(cli.variant)
    if cli.smoke:
        config["epochs"] = 1
        config["workers"] = 0
        config["eval_workers"] = 0
    output = ROOT / "study" / ("smoke" if cli.smoke else "runs") / cli.variant
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    previous = output / "effective_config.json"
    if previous.exists() and config_hash(json.loads(previous.read_text())) != config_hash(config):
        raise ValueError("Existing run uses a different configuration; use a fresh checkout")
    atomic_json(previous, config)
    if cli.phase == "train":
        train(config, output, device, smoke=cli.smoke)
    else:
        if not (output / "training_complete.json").exists():
            raise RuntimeError("Training has not completed")
        test(config, output, device)


if __name__ == "__main__":
    main()
