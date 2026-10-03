#!/usr/bin/env python3
"""Benchmark SPACE-CLIP inference efficiency."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List
import sys

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from space_clip import SPACECLIP


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return argparse.Namespace(**cfg_dict)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parse_batch_sizes(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one batch size is required.")
    return values


def benchmark_callable(
    fn: Callable[[], None],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> Dict[str, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize(device)
                times_ms.append(float(start.elapsed_time(end)))
            else:
                t0 = time.perf_counter()
                fn()
                times_ms.append(float((time.perf_counter() - t0) * 1000.0))

    mean_ms = float(statistics.mean(times_ms))
    p50_ms = float(statistics.median(times_ms))
    if len(times_ms) > 1:
        std_ms = float(statistics.pstdev(times_ms))
    else:
        std_ms = 0.0

    sorted_times = sorted(times_ms)
    p90_idx = max(0, min(len(sorted_times) - 1, math.ceil(0.9 * len(sorted_times)) - 1))
    p90_ms = float(sorted_times[p90_idx])
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else float("inf")

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    return {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": std_ms,
        "latency_p50_ms": p50_ms,
        "latency_p90_ms": p90_ms,
        "fps": fps,
        "peak_memory_mb": peak_mem_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SPACE-CLIP efficiency.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    parser.add_argument("--batch-sizes", default="1", help="Comma-separated list, e.g. 1,4")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"[info] device: {device}")
    print(f"[info] config: {args.config}")

    model = SPACECLIP(config_model=cfg).to(device)
    model.eval()

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)
    backbone_params = count_params(model.clip_vision_model)
    backbone_trainable_params = count_trainable_params(model.clip_vision_model)
    decoder_params = total_params - backbone_params
    decoder_trainable_params = trainable_params - backbone_trainable_params

    clip_input = int(getattr(cfg, "clip_input_size", 224))
    output_h = int(getattr(cfg, "input_height", clip_input))
    output_w = int(getattr(cfg, "input_width", clip_input))
    interpolate_pos = bool(getattr(cfg, "clip_interpolate_pos_encoding", False))

    report: Dict[str, object] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": str(Path(args.config)),
        "device": str(device),
        "warmup": int(args.warmup),
        "iterations": int(args.iters),
        "params": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "backbone_total": int(backbone_params),
            "backbone_trainable": int(backbone_trainable_params),
            "decoder_total": int(decoder_params),
            "decoder_trainable": int(decoder_trainable_params),
        },
        "benchmarks": [],
    }

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, clip_input, clip_input, device=device)

        def clip_forward() -> None:
            kwargs = {
                "pixel_values": x,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if interpolate_pos:
                kwargs["interpolate_pos_encoding"] = True
            try:
                _ = model.clip_vision_model(**kwargs)
            except TypeError:
                kwargs.pop("interpolate_pos_encoding", None)
                _ = model.clip_vision_model(**kwargs)

        def full_forward() -> None:
            _ = model(x, output_size=(output_h, output_w))

        print(f"[info] benchmarking batch_size={batch_size} ...")
        clip_stats = benchmark_callable(clip_forward, args.warmup, args.iters, device)
        full_stats = benchmark_callable(full_forward, args.warmup, args.iters, device)

        delta_ms = full_stats["latency_mean_ms"] - clip_stats["latency_mean_ms"]
        delta_mem = full_stats["peak_memory_mb"] - clip_stats["peak_memory_mb"]

        block = {
            "batch_size": int(batch_size),
            "clip_input_hw": [clip_input, clip_input],
            "output_hw": [output_h, output_w],
            "clip_vision": clip_stats,
            "full_model": full_stats,
            "decoder_overhead": {
                "latency_mean_ms": float(delta_ms),
                "peak_memory_mb": float(delta_mem),
            },
        }
        report["benchmarks"].append(block)

        print(
            "[result] "
            f"bs={batch_size} | full {full_stats['latency_mean_ms']:.2f} ms "
            f"({full_stats['fps']:.2f} FPS), peak {full_stats['peak_memory_mb']:.1f} MB | "
            f"decoder overhead {delta_ms:.2f} ms"
        )

    report_text = json.dumps(report, indent=2)
    print(report_text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text + "\n", encoding="utf-8")
        print(f"[info] wrote report: {out_path}")


if __name__ == "__main__":
    main()
