"""Train all predeclared arms, then evaluate once on the official test set."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.layer_study.configuration import load_config
from scripts.layer_study.run import atomic_json, config_hash

VARIANTS = ("early", "middle", "late")


def main():
    (ROOT / "study").mkdir(exist_ok=True)
    status_path = ROOT / "study/queue_status.json"
    lock = (ROOT / "study/.queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    for variant in VARIANTS:
        smoke = ROOT / "study/smoke" / variant
        if not (smoke / "training_complete.json").exists():
            raise RuntimeError(f"Missing real-data smoke test: {variant}")
    init = [
        json.loads((ROOT / "study/smoke" / v / "initialization.json").read_text()) for v in VARIANTS
    ]
    for key in (
        "trainable_parameters",
        "total_parameters",
        "initial_trainable_sha256",
        "backbone_sha256",
    ):
        if len({x[key] for x in init}) != 1:
            raise RuntimeError(f"Uncontrolled initialization: {key}")
    for phase in ("train", "test"):
        for variant in VARIANTS:
            output = ROOT / "study/runs" / variant
            complete = output / (
                "training_complete.json" if phase == "train" else "test_metrics.json"
            )
            cfg = load_config(variant)
            previous = output / "effective_config.json"
            if previous.exists() and config_hash(json.loads(previous.read_text())) != config_hash(
                cfg
            ):
                raise RuntimeError(f"Protocol changed for existing run: {variant}")
            if complete.exists():
                continue
            state = {
                "phase": phase,
                "variant": variant,
                "pid": os.getpid(),
                "started_unix": started,
                "updated_unix": time.time(),
            }
            atomic_json(status_path, state)
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ROOT / "scripts/layer_study/run.py"),
                "--variant",
                variant,
                "--phase",
                phase,
            ]
            print(json.dumps({"event": "launch", **state}), flush=True)
            with (output / f"{phase}.log").open("a") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                atomic_json(
                    status_path,
                    {
                        **state,
                        "phase": "failed",
                        "failed_phase": phase,
                        "exit_code": process.returncode,
                    },
                )
                raise SystemExit(process.returncode)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/layer_study/summarize.py")], check=True, cwd=ROOT
    )
    atomic_json(
        status_path,
        {
            "phase": "completed",
            "started_unix": started,
            "completed_unix": time.time(),
            "variants": VARIANTS,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BlockingIOError:
        print("Another layer-study queue is already running.", file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:
        atomic_json(
            ROOT / "study/queue_status.json",
            {"phase": "failed", "error": repr(error), "updated_unix": time.time()},
        )
        raise
