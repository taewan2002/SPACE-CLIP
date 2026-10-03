"""Train the semantic-only + FiLM control using the archived layer-study recipe."""

import argparse
import json
from pathlib import Path

from scripts.component_study.initialization import copy_shared_initialization
from scripts.layer_study.configuration import ROOT, load_config


def require_completed_controls(roots, current_root):
    """Require all five full training runs before any new control test evaluation."""
    roots = [Path(root).resolve() for root in roots]
    if len(roots) != 5 or len(set(roots)) != 5 or Path(current_root).resolve() not in roots:
        raise ValueError("Supply five distinct control checkouts, including this checkout")
    seeds = []
    for root in roots:
        output = root / "study/runs/main_only"
        config = json.loads((output / "effective_config.json").read_text())
        complete = json.loads((output / "training_complete.json").read_text())
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        if (
            config.get("variant") != "main_only"
            or config.get("use_structural_pathway") is not False
            or config.get("use_film") is not True
            or config.get("epochs") != 20
            or complete.get("completed") is not True
            or complete.get("backbone_unchanged") is not True
            or complete.get("official_test_used_for_selection") is not False
            or [row["epoch"] for row in history] != list(range(1, 21))
            or not (output / "best.pt").is_file()
        ):
            raise ValueError(f"Incomplete or incompatible control training: {root}")
        seeds.append(config["random_seed"])
    if sorted(seeds) != list(range(42, 47)):
        raise ValueError("Completed controls must cover seeds 42 through 46 exactly once")


def make_paired_model(config, output, core):
    from space_clip import SPACECLIP

    core.seed_everything(config["random_seed"])
    full = SPACECLIP(dict(config, use_structural_pathway=True))
    reference = ROOT / "results/nyu_multiseed" / str(config["random_seed"]) / "early"
    expected = json.loads((reference / "initialization.json").read_text())
    if (
        core.trainable_hash(full) != expected["initial_trainable_sha256"]
        or core.state_hash(full.clip_vision_model.state_dict()) != expected["backbone_sha256"]
    ):
        raise ValueError("Full-model initialization differs from the archived same-seed reference")
    control = SPACECLIP(config)
    audit = copy_shared_initialization(full, control)
    audit.update(
        seed=config["random_seed"],
        reference_initialization_matched=True,
        full_trainable_parameters=expected["trainable_parameters"],
        control_trainable_parameters=sum(
            p.numel() for p in control.parameters() if p.requires_grad
        ),
        control_initial_trainable_sha256=core.trainable_hash(control),
        backbone_sha256=expected["backbone_sha256"],
        parameter_matched=False,
    )
    core.atomic_json(output / "initialization_audit.json", audit)
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--control-roots", nargs=5, type=Path, metavar="CHECKOUT")
    args = parser.parse_args()
    if args.smoke and args.phase == "test":
        parser.error("Smoke runs only use training and validation images")
    if args.phase == "test":
        if args.control_roots is None:
            parser.error("--phase test requires --control-roots for all five seeds")
        require_completed_controls(args.control_roots, ROOT)

    import torch
    from scripts.layer_study import run as core

    config = load_config("early")
    if config["random_seed"] not in range(42, 47) or not config["use_film"]:
        raise ValueError("The control requires FiLM and a seed from 42 through 46")
    config.update(
        variant="main_only",
        name=f"COMPONENT_NYU_SEMANTIC_ONLY_SEED{config['random_seed']}",
        use_structural_pathway=False,
        notes="Semantic-only + FiLM; test after all five new control trainings complete.",
    )
    if args.smoke:
        config.update(epochs=1, workers=0, eval_workers=0)
    output = ROOT / "study" / ("smoke" if args.smoke else "runs") / "main_only"
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Training and checkpoint evaluation require a CUDA GPU")
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    previous = output / "effective_config.json"
    if previous.exists() and core.config_hash(json.loads(previous.read_text())) != core.config_hash(
        config
    ):
        raise ValueError("Existing run uses a different configuration; use a fresh checkout")
    core.atomic_json(previous, config)
    # Reuse the existing optimizer, losses, checkpoint selection, and evaluator.
    original_factory = core.SPACECLIP
    try:
        core.SPACECLIP = lambda cfg: make_paired_model(cfg, output, core)
        if args.phase == "train":
            core.train(config, output, device, smoke=args.smoke)
        else:
            core.test(config, output, device)
    finally:
        core.SPACECLIP = original_factory


if __name__ == "__main__":
    main()
