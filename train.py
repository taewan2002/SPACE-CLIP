# train.py

import argparse
import math
import os
import sys
import uuid
from datetime import datetime as dt
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import wandb
from torch.utils.data.dataloader import default_collate
from torchvision.utils import make_grid
from tqdm import tqdm

from space_clip import SPACECLIP
from utils.dataloader import DepthDataLoader
from utils.loss import GradientLoss, SILogLoss, SSIMLoss
from utils.utils import RunningAverage, RunningAverageDict, compute_errors
from utils import model_io

# --- 상수 정의 ---
PROJECT_DEFAULT_PREFIX = "space-clip"
LOGGING_ENABLED = False  # WandB 로깅 활성화 여부 (runtime setup)


def setup_wandb_logging(enable: bool = True) -> bool:
    """Configure WandB login at runtime to avoid import-time side effects."""
    global LOGGING_ENABLED
    if not enable:
        LOGGING_ENABLED = False
        return LOGGING_ENABLED
    try:
        wandb.login()
        LOGGING_ENABLED = True
    except Exception as e:
        LOGGING_ENABLED = False
        print(f"Warning: WandB login failed. Logging disabled. ({e})")
    return LOGGING_ENABLED

# --- 유틸리티 함수 ---
def set_random_seed(seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")
    else:
        print("Using random seed.")

def load_config_from_yaml(yaml_file: str) -> argparse.Namespace:
    try:
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return argparse.Namespace(**config_dict)
    except Exception as e:
        print(f"ERROR: Failed to load or parse YAML file '{yaml_file}': {e}")
        sys.exit(1)

def setup_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available():
        gpu_id_str = str(getattr(args, 'gpu', "0"))
        try:
            gpu_id = int(gpu_id_str.split(',')[0])
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Successfully set up GPU: {device}")
        except (ValueError, IndexError):
            device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
            print(f"Warning: Invalid GPU ID '{gpu_id_str}'. Using default: {device}.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Running on CPU.")
    return device

def safe_collate(batch):
    """DataLoader에서 None 샘플을 안전하게 걸러내는 collate_fn."""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)


def normalize_for_clip(image_tensor: torch.Tensor) -> torch.Tensor:
    """[0, 1] 범위 이미지를 CLIP 입력 정규화로 변환."""
    mean = image_tensor.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = image_tensor.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


def get_model_input_from_batch(batch: dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """설정에 따라 모델 입력을 선택한다."""
    if getattr(args, 'use_image_orig_for_clip', False):
        image_orig = batch['image_orig'].to(device, non_blocking=True)
        return normalize_for_clip(image_orig)
    return batch['image_clip'].to(device, non_blocking=True)


class ExponentialMovingAverage:
    """Simple EMA tracker for model parameters/buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, tensor in model.state_dict().items():
            self.shadow[name] = tensor.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                if name not in self.shadow:
                    self.shadow[name] = tensor.detach().clone()
                    continue
                if tensor.is_floating_point():
                    self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name] = tensor.detach().clone()

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}


def get_current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]['lr'])


def _resolve_aux_weight(idx: int, aux_weights) -> float:
    if aux_weights is None:
        return 0.0
    if isinstance(aux_weights, (int, float)):
        return float(aux_weights)
    if len(aux_weights) == 0:
        return 0.0
    if idx < len(aux_weights):
        return float(aux_weights[idx])
    return float(aux_weights[-1])


def infer_depth(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    output_size,
    use_flip_tta: bool = False,
) -> tuple:
    aux_preds, depth_pred = model(model_input, output_size=output_size)
    if not use_flip_tta:
        return aux_preds, depth_pred

    input_flip = torch.flip(model_input, dims=[3])
    _, depth_flip = model(input_flip, output_size=output_size)
    depth_flip = torch.flip(depth_flip, dims=[3])
    depth_pred = 0.5 * (depth_pred + depth_flip)
    return aux_preds, depth_pred

def log_depth_images_to_wandb(rgb_list, gt_depth_list, pred_depth_list, global_step, prefix="Val_Vis/"):
    if not LOGGING_ENABLED or not wandb.run: return

    def process_depth_for_vis(depth_tensor: torch.Tensor) -> torch.Tensor:
        depth_map = depth_tensor.squeeze().cpu().detach().numpy()
        valid_mask = depth_map > 0
        if not valid_mask.any(): return torch.zeros(3, *depth_map.shape)
        min_val, max_val = np.percentile(depth_map[valid_mask], [2, 98])
        depth_map_clipped = np.clip(depth_map, min_val, max_val)
        if (max_val - min_val) > 1e-5:
            depth_map_normalized = (depth_map_clipped - min_val) / (max_val - min_val)
        else:
            depth_map_normalized = np.zeros_like(depth_map_clipped)
        colored_map = plt.get_cmap('viridis')(depth_map_normalized)[:, :, :3]
        return torch.from_numpy(colored_map).permute(2, 0, 1)

    try:
        log_images = []
        for rgb, gt, pred in zip(rgb_list, gt_depth_list, pred_depth_list):
            rgb_vis = rgb.squeeze().cpu().detach()
            gt_vis = process_depth_for_vis(gt)
            pred_vis = process_depth_for_vis(pred)
            target_size = rgb_vis.shape[-2:]
            gt_vis = F.interpolate(gt_vis.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            pred_vis = F.interpolate(pred_vis.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            log_images.extend([rgb_vis, gt_vis, pred_vis])
        
        if log_images:
            grid = make_grid(log_images, nrow=3, normalize=False, pad_value=0.5)
            wandb.log({f"{prefix}Comparison_Grid_(RGB|GT|Pred)": [wandb.Image(grid)]}, step=global_step)
            
    except Exception as e:
        print(f"Error during WandB image logging: {e}")

def train(model: torch.nn.Module, args: argparse.Namespace):
    device = setup_device(args)
    model.to(device)
    setup_wandb_logging(enable=bool(getattr(args, 'use_wandb', True)))

    train_loader = DepthDataLoader(args, 'train', collate_fn=safe_collate).data
    val_loader = DepthDataLoader(args, 'online_eval', collate_fn=safe_collate).data
    print(f"Dataloaders ready. Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    criterion_silog = SILogLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)
    criterion_grad = GradientLoss().to(device)
    w_ssim = getattr(args, 'w_ssim', 0.85)
    w_grad = getattr(args, 'w_grad', 0.0)
    print(
        f"Using combined loss: (1.0 - {w_ssim:.2f}) * SILogLoss + {w_ssim:.2f} * SSIMLoss + {w_grad:.2f} * GradLoss"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', param_group['lr'])

    start_epoch = 0
    if getattr(args, 'checkpoint_path', None) and os.path.exists(args.checkpoint_path):
        model, optimizer, start_epoch = model_io.load_checkpoint(args.checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")

    scheduler_name = getattr(args, 'scheduler', 'step').lower()
    scheduler_step_per_iter = False
    if scheduler_name == 'cosine_warmup':
        total_train_steps = max(1, len(train_loader) * args.epochs)
        min_lr = getattr(args, 'min_lr', 1e-6)
        warmup_steps = int(getattr(args, 'warmup_steps', 0))
        if warmup_steps <= 0:
            warmup_ratio = float(getattr(args, 'warmup_ratio', 0.05))
            warmup_steps = int(total_train_steps * warmup_ratio)
        warmup_steps = max(1, min(warmup_steps, total_train_steps - 1))

        base_lr = max(float(args.learning_rate), 1e-12)
        min_lr_ratio = min(1.0, max(0.0, float(min_lr) / base_lr))
        resume_steps = max(0, start_epoch * len(train_loader))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=resume_steps - 1)
        scheduler_step_per_iter = True
        print(
            f"Using cosine_warmup scheduler: total_steps={total_train_steps}, "
            f"warmup_steps={warmup_steps}, min_lr={min_lr}"
        )
    elif scheduler_name == 'cosine':
        min_lr = getattr(args, 'min_lr', 1e-6)
        print(f"Using CosineAnnealingLR scheduler with T_max={args.epochs}, eta_min={min_lr}")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=min_lr)
        for _ in range(start_epoch):
            scheduler.step()
    else:
        step_size = getattr(args, 'step_lr_step_size', 10)
        gamma = getattr(args, 'step_lr_gamma', 0.5)
        print(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        for _ in range(start_epoch):
            scheduler.step()

    use_ema = bool(getattr(args, 'use_ema', False))
    eval_with_ema = bool(getattr(args, 'eval_with_ema', use_ema))
    eval_both_for_best = bool(getattr(args, 'eval_both_for_best', False))
    ema_decay = float(getattr(args, 'ema_decay', 0.999))
    ema = ExponentialMovingAverage(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(
            f"Using EMA with decay={ema_decay:.6f} "
            f"(eval_with_ema={eval_with_ema}, eval_both_for_best={eval_both_for_best})"
        )

    if LOGGING_ENABLED:
        run_id = f"{dt.now().strftime('%y%m%d-%H%M')}-{uuid.uuid4().hex[:4]}"
        exp_name = f"{args.name}_{run_id}"
        proj_name = f"{PROJECT_DEFAULT_PREFIX}-{args.dataset.lower()}"
        wandb.init(project=proj_name, name=exp_name, config=vars(args))
        print(f"WandB run initialized: project='{proj_name}', name='{exp_name}'")

    fixed_val_samples = []
    if LOGGING_ENABLED and len(val_loader.dataset) > 0:
        indices = np.linspace(0, len(val_loader.dataset) - 1, getattr(args, 'num_log_images', 5), dtype=int)
        for i in indices:
            sample = val_loader.dataset[i]
            if sample is not None:
                fixed_val_samples.append(sample)
        print(f"Prepared {len(fixed_val_samples)} fixed samples for validation logging.")

    best_abs_rel = float('inf')
    global_step = start_epoch * len(train_loader)
    use_multiscale_supervision = bool(getattr(args, 'use_multiscale_supervision', False))
    aux_loss_weights = getattr(args, 'aux_loss_weights', [0.25, 0.125, 0.0625])
    if use_multiscale_supervision:
        print(f"Using multi-scale auxiliary SILog loss. Weights={aux_loss_weights}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = RunningAverage()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} Train")
        for i, batch in pbar:
            if batch is None: continue

            optimizer.zero_grad(set_to_none=True)
            
            depth_gt = batch['depth'].to(device, non_blocking=True)

            model_input = get_model_input_from_batch(batch, args, device)
            aux_preds, depth_pred = infer_depth(
                model, model_input, output_size=depth_gt.shape[-2:], use_flip_tta=False
            )

            mask = (
                (depth_gt > getattr(args, 'min_depth', 0.001))
                & (depth_gt < getattr(args, 'max_depth', 80.0))
            )

            loss_silog = criterion_silog(depth_pred, depth_gt, mask=mask)
            loss_ssim = criterion_ssim(depth_pred, depth_gt, mask=mask)
            total_loss = (1.0 - w_ssim) * loss_silog + w_ssim * loss_ssim

            aux_loss = torch.zeros((), device=device)
            if use_multiscale_supervision and aux_preds:
                for aux_idx, aux_pred in enumerate(aux_preds):
                    aux_weight = _resolve_aux_weight(aux_idx, aux_loss_weights)
                    if aux_weight <= 0.0:
                        continue
                    aux_loss = aux_loss + aux_weight * criterion_silog(aux_pred, depth_gt, mask=mask)
                total_loss = total_loss + aux_loss

            if w_grad > 0.0:
                loss_grad = criterion_grad(depth_pred, depth_gt, mask=mask)
                total_loss = total_loss + w_grad * loss_grad
            else:
                loss_grad = torch.zeros((), device=device)
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss detected at step {global_step}. Skipping backward pass.")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            if scheduler_step_per_iter:
                scheduler.step()
            if ema is not None:
                ema.update(model)
            
            epoch_loss.append(total_loss.item())
            current_lr = get_current_lr(optimizer)
            pbar.set_postfix_str(f"Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}")
            
            if LOGGING_ENABLED and (global_step % getattr(args, 'log_scalar_every_iters', 10) == 0):
                wandb.log({
                    "Train/Step Loss Total": total_loss.item(),
                    "Train/Step Loss SILog": loss_silog.item(),
                    "Train/Step Loss SSIM": loss_ssim.item(),
                    "Train/Step Loss Aux": aux_loss.item(),
                    "Train/Step Loss Grad": loss_grad.item(),
                    "Train/Learning Rate": current_lr
                }, step=global_step)
            
            global_step += 1
            
        print(f"Epoch {epoch+1} Average Train Loss: {epoch_loss.get_value():.4f}")
        if LOGGING_ENABLED:
            wandb.log({"Train/Epoch Loss Avg": epoch_loss.get_value(), "Epoch": epoch + 1}, step=global_step)

        if not scheduler_step_per_iter:
            scheduler.step()

        if (epoch + 1) % getattr(args, 'validate_epochs', 1) == 0 or (epoch + 1) == args.epochs:
            metrics = None
            val_loss = None
            best_source = "raw"

            if ema is not None and eval_both_for_best:
                raw_metrics, raw_val_loss = validate(
                    model,
                    val_loader,
                    (criterion_silog, criterion_ssim, criterion_grad),
                    w_ssim,
                    w_grad,
                    args,
                    device,
                    global_step,
                    fixed_val_samples,
                )
                if LOGGING_ENABLED:
                    wandb.log({f"Val/raw/Loss Avg": raw_val_loss}, step=global_step)
                    wandb.log({f"Val/raw/Metrics/{k.replace(' ','_')}": v for k, v in raw_metrics.items()}, step=global_step)

                ema.store(model)
                ema.copy_to(model)
                try:
                    ema_metrics, ema_val_loss = validate(
                        model,
                        val_loader,
                        (criterion_silog, criterion_ssim, criterion_grad),
                        w_ssim,
                        w_grad,
                        args,
                        device,
                        global_step,
                        fixed_val_samples,
                    )
                finally:
                    ema.restore(model)
                if LOGGING_ENABLED:
                    wandb.log({f"Val/ema/Loss Avg": ema_val_loss}, step=global_step)
                    wandb.log({f"Val/ema/Metrics/{k.replace(' ','_')}": v for k, v in ema_metrics.items()}, step=global_step)

                raw_abs_rel = raw_metrics.get('abs_rel', float('inf'))
                ema_abs_rel = ema_metrics.get('abs_rel', float('inf'))
                if ema_abs_rel < raw_abs_rel:
                    metrics, val_loss = ema_metrics, ema_val_loss
                    best_source = "ema"
                else:
                    metrics, val_loss = raw_metrics, raw_val_loss
                    best_source = "raw"

                print(
                    "Validation Results - "
                    f"Avg Loss: {val_loss:.4f}, AbsRel: {metrics.get('abs_rel', -1):.4f} "
                    f"(selected={best_source}, raw={raw_abs_rel:.4f}, ema={ema_abs_rel:.4f})"
                )
            else:
                if ema is not None and eval_with_ema:
                    ema.store(model)
                    ema.copy_to(model)
                    best_source = "ema"
                try:
                    metrics, val_loss = validate(
                        model,
                        val_loader,
                        (criterion_silog, criterion_ssim, criterion_grad),
                        w_ssim,
                        w_grad,
                        args,
                        device,
                        global_step,
                        fixed_val_samples,
                    )
                finally:
                    if ema is not None and eval_with_ema:
                        ema.restore(model)

                print(f"Validation Results - Avg Loss: {val_loss:.4f}, AbsRel: {metrics.get('abs_rel', -1):.4f}")

            if LOGGING_ENABLED:
                wandb.log({f"Val/Loss Avg": val_loss}, step=global_step)
                wandb.log({f"Val/Metrics/{k.replace(' ','_')}": v for k, v in metrics.items()}, step=global_step)
                wandb.log({"Val/Selected Source": 1 if best_source == "ema" else 0}, step=global_step)

            is_best = metrics.get('abs_rel', float('inf')) < best_abs_rel
            checkpoint_dir = os.path.join(getattr(args, 'root', '.'), "checkpoints", args.name)
            
            model_io.save_checkpoint(model, optimizer, epoch, 'last_checkpoint.pt', checkpoint_dir)
            if is_best:
                best_abs_rel = metrics.get('abs_rel')
                if LOGGING_ENABLED:
                    wandb.summary['Best_AbsRel'] = best_abs_rel
                    wandb.summary['Best_Epoch'] = epoch + 1
                    wandb.summary['Best_Source'] = best_source
                if ema is not None and best_source == "ema":
                    ema.store(model)
                    ema.copy_to(model)
                    try:
                        model_io.save_checkpoint(model, optimizer, epoch, 'best_checkpoint.pt', checkpoint_dir)
                    finally:
                        ema.restore(model)
                else:
                    model_io.save_checkpoint(model, optimizer, epoch, 'best_checkpoint.pt', checkpoint_dir)
    
    if LOGGING_ENABLED: wandb.finish()
    print("Training finished.")

def validate(model, val_loader, criterions, w_ssim, w_grad, args, device, global_step, fixed_samples):
    model.eval()
    val_loss = RunningAverage()
    eval_metrics = RunningAverageDict()
    criterion_silog, criterion_ssim, criterion_grad = criterions
    use_flip_tta = bool(getattr(args, 'eval_flip_tta', False))
    use_multiscale_supervision = bool(getattr(args, 'use_multiscale_supervision', False))
    aux_loss_weights = getattr(args, 'aux_loss_weights', [0.25, 0.125, 0.0625])

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch in pbar:
            if batch is None: continue

            depth_gt = batch['depth'].to(device, non_blocking=True)
            
            if not batch.get('has_valid_depth', torch.tensor(True)).any(): continue

            model_input = get_model_input_from_batch(batch, args, device)
            aux_preds, depth_pred = infer_depth(
                model,
                model_input,
                output_size=depth_gt.shape[-2:],
                use_flip_tta=use_flip_tta,
            )

            mask = (
                (depth_gt > getattr(args, 'min_depth_eval', 0.001))
                & (depth_gt < getattr(args, 'max_depth_eval', 80.0))
            )
            if not mask.any(): continue

            loss_silog = criterion_silog(depth_pred, depth_gt, mask=mask)
            loss_ssim = criterion_ssim(depth_pred, depth_gt, mask=mask)
            total_loss = (1.0 - w_ssim) * loss_silog + w_ssim * loss_ssim
            if use_multiscale_supervision and aux_preds:
                for aux_idx, aux_pred in enumerate(aux_preds):
                    aux_weight = _resolve_aux_weight(aux_idx, aux_loss_weights)
                    if aux_weight <= 0.0:
                        continue
                    total_loss = total_loss + aux_weight * criterion_silog(aux_pred, depth_gt, mask=mask)
            if w_grad > 0.0:
                total_loss = total_loss + w_grad * criterion_grad(depth_pred, depth_gt, mask=mask)
            if not torch.isnan(total_loss): val_loss.append(total_loss.item())

            pred_np = depth_pred.cpu().numpy()
            gt_np = depth_gt.cpu().numpy()
            
            for i in range(gt_np.shape[0]):
                p, g = pred_np[i].squeeze(), gt_np[i].squeeze()
                valid_eval_mask = (g > args.min_depth_eval) & (g < args.max_depth_eval)
                
                if getattr(args, 'garg_crop', False):
                    h, w = g.shape
                    crop_mask = np.zeros_like(valid_eval_mask)
                    if args.dataset == 'kitti':
                        crop_mask[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
                    else: # NYU
                        crop_mask[45:471, 41:601] = True
                    valid_eval_mask &= crop_mask

                if not valid_eval_mask.any(): continue
                gt_valid, pred_valid = g[valid_eval_mask], p[valid_eval_mask]
                
                pred_median = np.median(pred_valid)
                gt_median = np.median(gt_valid)
                if np.isfinite(pred_median) and np.isfinite(gt_median) and pred_median > 1e-6:
                    scale_factor = gt_median / pred_median
                else:
                    scale_factor = 1.0

                pred_valid_scaled = pred_valid * scale_factor
                pred_valid_scaled = np.clip(pred_valid_scaled, args.min_depth_eval, args.max_depth_eval)

                metric_dict = compute_errors(gt_valid, pred_valid_scaled)
                if all(np.isfinite(v) for v in metric_dict.values()):
                    eval_metrics.update(metric_dict)

        if LOGGING_ENABLED and fixed_samples:
            rgb_list, gt_list, pred_list = [], [], []
            for sample in fixed_samples:
                if sample.get('has_valid_depth', False):
                    sample_batch = {
                        'image_clip': sample['image_clip'].unsqueeze(0),
                        'image_orig': sample['image_orig'].unsqueeze(0),
                    }
                    model_input = get_model_input_from_batch(sample_batch, args, device)
                    _, pred_depth = infer_depth(
                        model,
                        model_input,
                        output_size=sample['depth'].shape[-2:],
                        use_flip_tta=use_flip_tta,
                    )
                    rgb_list.append(sample['image_orig'])
                    gt_list.append(sample['depth'])
                    pred_list.append(pred_depth)
            log_depth_images_to_wandb(rgb_list, gt_list, pred_list, global_step)

    return eval_metrics.get_value(), val_loss.get_value()

def main():
    parser = argparse.ArgumentParser(description='Space-CLIP Training Script')
    parser.add_argument('--config_file', type=str, default='configs/kitti.yaml', help='Path to the YAML configuration file.')
    cli_args_raw = parser.parse_args()
    args = load_config_from_yaml(cli_args_raw.config_file)
            
    args.distributed = False
    args.mode = 'train'
    if not hasattr(args, 'root'): args.root = "."
    
    set_random_seed(getattr(args, 'random_seed', None))
            
    model = SPACECLIP(config_model=args)
    train(model, args)

if __name__ == '__main__':
    main()
