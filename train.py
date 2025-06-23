# train.py

import argparse
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
from utils.loss import SILogLoss, SSIMLoss
from utils.utils import RunningAverage, RunningAverageDict, compute_errors
from utils import model_io

# --- 상수 정의 ---
PROJECT_DEFAULT_PREFIX = "space-clip"
LOGGING_ENABLED = True  # WandB 로깅 활성화 여부

# wandb로그인
wandb.login(key="103b8fdf09e76d2dcd0af31a69b3741c07a208ff") # 실제 API 키로 교체하세요

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

    train_loader = DepthDataLoader(args, 'train', collate_fn=safe_collate).data
    val_loader = DepthDataLoader(args, 'online_eval', collate_fn=safe_collate).data
    print(f"Dataloaders ready. Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    criterion_silog = SILogLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)
    w_ssim = getattr(args, 'w_ssim', 0.85)
    print(f"Using combined loss: (1.0 - {w_ssim:.2f}) * SILogLoss + {w_ssim:.2f} * SSIMLoss")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    step_size = getattr(args, 'step_lr_step_size', 10)
    gamma = getattr(args, 'step_lr_gamma', 0.5)
    print(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_epoch = 0
    if getattr(args, 'checkpoint_path', None) and os.path.exists(args.checkpoint_path):
        model, optimizer, start_epoch = model_io.load_checkpoint(args.checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")

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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = RunningAverage()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} Train")
        for i, batch in pbar:
            if batch is None: continue

            optimizer.zero_grad(set_to_none=True)
            
            image_clip = batch['image_clip'].to(device, non_blocking=True)
            image_orig = batch['image_orig'].to(device, non_blocking=True)
            depth_gt = batch['depth'].to(device, non_blocking=True)
            
            _, depth_pred = model(image_clip) # 모델은 CLIP 이미지 크기를 받음
            
            # Loss 계산 시에는 원본 해상도 GT와 비교하기 위해 예측값 리사이즈
            depth_pred_resized_for_loss = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode='bilinear', align_corners=False)
            
            mask = depth_gt > getattr(args, 'min_depth', 0.001)
            
            loss_silog = criterion_silog(depth_pred_resized_for_loss, depth_gt, mask=mask)
            loss_ssim = criterion_ssim(depth_pred_resized_for_loss, depth_gt, mask=mask)
            total_loss = (1.0 - w_ssim) * loss_silog + w_ssim * loss_ssim
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss detected at step {global_step}. Skipping backward pass.")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            epoch_loss.append(total_loss.item())
            pbar.set_postfix_str(f"Loss: {total_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if LOGGING_ENABLED and (global_step % getattr(args, 'log_scalar_every_iters', 10) == 0):
                wandb.log({
                    "Train/Step Loss Total": total_loss.item(),
                    "Train/Step Loss SILog": loss_silog.item(),
                    "Train/Step Loss SSIM": loss_ssim.item(),
                    "Train/Learning Rate": scheduler.get_last_lr()[0]
                }, step=global_step)
            
            global_step += 1
            
        print(f"Epoch {epoch+1} Average Train Loss: {epoch_loss.get_value():.4f}")
        if LOGGING_ENABLED:
            wandb.log({"Train/Epoch Loss Avg": epoch_loss.get_value(), "Epoch": epoch + 1}, step=global_step)

        scheduler.step()

        if (epoch + 1) % getattr(args, 'validate_epochs', 1) == 0 or (epoch + 1) == args.epochs:
            metrics, val_loss = validate(model, val_loader, (criterion_silog, criterion_ssim), w_ssim, args, device, global_step, fixed_val_samples)
            
            print(f"Validation Results - Avg Loss: {val_loss:.4f}, AbsRel: {metrics.get('abs_rel', -1):.4f}")
            if LOGGING_ENABLED:
                wandb.log({f"Val/Loss Avg": val_loss}, step=global_step)
                wandb.log({f"Val/Metrics/{k.replace(' ','_')}": v for k, v in metrics.items()}, step=global_step)
            
            is_best = metrics.get('abs_rel', float('inf')) < best_abs_rel
            checkpoint_dir = os.path.join(getattr(args, 'root', '.'), "checkpoints", args.name)
            
            model_io.save_checkpoint(model, optimizer, epoch, 'last_checkpoint.pt', checkpoint_dir)
            if is_best:
                best_abs_rel = metrics.get('abs_rel')
                if LOGGING_ENABLED:
                    wandb.summary['Best_AbsRel'] = best_abs_rel
                    wandb.summary['Best_Epoch'] = epoch + 1
                model_io.save_checkpoint(model, optimizer, epoch, 'best_checkpoint.pt', checkpoint_dir)
    
    if LOGGING_ENABLED: wandb.finish()
    print("Training finished.")

def validate(model, val_loader, criterions, w_ssim, args, device, global_step, fixed_samples):
    model.eval()
    val_loss = RunningAverage()
    eval_metrics = RunningAverageDict()
    criterion_silog, criterion_ssim = criterions

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch in pbar:
            if batch is None: continue

            image_clip = batch['image_clip'].to(device, non_blocking=True)
            depth_gt = batch['depth'].to(device, non_blocking=True)
            
            if not batch.get('has_valid_depth', torch.tensor(True)).any(): continue

            _, depth_pred = model(image_clip)
            depth_pred_resized_for_eval = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode='bilinear', align_corners=False)

            mask = depth_gt > getattr(args, 'min_depth_eval', 0.001)
            if not mask.any(): continue

            loss_silog = criterion_silog(depth_pred_resized_for_eval, depth_gt, mask=mask)
            loss_ssim = criterion_ssim(depth_pred_resized_for_eval, depth_gt, mask=mask)
            total_loss = (1.0 - w_ssim) * loss_silog + w_ssim * loss_ssim
            if not torch.isnan(total_loss): val_loss.append(total_loss.item())

            pred_np = depth_pred_resized_for_eval.cpu().numpy()
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
                
                scale_factor = np.median(gt_valid) / np.median(pred_valid) if np.median(pred_valid) > 1e-6 else 1.0
                pred_valid_scaled = pred_valid * scale_factor

                eval_metrics.update(compute_errors(gt_valid, pred_valid_scaled))

        if LOGGING_ENABLED and fixed_samples:
            rgb_list, gt_list, pred_list = [], [], []
            for sample in fixed_samples:
                if sample.get('has_valid_depth', False):
                    _, pred_depth = model(sample['image_clip'].to(device).unsqueeze(0))
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