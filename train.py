"""
Training script for AlignmentVAE — single-machine multi-GPU (DDP).

    Image → VAE Encoder          → N(μ, σ²)  [B, h, w, D]
    Label → SpatialLabelEncoder  → N(μ, σ²)  [B, h, w, D]

Loss = MSE recon + VAE KL + Alignment KL + Label entropy

Usage:
    python train.py --config configs/default.yaml
    torchrun --nproc_per_node=4 train.py --config configs/default.yaml
"""

import argparse
import datetime
import math
import numpy as np
import os
import sys
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import AlignmentVAE
from model.text_encoder import SpatialLabelEncoder, SpatialLabelEncoder_models
from model.vae import VAE, VAE_models

import util.misc as misc
import util.lr_sched as lr_sched


def get_kl_anneal_factor(epoch: float, args) -> float:
    """Compute KL annealing factor (0 → 1) based on current epoch.

    Supports three strategies:
        - "linear":  linearly ramp from 0 to 1 over [0, kl_anneal_epochs)
        - "cosine":  cosine ramp (slow start, fast middle, slow end)
        - "constant": no annealing, always 1.0

    Args:
        epoch: current fractional epoch (e.g. 3.5 = halfway through epoch 4)
        args: must have .kl_anneal_strategy and .kl_anneal_epochs

    Returns:
        factor in [0, 1]
    """
    strategy = getattr(args, 'kl_anneal_strategy', 'linear')
    anneal_epochs = getattr(args, 'kl_anneal_epochs', 0)

    if strategy == 'constant' or anneal_epochs <= 0:
        return 1.0

    progress = min(epoch / anneal_epochs, 1.0)

    if strategy == 'linear':
        return progress
    elif strategy == 'cosine':
        return 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        return 1.0


# ============================================================================
# Argument Parser
# ============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('AlignmentVAE Training', add_help=True)

    # config file
    parser.add_argument('--config', type=str, default='configs/default.yaml')

    # architecture overrides
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--vae_variant', type=str, default=None,
                        help='VAE-B | VAE-L | VAE-H')
    parser.add_argument('--label_encoder_variant', type=str, default=None,
                        help='SLE-B | SLE-L | SLE-H')
    parser.add_argument('--num_classes', type=int, default=None)

    # training
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Per-GPU batch size')
    parser.add_argument('--accum_iter', type=int, default=None,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_iter * num_gpus)')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=None,
                        help='Base LR: actual = blr * eff_batch / 256')
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)

    # KL annealing
    parser.add_argument('--kl_anneal_strategy', type=str, default=None,
                        help='KL annealing strategy: linear | cosine | constant')
    parser.add_argument('--kl_anneal_epochs', type=float, default=None,
                        help='Number of epochs to anneal KL weight from 0 to 1')

    # dataset
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output_dir')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--log_freq', type=int, default=None)

    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')

    return parser


# ============================================================================
# Config Merge
# ============================================================================

def merge_config_and_args(cfg: Dict, args) -> argparse.Namespace:
    """Merge YAML config with CLI args (CLI takes priority)."""
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    optim_cfg = cfg.get('optimizer', {})
    log_cfg = cfg.get('logging', {})

    # Model
    args.input_size = args.input_size or model_cfg.get('input_size', 256)
    args.latent_dim = args.latent_dim or model_cfg.get('latent_dim', 16)
    args.vae_variant = args.vae_variant or model_cfg.get('vae_variant', 'VAE-B')
    args.label_encoder_variant = (args.label_encoder_variant
                                  or model_cfg.get('label_encoder_variant', 'SLE-B'))
    args.num_classes = args.num_classes or model_cfg.get('num_classes', 1000)
    args.image_channels = model_cfg.get('image_channels', 3)
    args.ch = model_cfg.get('ch', 128)
    args.ch_mult = model_cfg.get('ch_mult', [1, 2, 4, 4])
    args.num_res_blocks = model_cfg.get('num_res_blocks', 2)
    args.attn_resolutions = model_cfg.get('attn_resolutions', [16])
    args.double_z = model_cfg.get('double_z', True)
    args.label_init_logsigma = model_cfg.get('label_init_logsigma', -2.0)

    # Training
    args.epochs = args.epochs or train_cfg.get('epochs', 100)
    args.batch_size = args.batch_size or train_cfg.get('batch_size', 32)
    args.accum_iter = (args.accum_iter if args.accum_iter is not None
                       else train_cfg.get('accum_iter', 1))
    args.warmup_epochs = (args.warmup_epochs if args.warmup_epochs is not None
                          else train_cfg.get('warmup_epochs', 5))
    args.weight_decay = (args.weight_decay if args.weight_decay is not None
                         else optim_cfg.get('weight_decay', 0.01))
    args.grad_clip = (args.grad_clip if args.grad_clip is not None
                      else train_cfg.get('grad_clip', 1.0))

    # LR
    lr_from_cfg = float(train_cfg.get('lr', 1e-4))
    args.lr_label_encoder = float(train_cfg.get('lr_label_encoder', 5e-5))
    if args.lr is None and args.blr is None:
        args.lr = lr_from_cfg

    # Loss weights (explicit float() to guard against YAML parsing '1e-4' as str)
    args.weight_recon = float(train_cfg.get('weight_recon', 1.0))
    args.weight_kl = float(train_cfg.get('weight_kl', 1e-4))
    args.weight_alignment = float(train_cfg.get('weight_alignment', 0.1))
    args.weight_label_entropy = float(train_cfg.get('weight_label_entropy', 0.01))
    args.alignment_temp = float(train_cfg.get('alignment_temp', 1.0))

    # KL annealing
    args.kl_anneal_strategy = (args.kl_anneal_strategy
                               or train_cfg.get('kl_anneal_strategy', 'linear'))
    args.kl_anneal_epochs = (args.kl_anneal_epochs if args.kl_anneal_epochs is not None
                             else float(train_cfg.get('kl_anneal_epochs', 10)))

    # Dataset
    args.data_path = args.data_path or train_cfg.get('data_root', './data/imagenet')
    args.val_split = train_cfg.get('val_split', 0.1)
    args.test_split = train_cfg.get('test_split', 0.05)

    # Workers / logging
    args.num_workers = (args.num_workers if args.num_workers is not None
                        else train_cfg.get('num_workers', 4))
    args.log_freq = (args.log_freq if args.log_freq is not None
                     else train_cfg.get('log_interval', 100))
    args.save_freq = (args.save_freq if args.save_freq is not None
                      else train_cfg.get('save_interval', 10))
    args.use_tensorboard = log_cfg.get('use_tensorboard', True)

    # Evaluation
    args.compute_fid = train_cfg.get('compute_fid', False)
    args.fid_num_samples = train_cfg.get('fid_num_samples', 10000)

    args.yaml_config = cfg
    return args


# ============================================================================
# Evaluation Metrics
# ============================================================================

class InceptionV3Features(nn.Module):
    """InceptionV3 pool3 (2048-dim) feature extractor for FID."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.blocks.eval().to(device)
        for p in self.blocks.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] → [B,2048]"""
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.blocks(x).flatten(1)


def compute_fid_from_features(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    """Fréchet Inception Distance between two feature sets."""
    from scipy import linalg

    mu_r, mu_g = np.mean(feats_real, axis=0), np.mean(feats_gen, axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_g = np.cov(feats_gen, rowvar=False)

    diff = mu_r - mu_g
    covmean = linalg.sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        covmean = np.zeros_like(sigma_r)

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))


class MetricsComputer:
    """Evaluation metrics: MSE, PSNR, SSIM, Inception features."""

    def __init__(self, device: str = "cuda", compute_fid: bool = False):
        self.device = device
        self._ssim_kernel_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self.inception = InceptionV3Features(device) if compute_fid else None

    def _get_ssim_kernel(self, channels: int, window_size: int,
                         device: torch.device) -> torch.Tensor:
        """Get or create cached SSIM Gaussian kernel."""
        key = (channels, window_size, str(device))
        if key not in self._ssim_kernel_cache:
            sigma = 1.5
            coords = torch.arange(window_size, dtype=torch.float32, device=device)
            coords -= window_size // 2
            gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            gauss /= gauss.sum()
            window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
            kernel = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
            self._ssim_kernel_cache[key] = kernel
        return self._ssim_kernel_cache[key]

    def compute_mse(self, x: torch.Tensor, x_recon: torch.Tensor) -> float:
        return torch.mean((x - x_recon) ** 2).item()

    def compute_psnr(self, x: torch.Tensor, x_recon: torch.Tensor) -> float:
        mse = torch.mean((x - x_recon) ** 2)
        if mse == 0:
            return float('inf')
        max_val = 1.0 if x.max() <= 1.0 else 255.0
        return (20 * torch.log10(torch.tensor(max_val, device=x.device) / torch.sqrt(mse))).item()

    def compute_ssim(self, x: torch.Tensor, x_recon: torch.Tensor,
                     window_size: int = 11,
                     C1: float = 0.01**2, C2: float = 0.03**2) -> float:
        """SSIM between original and reconstructed images."""
        C = x.shape[1]
        window = self._get_ssim_kernel(C, window_size, x.device)
        pad = window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=C)
        mu_y = F.conv2d(x_recon, window, padding=pad, groups=C)
        mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(x_recon * x_recon, window, padding=pad, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * x_recon, window, padding=pad, groups=C) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean().item()

    @torch.no_grad()
    def extract_inception_features(self, images: torch.Tensor) -> np.ndarray:
        """[B,3,H,W] → [B,2048] numpy."""
        return self.inception(images).cpu().numpy()


# ============================================================================
# Dataset
# ============================================================================

class ImageLabelDataset(Dataset):
    """ImageNet or dummy (image, label) dataset."""

    def __init__(self, root: str, split: str = "train", transform=None,
                 num_classes: int = 1000, max_samples: Optional[int] = None):
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        ])

        imagenet_path = os.path.join(root, split)
        if os.path.isdir(imagenet_path):
            try:
                self.dataset = datasets.ImageFolder(imagenet_path, transform=self.transform)
                self.use_real = True
                if max_samples and max_samples < len(self.dataset):
                    self.dataset = torch.utils.data.Subset(
                        self.dataset, list(range(max_samples))
                    )
            except Exception:
                self.use_real = False
                self.length = max_samples or 1000
        else:
            self.use_real = False
            self.length = max_samples or 1000

    def __len__(self) -> int:
        return len(self.dataset) if self.use_real else self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.use_real:
            return self.dataset[idx]
        return torch.rand(3, 256, 256), idx % self.num_classes


# ============================================================================
# Model Building
# ============================================================================

def _fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def build_model(args) -> AlignmentVAE:
    """Build AlignmentVAE from config args."""

    # VAE
    vae_variant = getattr(args, 'vae_variant', 'VAE-B')
    if vae_variant in VAE_models:
        vae = VAE_models[vae_variant](
            latent_dim=args.latent_dim, resolution=args.input_size,
            in_channels=args.image_channels, double_z=args.double_z,
        )
    else:
        vae = VAE(
            in_channels=args.image_channels, latent_dim=args.latent_dim,
            ch=args.ch, ch_mult=args.ch_mult, num_res_blocks=args.num_res_blocks,
            attn_resolutions=args.attn_resolutions, resolution=args.input_size,
            double_z=args.double_z,
        )

    # Label encoder
    sle_variant = args.label_encoder_variant
    if sle_variant in SpatialLabelEncoder_models:
        label_encoder = SpatialLabelEncoder_models[sle_variant](
            input_size=args.input_size, num_classes=args.num_classes,
            latent_dim=args.latent_dim, init_logsigma=args.label_init_logsigma,
        )
    else:
        model_cfg = args.yaml_config.get('model', {})
        label_encoder = SpatialLabelEncoder(
            input_size=args.input_size, patch_size=16,
            num_classes=args.num_classes, latent_dim=args.latent_dim,
            hidden_size=model_cfg.get('label_hidden_size', 768),
            depth=model_cfg.get('label_depth', 12),
            num_heads=model_cfg.get('label_num_heads', 12),
            mlp_ratio=model_cfg.get('label_mlp_ratio', 4.0),
            in_context_len=model_cfg.get('label_in_context_len', 32),
            in_context_start=model_cfg.get('label_in_context_start', 4),
            init_logsigma=args.label_init_logsigma,
        )

    # Assemble
    model = AlignmentVAE(
        input_size=args.input_size,
        latent_dim=args.latent_dim,
        image_channels=args.image_channels,
        num_classes=args.num_classes,
        vae=vae,
        label_encoder=label_encoder,
        weight_recon=args.weight_recon,
        weight_kl=args.weight_kl,
        weight_alignment=args.weight_alignment,
        weight_label_entropy=args.weight_label_entropy,
    )

    # Print summary
    enc_params = sum(p.numel() for p in model.vae.encoder.parameters())
    dec_params = sum(p.numel() for p in model.vae.decoder.parameters())
    vae_params = sum(p.numel() for p in model.vae.parameters())
    sle_params = sum(p.numel() for p in model.label_encoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    h = args.input_size // 16
    sep = "─" * 52
    print(sep)
    print(f"  {'Component':<28s} {'Variant':<10s} {'Params':>10s}")
    print(sep)
    print(f"  {'VAE Encoder':<28s} {vae_variant:<10s} {_fmt_params(enc_params):>10s}")
    print(f"  {'VAE Decoder':<28s} {'':10s} {_fmt_params(dec_params):>10s}")
    print(f"  {'VAE (total)':<28s} {'':10s} {_fmt_params(vae_params):>10s}")
    print(f"  {'Label Encoder':<28s} {sle_variant:<10s} {_fmt_params(sle_params):>10s}")
    print(sep)
    print(f"  {'Total':<28s} {'':10s} {_fmt_params(total_params):>10s}")
    print(sep)
    print(f"  Latent shape: [B, {h}, {h}, {args.latent_dim}]")
    print()

    return model


# ============================================================================
# Training Engine
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_writer=None,
    args=None,
):
    """Train for one epoch with per-iteration LR scheduling and gradient accumulation."""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'[Epoch {epoch + 1}/{args.epochs}]'
    print_freq = args.log_freq
    accum_iter = getattr(args, 'accum_iter', 1)

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Only adjust LR at the boundary of each accumulation window
        if step % accum_iter == 0:
            fractional_epoch = step / len(data_loader) + epoch
            lr_sched.adjust_learning_rate(optimizer, fractional_epoch, args)

        # KL annealing: ramp KL weight from 0 → 1 over kl_anneal_epochs
        fractional_epoch = step / len(data_loader) + epoch
        kl_anneal_factor = get_kl_anneal_factor(fractional_epoch, args)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Use no_sync context for non-update steps in DDP to avoid redundant all-reduce
        if accum_iter > 1 and hasattr(model, 'no_sync'):
            ctx = model.no_sync if (step + 1) % accum_iter != 0 else nullcontext
        else:
            ctx = nullcontext

        with ctx():
            x_recon, loss, metrics = model(images, labels, kl_anneal_factor=kl_anneal_factor)
            loss = loss / accum_iter  # Normalize loss by accumulation steps

        loss_value = loss.item() * accum_iter  # Un-scale for logging (show true per-sample loss)
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        loss.backward()

        # Optimizer step at the end of each accumulation window
        if (step + 1) % accum_iter == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_recon=metrics.get('loss_recon', 0))
        metric_logger.update(loss_align=metrics.get('loss_alignment', 0))
        metric_logger.update(loss_kl=metrics.get('loss_kl_vae', 0))
        metric_logger.update(loss_label_ent=metrics.get('loss_label_entropy', 0))
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(kl_anneal=kl_anneal_factor)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and step % args.log_freq == 0:
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/loss_recon', metrics.get('loss_recon', 0), epoch_1000x)
            log_writer.add_scalar('train/loss_alignment', metrics.get('loss_alignment', 0), epoch_1000x)
            log_writer.add_scalar('train/loss_kl_vae', metrics.get('loss_kl_vae', 0), epoch_1000x)
            log_writer.add_scalar('train/loss_label_entropy', metrics.get('loss_label_entropy', 0), epoch_1000x)
            log_writer.add_scalar('train/mean_sigma_label', metrics.get('mean_sigma_label', 0), epoch_1000x)
            log_writer.add_scalar('train/kl_anneal_factor', kl_anneal_factor, epoch_1000x)
            log_writer.add_scalar('train/lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    metrics_computer: MetricsComputer,
    compute_fid: bool = False,
    fid_num_samples: int = 10000,
    header: str = "Val:",
):
    """Evaluate: loss, MSE, PSNR, SSIM, recon/gen FID."""
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")

    feats_real_list, feats_recon_list, feats_gen_list = [], [], []
    fid_samples_collected = 0

    model_unwrapped = model.module if hasattr(model, 'module') else model

    for images, labels in metric_logger.log_every(data_loader, 50, header):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x_recon, loss, metrics = model(images, labels)

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_recon=metrics.get('loss_recon', 0))
        metric_logger.update(loss_align=metrics.get('loss_alignment', 0))
        metric_logger.update(loss_kl=metrics.get('loss_kl_vae', 0))
        metric_logger.update(loss_label_ent=metrics.get('loss_label_entropy', 0))

        metric_logger.update(mse=metrics_computer.compute_mse(images, x_recon))
        metric_logger.update(psnr=metrics_computer.compute_psnr(images, x_recon))
        metric_logger.update(ssim=metrics_computer.compute_ssim(images, x_recon))

        # Collect inception features for FID
        if (metrics_computer.inception is not None
                and fid_samples_collected < fid_num_samples):
            feats_real_list.append(metrics_computer.extract_inception_features(images))
            feats_recon_list.append(metrics_computer.extract_inception_features(x_recon))
            if compute_fid:
                x_gen = model_unwrapped.generate_from_label(labels)
                feats_gen_list.append(metrics_computer.extract_inception_features(x_gen))
            fid_samples_collected += images.shape[0]

    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Compute FID from collected features
    if feats_real_list:
        feats_real = np.concatenate(feats_real_list, axis=0)
        feats_recon = np.concatenate(feats_recon_list, axis=0)
        feats_gen = np.concatenate(feats_gen_list, axis=0) if feats_gen_list else None

        # DDP gather
        if misc.get_world_size() > 1:
            feats_real, feats_recon, feats_gen = _gather_fid_features(
                feats_real, feats_recon, feats_gen, device
            )

        if misc.is_main_process():
            n = len(feats_real)
            try:
                recon_fid = compute_fid_from_features(feats_real, feats_recon)
                results['recon_fid'] = recon_fid
                print(f"  Recon FID (image→image, {n} samples): {recon_fid:.2f}")
            except Exception as e:
                print(f"  [WARN] Recon FID failed: {e}")

            if feats_gen is not None:
                try:
                    gen_fid = compute_fid_from_features(feats_real, feats_gen)
                    results['gen_fid'] = gen_fid
                    print(f"  Gen FID  (label→image, {n} samples): {gen_fid:.2f}")
                except Exception as e:
                    print(f"  [WARN] Gen FID failed: {e}")

    print(f"{header}", metric_logger)
    return results


def _gather_fid_features(feats_real, feats_recon, feats_gen, device):
    """all_gather FID features across DDP ranks."""
    def _gather(arr):
        t = torch.from_numpy(arr).to(device)
        gathered = [torch.zeros_like(t) for _ in range(misc.get_world_size())]
        torch.distributed.all_gather(gathered, t)
        return torch.cat(gathered, dim=0).cpu().numpy()

    feats_real = _gather(feats_real)
    feats_recon = _gather(feats_recon)
    if feats_gen is not None:
        feats_gen = _gather(feats_gen)
    return feats_real, feats_recon, feats_gen


# ============================================================================
# Config Printing
# ============================================================================

def _print_config(args):
    """Print hyperparameters in a grouped table."""
    sep = "=" * 58

    def _section(title, entries):
        print(f"  ┌─ {title}")
        for k, v in entries:
            print(f"  │  {k:<28s} {v}")
        print(f"  └{'─' * 40}")

    print(sep)
    print("  Configuration")
    print(sep)

    _section("Model", [
        ("vae_variant",           str(getattr(args, 'vae_variant', 'VAE-B'))),
        ("label_encoder_variant", str(getattr(args, 'label_encoder_variant', 'SLE-B'))),
        ("input_size",            str(args.input_size)),
        ("latent_dim",            str(args.latent_dim)),
        ("image_channels",        str(args.image_channels)),
        ("num_classes",           str(args.num_classes)),
        ("double_z",              str(args.double_z)),
    ])

    eff_bs = args.batch_size * args.accum_iter * getattr(args, 'world_size', 1)
    _section("Training", [
        ("epochs",               str(args.epochs)),
        ("batch_size (per GPU)", str(args.batch_size)),
        ("accum_iter",           str(args.accum_iter)),
        ("effective batch size", str(eff_bs)),
        ("lr / lr_label_encoder", f"{args.lr:.1e} / {args.lr_label_encoder:.1e}"),
        ("lr_schedule",          str(args.lr_schedule)),
        ("warmup_epochs",        str(args.warmup_epochs)),
        ("weight_decay",         str(args.weight_decay)),
        ("grad_clip",            str(args.grad_clip)),
    ])

    _section("Loss Weights", [
        ("recon",               str(args.weight_recon)),
        ("kl",                  str(args.weight_kl)),
        ("alignment",           str(args.weight_alignment)),
        ("label_entropy",       str(args.weight_label_entropy)),
        ("alignment_temp",      str(args.alignment_temp)),
        ("label_init_logsigma", str(args.label_init_logsigma)),
        ("kl_anneal_strategy",  str(args.kl_anneal_strategy)),
        ("kl_anneal_epochs",    str(args.kl_anneal_epochs)),
    ])

    _section("Data", [
        ("data_path",           str(args.data_path)),
        ("val_split",           str(args.val_split)),
        ("num_workers",         str(args.num_workers)),
    ])

    _section("Distributed", [
        ("world_size",          str(getattr(args, 'world_size', 1))),
        ("distributed",         str(getattr(args, 'distributed', False))),
        ("device",              str(args.device)),
    ])

    _section("Checkpoint", [
        ("output_dir",          str(args.output_dir)),
        ("resume",              str(args.resume) if args.resume else "(none)"),
        ("save_freq",           str(args.save_freq)),
    ])

    print(sep)


# ============================================================================
# Main
# ============================================================================

def main(args):
    misc.init_distributed_mode(args)
    print(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    _print_config(args)

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Resolve learning rate (effective batch accounts for accumulation)
    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    if args.blr is not None:
        args.lr = args.blr * eff_batch_size / 256
    lr_scale_label = args.lr_label_encoder / args.yaml_config.get('training', {}).get('lr', 1e-4)

    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    # TensorBoard
    if global_rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    else:
        log_writer = None

    # Dataset
    transform_train = transforms.Compose([
        transforms.Resize(int(args.input_size * 1.1)),
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(args.input_size * 1.1)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageLabelDataset(
        root=args.data_path, split="train",
        transform=transform_train, num_classes=args.num_classes,
    )

    total_len = len(full_dataset)
    val_len = int(total_len * args.val_split)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Dataset: train={train_len}, val={val_len}")

    # Samplers
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print(f"Sampler: DistributedSampler (rank={global_rank}, world={num_tasks})")
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem,
    )
    data_loader_train = DataLoader(train_dataset, sampler=sampler_train, drop_last=True, **loader_kwargs)
    data_loader_val = DataLoader(val_dataset, sampler=sampler_val, drop_last=False, **loader_kwargs)

    # Model
    model = build_model(args)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Optimizer (separate LR for label encoder)
    optimizer = torch.optim.AdamW([
        {'params': model_without_ddp.vae.parameters(), 'lr': args.lr},
        {'params': model_without_ddp.label_encoder.parameters(),
         'lr': args.lr, 'lr_scale': lr_scale_label},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Metrics
    metrics_computer = MetricsComputer(
        device=str(device),
        compute_fid=args.compute_fid,
    )

    # Resume
    if args.resume:
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "checkpoint-last.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_without_ddp.load_state_dict(ckpt['model'])
            print(f"Resumed model from: {ckpt_path}")
            if 'optimizer' in ckpt and 'epoch' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                args.start_epoch = ckpt['epoch'] + 1
                print(f"Resumed optimizer & epoch (start_epoch={args.start_epoch})")
            del ckpt
        else:
            print(f"[WARN] Checkpoint not found: {ckpt_path}. Training from scratch.")

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_writer=log_writer,
            args=args,
        )

        val_stats = evaluate(
            model, data_loader_val, device, metrics_computer,
            compute_fid=args.compute_fid,
            fid_num_samples=args.fid_num_samples,
            header="Val:",
        )

        # TensorBoard logging
        if log_writer is not None:
            for k, v in train_stats.items():
                log_writer.add_scalar(f'epoch_train/{k}', v, epoch)
            for k, v in val_stats.items():
                log_writer.add_scalar(f'epoch_val/{k}', v, epoch)

        # Checkpointing
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp,
                optimizer=optimizer, epoch=epoch, epoch_name="last",
            )
        if epoch > 0 and epoch % 50 == 0:
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp,
                optimizer=optimizer, epoch=epoch,
            )

        val_loss = val_stats.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp,
                optimizer=optimizer, epoch=epoch, epoch_name="best",
            )
            print(f"New best model saved (val_loss={best_val_loss:.4f})")

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training completed! Total time: {total_time}")

    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[OK] Config loaded: {args.config}")
    else:
        print(f"[WARN] Config not found: {args.config}, using defaults")
        cfg = {}

    args = merge_config_and_args(cfg, args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
