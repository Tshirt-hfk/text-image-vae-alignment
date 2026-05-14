"""
Training script for AlignmentVAE — single-machine multi-GPU (DDP).

    Image → VAE Encoder              → N(μ, σ²)  [B, h, w, D]
    Label → EmbeddingLabelEncoder    → N(μ, σ²)  [B, h, w, D]   (cond_type='label')
    Text  → SpatialTextEncoder       → N(μ, σ²)  [B, h, w, D]   (cond_type='text')

Loss = MSE recon + VAE KL + Alignment KL + Label entropy

Usage:
    python train.py --config configs/imagenet_l2i.yaml
    torchrun --nproc_per_node=4 train.py --config configs/imagenet_l2i.yaml
"""

import argparse
import datetime
import math
import numpy as np
import os
import sys
import time
import warnings
import yaml
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple, Optional

# Suppress DDP stride-mismatch warning for 1x1 Conv weights (cuDNN backward
# produces channels_last grads; harmless for H=W=1 since memory layout is identical)
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import AlignmentVAE
from model.label_encoder import EmbeddingLabelEncoder
from model.text_encoder import (
    SpatialTextEncoder, SpatialTextEncoder_models, TextEncoderWrapper,
)
from model.vae import VAE, VAE_models
from dataset import ImageLabelDataset

import util.misc as misc
import util.lr_sched as lr_sched


# ============================================================================
# Argument Parser
# ============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('AlignmentVAE Training', add_help=True)

    # config file
    parser.add_argument('--config', type=str, default='configs/imagenet_l2i.yaml')

    # architecture overrides
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--vae_variant', type=str, default=None,
                        help='VAE-B | VAE-L | VAE-H')
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

    # dataset
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # performance
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Enable AMP (automatic mixed precision) training')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable AMP training')
    parser.add_argument('--amp_dtype', type=str, default="bf16",
                        choices=['fp16', 'bf16', 'fp32'],
                        help='AMP autocast dtype. fp16 (default, needs GradScaler) | '
                             'bf16 (Ampere/Hopper, no GradScaler, more numerically stable) | '
                             'fp32 (disable AMP). Overrides yaml `amp_dtype`.')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Enable torch.compile for model optimization (PyTorch 2.0+)')

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
    args.num_classes = args.num_classes or model_cfg.get('num_classes', 1000)
    args.image_channels = model_cfg.get('image_channels', 3)
    args.ch = model_cfg.get('ch', 128)
    args.ch_mult = model_cfg.get('ch_mult', [1, 2, 4, 4])
    args.num_res_blocks = model_cfg.get('num_res_blocks', 2)
    args.attn_resolutions = model_cfg.get('attn_resolutions', [16])
    args.double_z = model_cfg.get('double_z', True)
    args.label_init_logsigma = model_cfg.get('label_init_logsigma', -2.0)

    # ── 条件编码器选择 ────────────────────────────────────────
    # cond_type:
    #   'label' → EmbeddingLabelEncoder              (默认；ImageNet 等离散类条件)
    #   'text'  → TextEncoderWrapper + SpatialTextEncoder (COCO/CC3M T2I)
    args.cond_type = model_cfg.get('cond_type', 'label')
    assert args.cond_type in ('label', 'text'), \
        f"cond_type must be one of 'label' | 'text', got {args.cond_type!r}"

    # ── T2I 扩展配置（仅 cond_type='text' 时生效） ───────────
    args.text_encoder_name = model_cfg.get('text_encoder_name', 'clip')
    args.text_encoder_pretrained = model_cfg.get('text_encoder_pretrained', None)
    args.text_encoder_freeze = bool(model_cfg.get('text_encoder_freeze', True))
    args.max_text_len = int(model_cfg.get('max_text_len', 77))
    args.text_encoder_variant = model_cfg.get('text_encoder_variant', 'STE-B')

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
    args.weight_alignment = float(train_cfg.get('weight_alignment', 0.1))
    args.weight_alignment_reverse = float(train_cfg.get('weight_alignment_reverse', 0.0))
    args.alignment_temp = float(train_cfg.get('alignment_temp', 1.0))

    # Dataset
    args.data_path = args.data_path or train_cfg.get('data_root', './data/imagenet')
    # dataset_type: 'imagenet' (旧) | 'text_image' (新)
    args.dataset_type = train_cfg.get('dataset_type', 'imagenet')
    args.dataset_format = train_cfg.get('dataset_format', 'auto')   # coco | csv | jsonl | auto

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

    # Performance (CLI flags override YAML)
    if not hasattr(args, 'use_amp') or args.use_amp is True:
        args.use_amp = train_cfg.get('use_amp', True)
    if not hasattr(args, 'use_compile') or args.use_compile is False:
        args.use_compile = train_cfg.get('use_compile', False)

    # AMP dtype: CLI > YAML > default('fp16'); 'fp32' implies use_amp=False
    cli_amp_dtype = getattr(args, 'amp_dtype', None)
    yaml_amp_dtype = train_cfg.get('amp_dtype', None)
    args.amp_dtype = (cli_amp_dtype or yaml_amp_dtype or 'fp16').lower()
    assert args.amp_dtype in ('fp16', 'bf16', 'fp32'), \
        f"amp_dtype must be 'fp16' | 'bf16' | 'fp32', got {args.amp_dtype!r}"
    if args.amp_dtype == 'fp32':
        args.use_amp = False

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
    """Evaluation metrics: MSE, PSNR, SSIM, Inception features.

    PSNR is sensitive to data range. When inputs are normalized with ImageNet
    mean/std (typical range ≈ [-2.12, 2.64]), the previous heuristic
    `max_val = 1.0 if x.max() <= 1.0 else 255.0` incorrectly fell back to 255,
    giving wildly inflated PSNR values (~57 dB instead of true ~9 dB).

    To get a meaningful PSNR, this class accepts the normalization stats
    (`mean`, `std`) used by the data pipeline and de-normalizes tensors back
    to [0, 1] before computing PSNR (with MAX=1.0).
    """

    def __init__(
        self,
        device: str = "cuda",
        compute_fid: bool = False,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ):
        self.device = device
        self._ssim_kernel_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self.inception = InceptionV3Features(device) if compute_fid else None

        # Cache (mean, std) tensors for de-normalization. Shape [1, C, 1, 1] so
        # they broadcast with [B, C, H, W] inputs without per-call allocation.
        if mean is not None and std is not None:
            self._denorm_mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
            self._denorm_std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        else:
            self._denorm_mean = None
            self._denorm_std = None

    def _denormalize_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        """De-normalize x using stored (mean, std) and clamp to [0, 1].

        If no stats were provided, returns x unchanged so the caller can
        fall back to its own heuristic. Channel count must match.
        """
        if self._denorm_mean is None or self._denorm_std is None:
            return x
        mean = self._denorm_mean.to(dtype=x.dtype, device=x.device)
        std = self._denorm_std.to(dtype=x.dtype, device=x.device)
        return (x * std + mean).clamp_(0.0, 1.0)

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
        """Peak Signal-to-Noise Ratio with consistent data range.

        Both `x` and `x_recon` are first de-normalized to [0, 1] using the
        cached (mean, std) so MSE and MAX live in the same numeric space.
        Without this, ImageNet-normalized inputs (range ≈ [-2.12, 2.64])
        would mis-trigger the old `max_val=255` branch, producing nonsense
        PSNR values (e.g. ~57 dB) that don't reflect true reconstruction quality.

        Falls back to the original heuristic only when no normalization stats
        are configured (e.g., legacy callers passing already-unit-range tensors).
        """
        if self._denorm_mean is not None and self._denorm_std is not None:
            x_unit = self._denormalize_to_unit(x)
            x_recon_unit = self._denormalize_to_unit(x_recon)
            mse = torch.mean((x_unit - x_recon_unit) ** 2)
            if mse == 0:
                return float('inf')
            # MAX = 1.0 because both tensors are now clamped to [0, 1].
            return (-10.0 * torch.log10(mse)).item()

        # Legacy fallback: assume caller-provided tensors are already in
        # either [0, 1] or [0, 255]. Kept for backward compatibility.
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
# Batch helpers — 统一处理 (image, label) 与 dict-batch 两种格式
# ============================================================================

def _unpack_batch(batch):
    """
    返回 (images, cond)
      - 旧格式 (image, label) tuple/list → cond 为 LongTensor[B]
      - 新格式 dict {image, input_ids, attention_mask} → cond 为 dict
    """
    if isinstance(batch, dict):
        image = batch["image"]
        cond = {k: v for k, v in batch.items() if k != "image"}
        return image, cond
    # tuple / list
    return batch[0], batch[1]


def _cond_to_device(cond, device):
    """把 cond 中的所有 tensor 搬到 device。"""
    if isinstance(cond, dict):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in cond.items()}
    return cond.to(device, non_blocking=True)


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

    # ── Conditioning encoder ─────────────────────────────────────────────────
    cond_type = getattr(args, 'cond_type', 'label')

    if cond_type == 'text':
        # —— T2I 模式：HF 文本编码器 + SpatialTextEncoder ——
        text_encoder_wrapper = TextEncoderWrapper(
            model_name=args.text_encoder_name,
            pretrained_path=args.text_encoder_pretrained,
            freeze=args.text_encoder_freeze,
        )
        text_dim = text_encoder_wrapper.hidden_dim

        ste_variant = args.text_encoder_variant
        if ste_variant in SpatialTextEncoder_models:
            label_encoder = SpatialTextEncoder_models[ste_variant](
                input_size=args.input_size, latent_dim=args.latent_dim,
                text_dim=text_dim, max_text_len=args.max_text_len,
                text_encoder=text_encoder_wrapper,
                init_logsigma=args.label_init_logsigma,
            )
        else:
            model_cfg = args.yaml_config.get('model', {})
            label_encoder = SpatialTextEncoder(
                input_size=args.input_size, patch_size=16,
                text_dim=text_dim, max_text_len=args.max_text_len,
                text_encoder=text_encoder_wrapper,
                hidden_size=model_cfg.get('text_hidden_size', 768),
                latent_dim=args.latent_dim,
                depth=model_cfg.get('text_depth', 12),
                num_heads=model_cfg.get('text_num_heads', 12),
                mlp_ratio=model_cfg.get('text_mlp_ratio', 4.0),
                in_context_start=model_cfg.get('text_in_context_start', 4),
                init_logsigma=args.label_init_logsigma,
            )
        sle_variant = f"{ste_variant} ({args.text_encoder_name})"

    else:
        # —— Label 模式：EmbeddingLabelEncoder（轻量、无 Transformer） ——
        sle_variant = 'EmbedLabel'
        label_encoder = EmbeddingLabelEncoder(
            input_size=args.input_size, patch_size=16,
            num_classes=args.num_classes, latent_dim=args.latent_dim,
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
        weight_alignment=args.weight_alignment,
        weight_alignment_reverse=args.weight_alignment_reverse,
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
    scaler: Optional[GradScaler] = None,
    log_writer=None,
    args=None,
    amp_dtype: torch.dtype = torch.float16,
):
    """Train for one epoch with AMP, per-iteration LR scheduling and gradient accumulation."""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'[Epoch {epoch + 1}/{args.epochs}]'
    print_freq = args.log_freq
    accum_iter = getattr(args, 'accum_iter', 1)
    use_amp = getattr(args, 'use_amp', True) and device.type == 'cuda'

    # Use set_to_none=False with gradient_as_bucket_view=True:
    # DDP bucket views serve as .grad storage, zero_grad() clears them in-place,
    # backward accumulates directly into buckets — avoids cuDNN channels_last stride mismatch.
    optimizer.zero_grad()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, cond = _unpack_batch(batch)

        # Only adjust LR at the boundary of each accumulation window
        if step % accum_iter == 0:
            fractional_epoch = step / len(data_loader) + epoch
            lr_sched.adjust_learning_rate(optimizer, fractional_epoch, args)

        images = images.to(device, non_blocking=True)
        cond = _cond_to_device(cond, device)

        # Use no_sync context for non-update steps in DDP to avoid redundant all-reduce
        if accum_iter > 1 and hasattr(model, 'no_sync'):
            sync_ctx = model.no_sync if (step + 1) % accum_iter != 0 else nullcontext
        else:
            sync_ctx = nullcontext

        with sync_ctx():
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                x_recon, loss, metrics = model(images, cond)
                loss = loss / accum_iter  # Normalize loss by accumulation steps

        loss_value = loss.item() * accum_iter  # Un-scale for logging (show true per-sample loss)
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        # AMP-aware backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step at the end of each accumulation window
        if (step + 1) % accum_iter == 0:
            if scaler is not None:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        # Update local metrics (removed torch.cuda.synchronize() — unnecessary bottleneck)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_recon=metrics.get('loss_recon', 0))
        metric_logger.update(loss_align=metrics.get('loss_alignment', 0))
        metric_logger.update(loss_align_rev=metrics.get('loss_kl_reverse', 0))
        metric_logger.update(sigma_img=metrics.get('mean_sigma_img', 0))
        metric_logger.update(sigma_lbl=metrics.get('mean_sigma_label', 0))
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # all_reduce must be called by ALL ranks (collective op), but only log on rank 0
        if step % args.log_freq == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None:
                epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train/loss_recon', metrics.get('loss_recon', 0), epoch_1000x)
                log_writer.add_scalar('train/loss_alignment', metrics.get('loss_alignment', 0), epoch_1000x)
                log_writer.add_scalar('train/loss_kl_reverse', metrics.get('loss_kl_reverse', 0), epoch_1000x)
                log_writer.add_scalar('train/mean_sigma_label', metrics.get('mean_sigma_label', 0), epoch_1000x)
                log_writer.add_scalar('train/mean_sigma_img', metrics.get('mean_sigma_img', 0), epoch_1000x)
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
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
):
    """Evaluate: loss, MSE, PSNR, SSIM, recon/gen FID."""
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    use_amp = use_amp and device.type == 'cuda'

    feats_real_list, feats_recon_list, feats_gen_list = [], [], []
    fid_samples_collected = 0

    model_unwrapped = model.module if hasattr(model, 'module') else model

    for batch in metric_logger.log_every(data_loader, 50, header):
        images, cond = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        cond = _cond_to_device(cond, device)

        with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            x_recon, loss, metrics = model(images, cond)

        # Cast back to float32 for metric computation
        x_recon = x_recon.float()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_recon=metrics.get('loss_recon', 0))
        metric_logger.update(loss_align=metrics.get('loss_alignment', 0))
        metric_logger.update(loss_align_rev=metrics.get('loss_kl_reverse', 0))
        metric_logger.update(sigma_img=metrics.get('mean_sigma_img', 0))
        metric_logger.update(sigma_lbl=metrics.get('mean_sigma_label', 0))

        metric_logger.update(mse=metrics_computer.compute_mse(images, x_recon))
        metric_logger.update(psnr=metrics_computer.compute_psnr(images, x_recon))
        metric_logger.update(ssim=metrics_computer.compute_ssim(images, x_recon))

        # Collect inception features for FID
        if (metrics_computer.inception is not None
                and fid_samples_collected < fid_num_samples):
            feats_real_list.append(metrics_computer.extract_inception_features(images))
            feats_recon_list.append(metrics_computer.extract_inception_features(x_recon))
            if compute_fid:
                x_gen = model_unwrapped.generate_from_label(cond)
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

    cond_type = getattr(args, 'cond_type', 'label')
    if cond_type == 'text':
        cond_encoder_str = f"{getattr(args, 'text_encoder_variant', 'STE-B')} " \
                           f"({getattr(args, 'text_encoder_name', 'clip')}, " \
                           f"frozen={getattr(args, 'text_encoder_freeze', True)})"
    else:
        cond_encoder_str = "EmbeddingLabelEncoder"

    _section("Model", [
        ("vae_variant",           str(getattr(args, 'vae_variant', 'VAE-B'))),
        ("cond_type",             cond_type),
        ("cond_encoder",          cond_encoder_str),
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
        ("alignment (fwd)",     str(args.weight_alignment)),
        ("alignment (rev)",     str(args.weight_alignment_reverse)),
        ("alignment_temp",      str(args.alignment_temp)),
        ("label_init_logsigma", str(args.label_init_logsigma)),
    ])

    _section("Data", [
        ("data_path",           str(args.data_path)),
        ("num_workers",         str(args.num_workers)),
    ])

    _section("Distributed & Performance", [
        ("world_size",          str(getattr(args, 'world_size', 1))),
        ("distributed",         str(getattr(args, 'distributed', False))),
        ("device",              str(args.device)),
        ("use_amp",             str(getattr(args, 'use_amp', True))),
        ("amp_dtype",           str(getattr(args, 'amp_dtype', 'fp16'))),
        ("use_compile",         str(getattr(args, 'use_compile', False))),
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

    # AMP / torch.compile flags
    use_amp = getattr(args, 'use_amp', True) and device.type == 'cuda'
    use_compile = getattr(args, 'use_compile', False)

    # Resolve AMP autocast dtype (fp16 | bf16 | fp32).
    #   - fp16: needs GradScaler to avoid gradient underflow
    #   - bf16: same exponent range as fp32, no GradScaler required, much
    #           more numerically stable for KL/exp/log heavy losses
    #   - fp32: AMP fully disabled
    amp_dtype_str = getattr(args, 'amp_dtype', 'fp16')
    if amp_dtype_str == 'bf16':
        amp_dtype = torch.bfloat16
    elif amp_dtype_str == 'fp16':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    # Warn & fall back if hardware lacks native bf16 support (e.g. V100/T4)
    if use_amp and amp_dtype == torch.bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("[WARN] amp_dtype='bf16' requested but current CUDA device "
                  "does NOT natively support bfloat16. Falling back to fp16.")
            amp_dtype = torch.float16
    use_grad_scaler = use_amp and (amp_dtype == torch.float16)

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
    # NOTE: Keep these stats in sync with `MetricsComputer(mean=..., std=...)`
    # below so PSNR de-normalization uses the same constants as the data
    # pipeline. Mismatching them would silently corrupt PSNR readings again.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.Resize(int(args.input_size * 1.1)),
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(imagenet_mean), std=list(imagenet_std)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(args.input_size * 1.1)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(imagenet_mean), std=list(imagenet_std)),
    ])

    if getattr(args, 'dataset_type', 'imagenet') == 'text_image':
        # —— T2I: COCO Captions / CC3M / 自定义图文对 ——
        from dataset import build_text_image_dataset
        # 复用 text encoder 同源 tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = (
            args.text_encoder_pretrained
            or TextEncoderWrapper.SUPPORTED.get(args.text_encoder_name, args.text_encoder_name)
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        train_dataset = build_text_image_dataset(
            root=args.data_path, split="train", tokenizer=tokenizer,
            format=args.dataset_format, max_text_len=args.max_text_len,
            input_size=args.input_size,
        )
        val_dataset = build_text_image_dataset(
            root=args.data_path, split="val", tokenizer=tokenizer,
            format=args.dataset_format, max_text_len=args.max_text_len,
            input_size=args.input_size,
        )
    else:
        # —— 原 ImageNet 流程 ——
        train_dataset = ImageLabelDataset(
            root=args.data_path, split="train",
            transform=transform_train, num_classes=args.num_classes,
        )
        val_dataset = ImageLabelDataset(
            root=args.data_path, split="val",
            transform=transform_val, num_classes=args.num_classes,
        )

    print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)} "
          f"[type={getattr(args, 'dataset_type', 'imagenet')}]")

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
        persistent_workers=args.num_workers > 0,  # Keep worker processes alive between epochs
        prefetch_factor=4 if args.num_workers > 0 else None,  # Pre-fetch more batches
    )
    data_loader_train = DataLoader(train_dataset, sampler=sampler_train, drop_last=True, **loader_kwargs)
    data_loader_val = DataLoader(val_dataset, sampler=sampler_val, drop_last=False, **loader_kwargs)

    # Model
    model = build_model(args)
    model.to(device)

    # torch.compile for kernel fusion & graph optimization (PyTorch 2.0+)
    if use_compile:
        print("Compiling model with torch.compile (mode='reduce-overhead')...")
        model = torch.compile(model, mode='reduce-overhead')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False,
            gradient_as_bucket_view=True)  # Reuse comm buffer as grad storage
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Optimizer (separate LR for label encoder; fused=True for CUDA-accelerated AdamW)
    try:
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    except Exception:
        fused_available = False
    use_fused = fused_available and device.type == 'cuda'
    extra_optim_kwargs = {'fused': True} if use_fused else {}
    # 仅把 requires_grad=True 的参数交给优化器
    # （T2I 模式下文本编码器通常 frozen，避免 AdamW 为其分配 state）
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model_without_ddp.vae.parameters() if p.requires_grad],
         'lr': args.lr},
        {'params': [p for p in model_without_ddp.label_encoder.parameters() if p.requires_grad],
         'lr': args.lr, 'lr_scale': lr_scale_label},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999),
       **extra_optim_kwargs)
    if use_fused:
        print("Using fused AdamW optimizer (CUDA-accelerated)")

    # AMP GradScaler — only fp16 needs grad scaling; bf16 has fp32-equivalent
    # range so its gradients won't underflow.
    scaler = GradScaler('cuda', enabled=use_grad_scaler) if use_grad_scaler else None
    if use_amp:
        print(f"Using AMP (automatic mixed precision): "
              f"dtype={amp_dtype_str}, GradScaler={'on' if use_grad_scaler else 'off'}")
    else:
        print("AMP disabled (training in fp32)")

    # Metrics
    # Pass the same (mean, std) used by transform_{train,val} so PSNR can
    # de-normalize tensors back to [0, 1] before computing 10·log10(1/MSE).
    metrics_computer = MetricsComputer(
        device=str(device),
        compute_fid=args.compute_fid,
        mean=imagenet_mean,
        std=imagenet_std,
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
            if scaler is not None and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
                print("Resumed AMP scaler state")
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
            scaler=scaler,
            log_writer=log_writer,
            args=args,
            amp_dtype=amp_dtype,
        )

        val_stats = evaluate(
            model, data_loader_val, device, metrics_computer,
            compute_fid=args.compute_fid,
            fid_num_samples=args.fid_num_samples,
            header="Val:",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
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
                scaler=scaler,
            )
        if epoch > 0 and epoch % 50 == 0:
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp,
                optimizer=optimizer, epoch=epoch, scaler=scaler,
            )

        val_loss = val_stats.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp,
                optimizer=optimizer, epoch=epoch, epoch_name="best",
                scaler=scaler,
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
