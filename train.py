"""
Training script for AlignmentVAE — built on Hugging Face accelerate.

    Image → VAE Encoder              → N(μ, σ²)  [B, h, w, D]
    Label → EmbeddingLabelEncoder    → N(μ, σ²)  [B, h, w, D]   (cond_type='label')
    Text  → SpatialTextEncoder       → N(μ, σ²)  [B, h, w, D]   (cond_type='text')

Loss = MSE recon + Alignment KL (forward) + Alignment KL (reverse, optional)

`accelerate` transparently handles DDP, AMP autocast, GradScaler (fp16 only),
gradient accumulation + DDP no_sync, and checkpoint sharding. We keep this
script focused on business logic.

Usage:
    # 0) (one-time) generate a default accelerate config
    accelerate config

    # 1) Single GPU
    accelerate launch --num_processes 1 train.py --config configs/imagenet_l2i.yaml

    # 2) 8-GPU single node
    accelerate launch --multi_gpu --num_processes 8 \
        train.py --config configs/imagenet_l2i.yaml

    # 3) Override mixed precision on the fly (bf16 on Ampere+ recommended)
    accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
        train.py --config configs/imagenet_l2i.yaml
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
from pathlib import Path
from typing import Dict, Tuple, Optional

# Suppress DDP stride-mismatch warning for 1x1 Conv weights (cuDNN backward
# produces channels_last grads; harmless for H=W=1 since memory layout is identical)
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

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
    parser = argparse.ArgumentParser('AlignmentVAE Training (accelerate)', add_help=True)

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
                        help='Gradient accumulation steps. Effective batch = batch_size * accum_iter * num_processes')
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

    # performance — accelerate handles AMP/DDP. We only declare dtype here and
    # forward it to Accelerator(mixed_precision=...).
    parser.add_argument('--amp_dtype', type=str, default=None,
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Mixed precision dtype. fp16 (needs GradScaler) | '
                             'bf16 (Ampere+, recommended) | fp32 (disable AMP). '
                             'Maps to accelerate `mixed_precision` (no/fp16/bf16). '
                             'Overrides yaml `amp_dtype` and `accelerate launch --mixed_precision`.')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Enable torch.compile (PyTorch 2.0+)')

    # misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output_dir')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a checkpoint dir saved by accelerator.save_state(); '
                             'if it points to {output_dir}, the latest "checkpoint-last" '
                             'subdir will be picked up automatically.')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=None)
    parser.add_argument('--log_freq', type=int, default=None)

    return parser


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
    args.cond_type = model_cfg.get('cond_type', 'label')
    assert args.cond_type in ('label', 'text'), \
        f"cond_type must be one of 'label' | 'text', got {args.cond_type!r}"

    # ── T2I 扩展配置 ────────────────────────────────────────
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

    # Loss weights
    args.weight_recon = float(train_cfg.get('weight_recon', 1.0))
    args.weight_alignment = float(train_cfg.get('weight_alignment', 0.1))
    args.weight_alignment_reverse = float(train_cfg.get('weight_alignment_reverse', 0.0))
    args.alignment_temp = float(train_cfg.get('alignment_temp', 1.0))

    # Dataset
    args.data_path = args.data_path or train_cfg.get('data_root', './data/imagenet')
    args.dataset_type = train_cfg.get('dataset_type', 'imagenet')
    args.dataset_format = train_cfg.get('dataset_format', 'auto')

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

    # Performance
    if not hasattr(args, 'use_compile') or args.use_compile is False:
        args.use_compile = train_cfg.get('use_compile', False)

    # AMP dtype: CLI > YAML > default('fp16')
    cli_amp_dtype = getattr(args, 'amp_dtype', None)
    yaml_amp_dtype = train_cfg.get('amp_dtype', None)
    args.amp_dtype = (cli_amp_dtype or yaml_amp_dtype or 'fp16').lower()
    assert args.amp_dtype in ('fp16', 'bf16', 'fp32'), \
        f"amp_dtype must be 'fp16' | 'bf16' | 'fp32', got {args.amp_dtype!r}"

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

    PSNR de-normalizes inputs back to [0, 1] using the same (mean, std) the
    data pipeline applied, so the resulting dB values are physically meaningful.
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

        if mean is not None and std is not None:
            self._denorm_mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
            self._denorm_std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        else:
            self._denorm_mean = None
            self._denorm_std = None

    def _denormalize_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        if self._denorm_mean is None or self._denorm_std is None:
            return x
        mean = self._denorm_mean.to(dtype=x.dtype, device=x.device)
        std = self._denorm_std.to(dtype=x.dtype, device=x.device)
        return (x * std + mean).clamp_(0.0, 1.0)

    def _get_ssim_kernel(self, channels: int, window_size: int,
                         device: torch.device) -> torch.Tensor:
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
        if self._denorm_mean is not None and self._denorm_std is not None:
            x_unit = self._denormalize_to_unit(x)
            x_recon_unit = self._denormalize_to_unit(x_recon)
            mse = torch.mean((x_unit - x_recon_unit) ** 2)
            if mse == 0:
                return float('inf')
            return (-10.0 * torch.log10(mse)).item()

        mse = torch.mean((x - x_recon) ** 2)
        if mse == 0:
            return float('inf')
        max_val = 1.0 if x.max() <= 1.0 else 255.0
        return (20 * torch.log10(torch.tensor(max_val, device=x.device) / torch.sqrt(mse))).item()

    def compute_ssim(self, x: torch.Tensor, x_recon: torch.Tensor,
                     window_size: int = 11,
                     C1: float = 0.01**2, C2: float = 0.03**2) -> float:
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
        return self.inception(images).cpu().numpy()


# ============================================================================
# Batch helpers
# ============================================================================

def _unpack_batch(batch):
    """Returns (images, cond). Supports both (image, label) tuple and dict batch."""
    if isinstance(batch, dict):
        image = batch["image"]
        cond = {k: v for k, v in batch.items() if k != "image"}
        return image, cond
    return batch[0], batch[1]


def _cond_to_device(cond, device):
    """Move all tensors inside cond to device."""
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

    cond_type = getattr(args, 'cond_type', 'label')
    if cond_type == 'text':
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
        sle_variant = 'EmbedLabel'
        label_encoder = EmbeddingLabelEncoder(
            input_size=args.input_size, patch_size=16,
            num_classes=args.num_classes, latent_dim=args.latent_dim,
            init_logsigma=args.label_init_logsigma,
        )

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
# Accelerate helpers
# ============================================================================

def _sync_metric_logger(metric_logger: misc.MetricLogger, accelerator: Accelerator):
    """All-reduce SmoothedValue counters across processes via accelerate.

    Replaces `metric_logger.synchronize_between_processes()` whose hard-coded
    `device='cuda'` is unsafe under multi-GPU (would default to cuda:0 on every
    rank). accelerator.reduce honours each rank's actual device.
    """
    if accelerator.num_processes <= 1:
        return
    for meter in metric_logger.meters.values():
        t = torch.tensor([meter.count, meter.total],
                         dtype=torch.float64, device=accelerator.device)
        gathered = accelerator.reduce(t, reduction='sum')
        meter.count = int(gathered[0].item())
        meter.total = float(gathered[1].item())


def _gather_numpy(arr: np.ndarray, accelerator: Accelerator) -> np.ndarray:
    """Gather a per-rank numpy array across processes; returns concatenated np."""
    if accelerator.num_processes <= 1:
        return arr
    t = torch.from_numpy(arr).to(accelerator.device)
    gathered = accelerator.gather(t)
    return gathered.cpu().numpy()


def _save_checkpoint(accelerator: Accelerator, args, epoch: int, name: str):
    """Save full training state via accelerator (model + optim + scaler + RNG)
    plus a small `meta.pt` carrying epoch / args for resume bookkeeping."""
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{name}")
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        torch.save(
            {'epoch': epoch, 'args': vars(args)},
            os.path.join(ckpt_dir, "meta.pt"),
        )


# ============================================================================
# Training Engine
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    accelerator: Accelerator,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args,
):
    """Train for one epoch.

    accelerate handles for us:
      - autocast(dtype=...) via `mixed_precision`
      - GradScaler scale/unscale (fp16 only)
      - DDP no_sync within accumulation window (via `accelerator.accumulate`)
      - loss /= gradient_accumulation_steps inside `accelerator.backward`
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'[Epoch {epoch + 1}/{args.epochs}]'
    print_freq = args.log_freq
    accum_iter = max(1, getattr(args, 'accum_iter', 1))

    optimizer.zero_grad()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, cond = _unpack_batch(batch)
        # accelerate moves prepared-loader tensors automatically; cond may be a
        # nested dict (T2I). Be explicit so we never depend on auto-handling.
        cond = _cond_to_device(cond, accelerator.device)

        # Per-iteration LR schedule, fired only at accumulation boundaries.
        if step % accum_iter == 0:
            fractional_epoch = step / len(data_loader) + epoch
            lr_sched.adjust_learning_rate(optimizer, fractional_epoch, args)

        with accelerator.accumulate(model):
            x_recon, loss, metrics = model(images, cond)
            accelerator.backward(loss)

            # Gradient clipping must run only on update steps, after gradients
            # have been all-reduced (sync_gradients=True at the boundary).
            if accelerator.sync_gradients and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        # Logging — loss.item() is the un-scaled per-batch loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            accelerator.print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_recon=metrics.get('loss_recon', 0))
        metric_logger.update(loss_align=metrics.get('loss_alignment', 0))
        metric_logger.update(loss_align_rev=metrics.get('loss_kl_reverse', 0))
        metric_logger.update(sigma_img=metrics.get('mean_sigma_img', 0))
        metric_logger.update(sigma_lbl=metrics.get('mean_sigma_label', 0))
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Per-step tensorboard log (loss reduced across processes for accuracy)
        if step % args.log_freq == 0:
            loss_t = torch.tensor(loss_value, device=accelerator.device)
            loss_value_reduce = accelerator.reduce(loss_t, reduction='mean').item()
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            accelerator.log({
                'train/loss':            loss_value_reduce,
                'train/loss_recon':      metrics.get('loss_recon', 0),
                'train/loss_alignment':  metrics.get('loss_alignment', 0),
                'train/loss_kl_reverse': metrics.get('loss_kl_reverse', 0),
                'train/mean_sigma_label': metrics.get('mean_sigma_label', 0),
                'train/mean_sigma_img':  metrics.get('mean_sigma_img', 0),
                'train/lr':              lr,
            }, step=epoch_1000x)

    _sync_metric_logger(metric_logger, accelerator)
    accelerator.print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    accelerator: Accelerator,
    data_loader: DataLoader,
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

    model_unwrapped = accelerator.unwrap_model(model)

    for batch in metric_logger.log_every(data_loader, 50, header):
        images, cond = _unpack_batch(batch)
        cond = _cond_to_device(cond, accelerator.device)

        with accelerator.autocast():
            x_recon, loss, metrics = model(images, cond)

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

        if (metrics_computer.inception is not None
                and fid_samples_collected < fid_num_samples):
            feats_real_list.append(metrics_computer.extract_inception_features(images))
            feats_recon_list.append(metrics_computer.extract_inception_features(x_recon))
            if compute_fid:
                x_gen = model_unwrapped.generate_from_label(cond)
                feats_gen_list.append(metrics_computer.extract_inception_features(x_gen))
            fid_samples_collected += images.shape[0]

    _sync_metric_logger(metric_logger, accelerator)
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # FID: gather features across processes, compute on main only
    if feats_real_list:
        feats_real = _gather_numpy(np.concatenate(feats_real_list, axis=0), accelerator)
        feats_recon = _gather_numpy(np.concatenate(feats_recon_list, axis=0), accelerator)
        feats_gen = (_gather_numpy(np.concatenate(feats_gen_list, axis=0), accelerator)
                     if feats_gen_list else None)

        if accelerator.is_main_process:
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

    accelerator.print(f"{header}", metric_logger)
    return results


# ============================================================================
# Config Printing
# ============================================================================

def _print_config(args, accelerator: Accelerator):
    """Print hyperparameters in a grouped table (main process only)."""
    if not accelerator.is_main_process:
        return

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

    eff_bs = args.batch_size * args.accum_iter * accelerator.num_processes
    _section("Training", [
        ("epochs",                str(args.epochs)),
        ("batch_size (per GPU)",  str(args.batch_size)),
        ("accum_iter",            str(args.accum_iter)),
        ("effective batch size",  str(eff_bs)),
        ("lr / lr_label_encoder", f"{args.lr:.1e} / {args.lr_label_encoder:.1e}"),
        ("lr_schedule",           str(args.lr_schedule)),
        ("warmup_epochs",         str(args.warmup_epochs)),
        ("weight_decay",          str(args.weight_decay)),
        ("grad_clip",             str(args.grad_clip)),
    ])

    _section("Loss Weights", [
        ("recon",               str(args.weight_recon)),
        ("alignment (fwd)",     str(args.weight_alignment)),
        ("alignment (rev)",     str(args.weight_alignment_reverse)),
        ("alignment_temp",      str(args.alignment_temp)),
        ("label_init_logsigma", str(args.label_init_logsigma)),
    ])

    _section("Data", [
        ("data_path",   str(args.data_path)),
        ("num_workers", str(args.num_workers)),
    ])

    _section("Accelerate / Performance", [
        ("num_processes",    str(accelerator.num_processes)),
        ("distributed_type", str(accelerator.distributed_type)),
        ("device",           str(accelerator.device)),
        ("mixed_precision",  str(accelerator.mixed_precision)),
        ("amp_dtype (cfg)",  str(getattr(args, 'amp_dtype', 'fp16'))),
        ("use_compile",      str(getattr(args, 'use_compile', False))),
    ])

    _section("Checkpoint", [
        ("output_dir",  str(args.output_dir)),
        ("resume",      str(args.resume) if args.resume else "(none)"),
        ("save_freq",   str(args.save_freq)),
    ])

    print(sep)


# ============================================================================
# Main
# ============================================================================

def main(args):
    # ── Build Accelerator ─────────────────────────────────────────────────
    # Map our amp_dtype to accelerate's mixed_precision string:
    #   fp16 → 'fp16' (auto GradScaler)
    #   bf16 → 'bf16' (no GradScaler, fp32-equivalent range — recommended on Ampere+)
    #   fp32 → 'no'   (AMP fully off)
    mixed_precision_map = {'fp16': 'fp16', 'bf16': 'bf16', 'fp32': 'no'}
    mixed_precision = mixed_precision_map[getattr(args, 'amp_dtype', 'fp16')]

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, 'tensorboard'),
        automatic_checkpoint_naming=False,  # we name checkpoints ourselves
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.accum_iter,
        log_with="tensorboard" if args.use_tensorboard else None,
        project_config=project_config,
    )

    # Hardware sanity check for bf16
    if (mixed_precision == 'bf16'
            and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()):
        accelerator.print(
            "[WARN] amp_dtype='bf16' was requested but the current CUDA device "
            "does NOT natively support bfloat16 (e.g. V100/T4). accelerate may "
            "fall back to a software emulation; consider --amp_dtype fp16."
        )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")

    # Init tensorboard tracker (no-op on non-main rank)
    if args.use_tensorboard:
        # `init_trackers` requires a JSON-serialisable config — strip yaml dict
        safe_cfg = {k: v for k, v in vars(args).items()
                    if isinstance(v, (str, int, float, bool, list, tuple))}
        accelerator.init_trackers(project_name="alignment_vae", config=safe_cfg)

    _print_config(args, accelerator)

    # Seed (rank-aware via accelerate)
    set_seed(args.seed, device_specific=True)
    cudnn.benchmark = True

    # ── LR scaling (effective batch accounts for processes × accum) ─────
    eff_batch_size = args.batch_size * args.accum_iter * accelerator.num_processes
    if args.blr is not None:
        args.lr = args.blr * eff_batch_size / 256
    lr_scale_label = args.lr_label_encoder / args.yaml_config.get('training', {}).get('lr', 1e-4)

    accelerator.print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    accelerator.print(f"Actual lr: {args.lr:.2e}")
    accelerator.print(f"Effective batch size: {eff_batch_size}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    # NOTE: keep these stats in sync with `MetricsComputer(mean=..., std=...)`
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
        from dataset import build_text_image_dataset
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
        train_dataset = ImageLabelDataset(
            root=args.data_path, split="train",
            transform=transform_train, num_classes=args.num_classes,
        )
        val_dataset = ImageLabelDataset(
            root=args.data_path, split="val",
            transform=transform_val, num_classes=args.num_classes,
        )

    accelerator.print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)} "
                      f"[type={getattr(args, 'dataset_type', 'imagenet')}]")

    # No manual DistributedSampler: accelerator.prepare wraps DataLoader and
    # injects the proper sampler / batch sharding for us.
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    data_loader_train = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    data_loader_val = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args)

    if args.use_compile:
        accelerator.print("Compiling model with torch.compile (mode='reduce-overhead')...")
        model = torch.compile(model, mode='reduce-overhead')

    # ── Optimizer (separate LR for label encoder; fused AdamW where possible) ──
    try:
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    except Exception:
        fused_available = False
    use_fused = fused_available and torch.cuda.is_available()
    extra_optim_kwargs = {'fused': True} if use_fused else {}

    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.vae.parameters() if p.requires_grad],
         'lr': args.lr},
        {'params': [p for p in model.label_encoder.parameters() if p.requires_grad],
         'lr': args.lr, 'lr_scale': lr_scale_label},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999), **extra_optim_kwargs)
    if use_fused:
        accelerator.print("Using fused AdamW optimizer (CUDA-accelerated)")

    # ── accelerate.prepare wraps everything (DDP / AMP / DataLoader sharding) ──
    model, optimizer, data_loader_train, data_loader_val = accelerator.prepare(
        model, optimizer, data_loader_train, data_loader_val
    )
    accelerator.print(
        f"Accelerator ready: distributed_type={accelerator.distributed_type}, "
        f"num_processes={accelerator.num_processes}, "
        f"mixed_precision={accelerator.mixed_precision}"
    )

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics_computer = MetricsComputer(
        device=str(accelerator.device),
        compute_fid=args.compute_fid,
        mean=imagenet_mean,
        std=imagenet_std,
    )

    # ── Resume ───────────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path) and os.path.basename(ckpt_path).startswith("checkpoint-"):
            resume_dir = ckpt_path
        else:
            resume_dir = os.path.join(ckpt_path, "checkpoint-last")
        if os.path.isdir(resume_dir):
            accelerator.print(f"Resuming from: {resume_dir}")
            accelerator.load_state(resume_dir)
            meta_path = os.path.join(resume_dir, "meta.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location='cpu', weights_only=False)
                args.start_epoch = int(meta.get('epoch', -1)) + 1
                accelerator.print(f"Resumed epoch: start_epoch={args.start_epoch}")
        else:
            accelerator.print(f"[WARN] Checkpoint not found: {resume_dir}. "
                              "Training from scratch.")

    # ── Training loop ────────────────────────────────────────────────────
    accelerator.print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model=model,
            accelerator=accelerator,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )

        val_stats = evaluate(
            model=model,
            accelerator=accelerator,
            data_loader=data_loader_val,
            metrics_computer=metrics_computer,
            compute_fid=args.compute_fid,
            fid_num_samples=args.fid_num_samples,
            header="Val:",
        )

        # Per-epoch tensorboard logging
        if args.use_tensorboard:
            log_payload = {f'epoch_train/{k}': v for k, v in train_stats.items()}
            log_payload.update({f'epoch_val/{k}': v for k, v in val_stats.items()})
            accelerator.log(log_payload, step=epoch)

        # Checkpointing — save_state handles model/optim/scaler/RNG together
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            _save_checkpoint(accelerator, args, epoch, "last")
        if epoch > 0 and epoch % 50 == 0:
            _save_checkpoint(accelerator, args, epoch, str(epoch))

        val_loss = val_stats.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(accelerator, args, epoch, "best")
            accelerator.print(f"New best model saved (val_loss={best_val_loss:.4f})")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    accelerator.print(f"Training completed! Total time: {total_time}")

    if args.use_tensorboard:
        accelerator.end_training()


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
