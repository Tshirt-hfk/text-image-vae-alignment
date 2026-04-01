# Text-Image VAE Latent Alignment

**Core Idea**: Encode images as points in a high-dimensional latent space, encode text as a Gaussian distribution. During training, maximize the probability density of the image latent point under the text-encoded Gaussian distribution.

```
Text "cat" → Gaussian N(μ_text, σ²I)  ← Text encodes a semantic region
Image latent z_img  ←  point in that semantic region
Maximize p(z_img | text) + MSE reconstruction
```

## Architecture

```
Text  → Text Encoder (CLIP/T5) → (μ, logσ) → Gaussian Distribution
Image → VAE Encoder → z_img → VAE Decoder → Reconstructed Image

Loss = L_recon + β·L_KL + L_alignment
     = MSE(x, x̂) + β·KL(q(z|x)||N(0,I)) - log N(z_img; μ_text, σ²_text·I)
```

## Key Innovation

Standard VAE: `p(z)` is fixed (N(0,I))
This work: `p(z|text)` is a learnable Gaussian conditioned on text

→ The image latent space becomes a **semantic subspace** of the text latent space

## Setup

```bash
pip install torch torchvision clip-by-openai open_clip_torch pyyaml tqdm
```

## Train

```bash
python train.py --config configs/default.yaml
```

## Project Structure

```
.
├── model/
│   ├── vae.py           # Image VAE (encoder + decoder with skip connections)
│   ├── text_encoder.py # Text encoder outputting Gaussian (μ, logσ)
│   └── alignment_vae.py# Combined model
├── losses/
│   └── alignment_loss.py  # Gaussian probability alignment loss
├── configs/
│   └── default.yaml
└── train.py
```

## Reference

Based on discussion with 方磊 (2026-04-01)
