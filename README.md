# Text-Image VAE Spatial Alignment

将图像和类别标签编码到同一空间高斯潜空间 `[B, H//16, W//16, D]`，通过 KL 散度对齐两者的分布，实现从标签条件生成图像、图像重建以及标签间插值生成。

## 架构

```
Label (int)  → SpatialLabelEncoder (AdaLN Transformer + RoPE)  → (μ_label, logσ_label)  [B, h, w, D]
Image (RGB)  → VAE Encoder (4-stage Conv2d ↓2×)                → (μ_img,   logσ_img)    [B, h, w, D]
                                                                → VAE Decoder            → Recon Image

Loss = L_recon (MSE) + β · L_kl_vae + γ · KL(N_img || N_label) + δ · L_label_entropy
```

### 核心组件

- **SpatialLabelEncoder**：基于 AdaLN Block（adaLN 调制 + RoPE + SwiGLU FFN + RMSNorm），支持 in-context label injection。将离散标签嵌入为可学习空间令牌，叠加 2D sin-cos 位置编码后通过多层 AdaLN Transformer 输出空间高斯分布。
- **VAE**：Flux2 风格的 4 级卷积下采样 / 上采样，每级包含 ResBlock + 可选 AttnBlock，中间层含自注意力。支持重参数化采样 `z = μ + σ·ε`。
- **空间高斯对齐**：在每个空间位置 (h, w) 独立计算 KL 散度，允许不同位置有不同的分布特征。

### 预设变体

**标签编码器**

| 变体 | 深度 | 隐层维度 | 注意力头数 |
|------|------|---------|-----------|
| SLE-B | 12 | 768 | 12 |
| SLE-L | 24 | 1024 | 16 |
| SLE-H | 32 | 1280 | 16 |

**VAE**

| 变体 | 基础通道数 | ResBlock 数 | 注意力分辨率 |
|------|-----------|------------|-------------|
| VAE-B | 128 | 2 | 16 |
| VAE-L | 192 | 2 | 32, 16 |
| VAE-H | 256 | 3 | 32, 16 |

## 损失函数

```
L_total = w_recon · L_recon + w_kl · L_kl_vae + w_align · L_kl_align + w_ent · L_label_entropy
```

| 损失分量 | 公式 | 默认权重 | 作用 |
|---------|------|---------|------|
| 重建损失 | MSE(x_recon, x) | 1.0 | 保证图像重建质量 |
| VAE KL 散度 | KL(q_img \|\| N(0,1)) | 1e-4 | 正则化图像潜在分布 |
| 对齐 KL 散度 | KL(N_img \|\| N_label) | 1.2e-4 | 对齐图像和标签分布 |
| 标签熵正则化 | Σ logσ_label | 2.5e-6 | 防止标签方差坍缩 |

## 使用

### 安装依赖

```bash
pip install -r requirements.txt
```

### 单机训练

```bash
python train.py --config configs/default.yaml
```

### 分布式训练

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/default.yaml \
    --batch_size 32 \
    --lr 1e-4
```

### 主要训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_size` | 256 | 输入图像分辨率 |
| `--latent_dim` | 16 | 潜在空间维度 |
| `--vae_variant` | VAE-B | VAE 变体 |
| `--label_encoder_variant` | SLE-B | 标签编码器变体 |
| `--num_classes` | 1000 | 类别数（ImageNet） |
| `--batch_size` | 32 | 每 GPU 批大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | VAE 学习率 |
| `--lr_label_encoder` | 5e-5 | 标签编码器学习率 |
| `--warmup_epochs` | 5 | 学习率预热轮数 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |

更多参数见 `configs/default.yaml` 或 `python train.py --help`。

### 推理示例

```python
from model import AlignmentVAE

model = AlignmentVAE.load_from_checkpoint("checkpoint.pth")
model.eval()

# 从标签生成图像
images = model.generate_from_label(labels=torch.tensor([5, 10, 42]))

# 图像重建
recon = model.reconstruct_image(images)

# 标签间插值生成
interp = model.label_interpolate(label_a=5, label_b=42, steps=8)
```

## 文件结构

```
├── model/
│   ├── text_encoder.py      # SpatialLabelEncoder + AdaLNBlock + RoPE Attention
│   ├── vae.py               # VAE Encoder / Decoder (Flux2 风格)
│   └── alignment_vae.py     # AlignmentVAE（组合模型 + 推理接口）
├── losses/
│   └── alignment_loss.py    # GaussianAlignmentLoss（空间 KL 散度）
├── util/
│   ├── misc.py              # 分布式工具（DDP 初始化、指标同步、检查点）
│   ├── lr_sched.py          # 学习率调度（Cosine 退火 + 线性预热）
│   └── model_util.py        # RMSNorm、2D sin-cos 位置编码、2D RoPE
├── configs/
│   └── default.yaml         # 默认训练配置
├── train.py                 # 训练入口（含评估、FID 计算）
└── requirements.txt         # 依赖列表
```

## 评估指标

训练过程中自动计算以下指标：

- **MSE** / **PSNR** / **SSIM**：图像重建质量
- **FID**（Fréchet Inception Distance）：基于 Inception V3 特征的生成质量
- 潜在空间诊断：μ 范数、σ 均值等
