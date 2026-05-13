# Text-Image VAE Spatial Alignment

将图像和类别标签编码到**同一空间高斯潜空间** `[B, H/16, W/16, D]`，通过 KL 散度对齐两者的分布，从而支持：

- **从标签条件生成图像**：标签 → 高斯参数 → 采样 → 解码
- **图像重建**：图像 → VAE 潜变量 → 解码
- **标签间插值生成**：在两个标签的高斯分布间线性插值后解码

## 架构总览

```
                                            ┌───────────────────────┐
Image (RGB)  ──► VAE Encoder (4× ↓2 conv)  ──►  N(μ_img,   σ_img²)  │
                                            │   [B, h, w, D]        │
                                            ├───── KL 对齐 ─────────┤
Label (int)  ──► Spatial Label Encoder ────►  N(μ_label, σ_label²)  │
                                            └─────────┬─────────────┘
                                                      │
                                z = μ_img + σ_img · ε │ (重参数化)
                                                      ▼
                                            VAE Decoder ──► Recon Image

Loss = w_recon · MSE(x, x̂)
     + w_align · KL(q_img || q_label)        # 前向：图像分布逼近标签分布
     + w_align_reverse · KL(q_label || q_img) # 反向：稳定 μ 对齐、防止 σ 过冲
```

### 核心组件

- [`VAE`](model/vae.py:246-287)：Flux2 风格的 4 级 Conv2d 下采样 / 上采样（共 16× 压缩），ResBlock + 可选 [`AttnBlock`](model/vae.py:44-70)（基于 PyTorch SDPA / FlashAttention），中间层 `Res-Attn-Res` 瓶颈。Encoder 输出 `[B, h, w, D]` 的对角高斯 `(μ, logσ)` 并重参数化采样 `z = μ + σ·ε`。
- [`SpatialLabelEncoder`](model/text_encoder.py:112-228)：基于 [`AdaLNBlock`](model/text_encoder.py:87-105)（adaLN 调制 + 2D-RoPE + SwiGLU FFN + RMSNorm + QK-norm）的 Transformer。可学习空间 token 叠加固定 2D sin-cos 位置编码后，从第 `in_context_start` 层起注入若干 in-context label token，最终输出空间高斯分布。
- [`EmbeddingLabelEncoder`](model/text_encoder.py:235-290)：**消融基线**，跳过 Transformer，直接用 `nn.Embedding` 把 label 映射到每个空间位置的 `(μ, logσ)`，用于验证「对齐 KL」是否真的需要表达力强的标签编码器。
- [`AlignmentVAE`](model/alignment_vae.py:13-187)：组合模型，封装训练 forward 与多种推理接口。
- [`GaussianAlignmentLoss`](losses/alignment_loss.py:15-78)：可单独使用的 KL 对齐损失模块（支持双向、温度缩放）。

### 预设变体

**VAE**（[`VAE_models`](model/vae.py:309)）

| 变体 | 基础通道 `ch` | `ch_mult` | ResBlock 数 | 注意力分辨率 |
|------|:---:|:---:|:---:|:---:|
| VAE-B | 128 | [1,2,4,4] | 2 | {16} |
| VAE-L | 192 | [1,2,4,4] | 2 | {32, 16} |
| VAE-H | 256 | [1,2,4,4] | 3 | {32, 16} |

**Spatial Label Encoder**（[`SpatialLabelEncoder_models`](model/text_encoder.py:329-334)）

| 变体 | 深度 | 隐层维度 | 注意力头数 | in-context 注入起始层 |
|------|:---:|:---:|:---:|:---:|
| SLE-B | 6  | 768  | 12 | 4  |
| SLE-L | 12 | 1024 | 16 | 8  |
| SLE-H | 16 | 1280 | 16 | 10 |
| SLE-E | —  | —    | —  | — *（embedding-only baseline）* |

> 注：当 `use_embedding_label_encoder=true` 时，会强制使用 `EmbeddingLabelEncoder`，该开关优先级高于 `label_encoder_variant`。

## 损失函数

实现位于 [`AlignmentVAE.forward`](model/alignment_vae.py:81-133)，对角高斯 KL 闭式解：

```
KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```

逐元素计算后对所有维度（batch + 空间 + 通道）取均值。

| 损失分量 | 公式 | 默认权重 | 作用 |
|---------|------|:---:|------|
| 重建损失 | `MSE(x_recon, x)` | 1.0 | 保证图像重建质量 |
| 前向对齐 KL | `KL(q_img \|\| q_label)` | 2.0 | 拉近图像分布到标签分布（保持多样性） |
| 反向对齐 KL | `KL(q_label \|\| q_img)` | 0.5 | 稳定 μ 对齐，防止标签 σ 过冲 |

> 模型层 `forward` 不包含独立的 VAE-KL 项与标签熵正则；`logσ` 通过 `clamp(-5, 2)` 约束数值稳定性。如需经典 VAE-KL，可使用 [`VAE.forward`](model/vae.py:273-280) 单独获取。

## 安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`torch>=2.0`、`torchvision>=0.15`、`tensorboard`、`scipy`、`pyyaml`、`pillow`、`tqdm`。

### 数据准备

预期 ImageNet 风格目录结构：

```
data_root/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

通过 [`ImageLabelDataset`](train.py:314-346) 加载；若指定路径不存在，会自动回退到随机张量的 dummy 数据集（便于快速 smoke test）。训练数据采用 `Resize(1.1×) + RandomCrop + RandomHFlip + ImageNet Normalize`。

### 单机训练

```bash
python train.py --config configs/default.yaml
```

### 单机多卡（DDP）

```bash
torchrun --nproc_per_node=8 train.py --config configs/default.yaml
```

### 断点续训

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/default.yaml \
    --resume ./output_dir
```

`--resume` 可以是目录（自动加载 `checkpoint-last.pth`）或具体 `.pth` 文件，恢复 model / optimizer / AMP scaler / epoch。

### 主要训练参数

完整参数见 [`configs/default.yaml`](configs/default.yaml) 与 `python train.py --help`。常用参数：

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--input_size` | 256 | 输入图像分辨率 |
| `--latent_dim` | 64 | 潜空间通道数 D |
| `--vae_variant` | `VAE-H` | `VAE-B` / `VAE-L` / `VAE-H` 或自定义 |
| `--label_encoder_variant` | `SLE-B` | `SLE-B` / `SLE-L` / `SLE-H` / `SLE-E` |
| `--num_classes` | 1000 | 类别数（默认 ImageNet） |
| `--batch_size` | 64 | 每 GPU 批大小 |
| `--accum_iter` | 2 | 梯度累积步数（有效 batch = `batch × accum × world`） |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率（绝对值） |
| `--blr` | — | 基准 LR：`lr = blr × eff_batch / 256`（与 `--lr` 二选一） |
| `--lr_schedule` | `cosine` | `cosine` / `constant` |
| `--warmup_epochs` | 5 | 学习率线性预热轮数 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值（≤0 关闭） |
| `--use_amp` / `--no_amp` | 启用 | AMP 混合精度训练（fp16） |
| `--use_compile` | 关闭 | `torch.compile(mode='reduce-overhead')` 内核融合 |
| `--output_dir` | `./output_dir` | 输出目录（含 checkpoint 与 tensorboard 日志） |

性能相关默认开启项：fused AdamW、`gradient_as_bucket_view=True`、`persistent_workers`、`prefetch_factor=4`、DDP `no_sync` 上下文（梯度累积步内跳过 all-reduce）。

### 推理示例

```python
import torch
from model import AlignmentVAE
from model.vae import VAE_H
from model.text_encoder import SpatialLabelEncoder_B

# 1) 构建模型（参数需与训练时一致）
vae = VAE_H(latent_dim=64, resolution=256)
label_encoder = SpatialLabelEncoder_B(input_size=256, num_classes=1000, latent_dim=64)
model = AlignmentVAE(
    input_size=256, latent_dim=64, num_classes=1000,
    vae=vae, label_encoder=label_encoder,
).cuda().eval()

# 2) 加载 checkpoint
ckpt = torch.load("output_dir/checkpoint-best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

# 3) 标签 → 图像生成
labels = torch.tensor([5, 10, 42], device="cuda")
images = model.generate_from_label(labels)         # [3, 3, 256, 256]

# 4) 图像重建（确定性：使用 μ 而非采样）
x = torch.randn(2, 3, 256, 256, device="cuda")
recon = model.reconstruct_image(x)

# 5) 两个标签间的潜空间插值
frames = model.label_interpolate(label1=5, label2=42, num_steps=8, device="cuda")

# 6) 拿到中间高斯分布（用于诊断 / 下游任务）
mu_img,  sigma_img  = model.encode_image_to_gaussian(x)
mu_lbl,  sigma_lbl  = model.encode_label_to_gaussian(labels)
```

## 文件结构

```
.
├── model/
│   ├── vae.py               # VAE Encoder/Decoder（Flux2 风格）+ VAE_B/L/H
│   ├── text_encoder.py      # SpatialLabelEncoder + AdaLNBlock + EmbeddingLabelEncoder
│   └── alignment_vae.py     # AlignmentVAE 组合模型（训练 forward + 推理接口）
├── losses/
│   └── alignment_loss.py    # GaussianAlignmentLoss（可独立使用）
├── util/
│   ├── misc.py              # 分布式工具（DDP 初始化、指标同步、checkpoint）
│   ├── lr_sched.py          # Cosine / Constant 学习率 + 线性预热
│   └── model_util.py        # RMSNorm、2D sin-cos PE、2D Vision RoPE
├── configs/
│   └── default.yaml         # 默认训练配置
├── train.py                 # 训练 / 评估入口（DDP + AMP + FID）
├── requirements.txt
└── README.md
```

## 评估指标

[`evaluate`](train.py:549-634) 在每个 epoch 后于验证集计算：

- **重建质量**：`MSE` / `PSNR` / `SSIM`（[`MetricsComputer`](train.py:251-307)）
- **生成质量（可选）**：基于 InceptionV3 pool3 (2048-d) 特征
  - **Recon FID**：原图 vs 重建图像
  - **Gen FID**：原图 vs `label → image` 生成图像
- **潜空间诊断**：`mean_mu_img_norm`、`mean_sigma_img`、`mean_sigma_label`，以及前向 / 反向 KL

启用 FID 需要在配置中设置 `compute_fid: true` 与 `fid_num_samples`，DDP 下会自动跨 rank `all_gather` 特征。所有指标会写入 TensorBoard（`output_dir/tensorboard/`）。

## Checkpoint 策略

[`main`](train.py:730-941) 自动保存三类 checkpoint 到 `--output_dir`：

- `checkpoint-last.pth`：每 `save_freq` 个 epoch 与最后一个 epoch 覆写
- `checkpoint-{epoch}.pth`：每 50 个 epoch 归档
- `checkpoint-best.pth`：验证 loss 创新低时保存

每个 checkpoint 包含 `model` / `optimizer` / `epoch` / `scaler`（启用 AMP 时） / `args`。
