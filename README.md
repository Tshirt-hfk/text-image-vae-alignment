# Text-Image VAE Spatial Alignment

将图像和**条件输入（类别标签 / 自然语言 caption）**编码到**同一空间高斯潜空间** `[B, H/16, W/16, D]`，通过 KL 散度对齐两者的分布，从而支持：

- **条件生成**：标签 / caption → 高斯参数 → 采样 → 解码
- **图像重建**：图像 → VAE 潜变量 → 解码
- **条件间插值生成**：在两个条件的高斯分布间线性插值后解码

> 项目支持两套训练流程，通过配置开关无缝切换：
> - **Label 模式（默认）**：ImageNet 1000 类条件 → 图像
> - **Text 模式（T2I）**：COCO Captions / CC3M 等图文对 → 真正的文生图

## 架构总览

```
                                            ┌───────────────────────┐
Image (RGB)  ──► VAE Encoder (4× ↓2 conv)  ──►  N(μ_img,   σ_img²)  │
                                            │   [B, h, w, D]        │
                                            ├───── KL 对齐 ─────────┤
Label (int)  ──► Embedding Label Encoder ──►  N(μ_label, σ_label²)  │
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
- [`EmbeddingLabelEncoder`](model/label_encoder.py)：**Label 模式默认编码器**。直接用两张 `nn.Embedding` 表把 int label 映射到每个空间位置的 `(μ, logσ)`，无 Transformer，参数量小、训练快。
- [`SpatialTextEncoder`](model/text_encoder.py) + [`TextEncoderWrapper`](model/text_encoder.py)：**T2I 模式**。基于 [`AdaLNBlock`](model/text_encoder.py)（adaLN 调制 + 2D-RoPE + SwiGLU FFN + RMSNorm + QK-norm）的 Transformer 主干，配合 HuggingFace CLIP / T5 文本编码器，把 caption 编码为同形状的空间高斯分布。
- [`AlignmentVAE`](model/alignment_vae.py)：组合模型，封装训练 forward 与多种推理接口。
- [`GaussianAlignmentLoss`](losses/alignment_loss.py:15-78)：可单独使用的 KL 对齐损失模块（支持双向、温度缩放）。

### 预设变体

**VAE**（[`VAE_models`](model/vae.py:309)）

| 变体 | 基础通道 `ch` | `ch_mult` | ResBlock 数 | 注意力分辨率 |
|------|:---:|:---:|:---:|:---:|
| VAE-B | 128 | [1,2,4,4] | 2 | {16} |
| VAE-L | 192 | [1,2,4,4] | 2 | {32, 16} |
| VAE-H | 256 | [1,2,4,4] | 3 | {32, 16} |

**条件编码器**（通过 `cond_type` 选择）

| `cond_type` | 编码器 | 适用场景 |
|------|------|------|
| `label`（默认） | [`EmbeddingLabelEncoder`](model/label_encoder.py) | 离散类条件（ImageNet 等） |
| `text` | [`TextEncoderWrapper`](model/text_encoder.py) + [`SpatialTextEncoder`](model/text_encoder.py) | 自然语言 caption（COCO / CC3M 等） |

**Spatial Text Encoder（T2I 模式）**（[`SpatialTextEncoder_models`](model/text_encoder.py)）

| 变体 | 深度 | 隐层维度 | 注意力头数 | in-context 注入起始层 |
|------|:---:|:---:|:---:|:---:|
| STE-B | 6  | 768  | 12 | 4  |
| STE-L | 12 | 1024 | 16 | 8  |
| STE-H | 16 | 1280 | 16 | 10 |

T2I 模式下，由 [`TextEncoderWrapper`](model/text_encoder.py) 加载 HuggingFace 预训练文本编码器（CLIP / T5），输出 `(seq_features, pooled, mask)`：
- `pooled` → AdaLN conditioning `c`
- `seq_features` → in-context tokens（替代 broadcasted label）

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
T2I 模式额外需要 `transformers>=4.30`（用于加载 HF 文本编码器）。

### 数据准备

#### 1) Label 模式（ImageNet）

```
data_root/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

通过 [`ImageLabelDataset`](dataset/label_image_dataset.py) 加载；若指定路径不存在，会自动回退到随机张量的 dummy 数据集（便于快速 smoke test）。

#### 2) Text 模式（T2I）

由 [`TextImageDataset`](dataset/text_image_dataset.py) 自动探测以下三种格式：

**COCO Captions（推荐）**
```
data_root/
├── train2017/                                  # 图像
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

**通用 CSV / JSONL（适合 CC3M / CC12M / 自定义）**
- `train.csv` / `val.csv`：每行 `image_path,caption`（首行可为表头）
- `train.jsonl` / `val.jsonl`：每行 `{"image": "...", "caption": "..."}`

数据路径不存在时同样回退到 dummy 模式。

### 单机训练

```bash
# Label 模式（ImageNet）
python train.py --config configs/imagenet_l2i.yaml

# Text 模式（COCO T2I），先在 configs/coco_t2i.yaml 中改 data_root
python train.py --config configs/coco_t2i.yaml
```

### 单机多卡（DDP）

```bash
torchrun --nproc_per_node=8 train.py --config configs/imagenet_l2i.yaml
torchrun --nproc_per_node=8 train.py --config configs/coco_t2i.yaml
```

### 断点续训

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/imagenet_l2i.yaml \
    --resume ./output_dir
```

`--resume` 可以是目录（自动加载 `checkpoint-last.pth`）或具体 `.pth` 文件，恢复 model / optimizer / AMP scaler / epoch。

### 主要训练参数

完整参数见 [`configs/imagenet_l2i.yaml`](configs/imagenet_l2i.yaml) 与 `python train.py --help`。常用参数：

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--input_size` | 256 | 输入图像分辨率 |
| `--latent_dim` | 64 | 潜空间通道数 D |
| `--vae_variant` | `VAE-H` | `VAE-B` / `VAE-L` / `VAE-H` 或自定义 |
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

#### Label 模式（ImageNet）

```python
import torch
from model import AlignmentVAE
from model.vae import VAE_H
from model.label_encoder import EmbeddingLabelEncoder

vae = VAE_H(latent_dim=64, resolution=256)
label_encoder = EmbeddingLabelEncoder(
    input_size=256, num_classes=1000, latent_dim=64,
)
model = AlignmentVAE(
    input_size=256, latent_dim=64, num_classes=1000,
    vae=vae, label_encoder=label_encoder,
).cuda().eval()

ckpt = torch.load("output_dir/checkpoint-best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

labels = torch.tensor([5, 10, 42], device="cuda")
images = model.generate_from_label(labels)         # [3, 3, 256, 256]
```

#### Text 模式（T2I：caption → image）

```python
import torch
from model import AlignmentVAE
from model.vae import VAE_H
from model.text_encoder import SpatialTextEncoder_B, TextEncoderWrapper
from transformers import AutoTokenizer

# 1) 构建文本编码器 + 标签编码器
text_encoder = TextEncoderWrapper(model_name="clip", freeze=True)
label_encoder = SpatialTextEncoder_B(
    input_size=256, latent_dim=64,
    text_dim=text_encoder.hidden_dim, max_text_len=77,
    text_encoder=text_encoder,
)
vae = VAE_H(latent_dim=64, resolution=256)
model = AlignmentVAE(
    input_size=256, latent_dim=64, num_classes=1,   # T2I 不使用 num_classes
    vae=vae, label_encoder=label_encoder,
).cuda().eval()

ckpt = torch.load("output_dir/checkpoint-best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

# 2) tokenize caption（与训练时同源）
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
captions = [
    "a photo of a cat sitting on a sofa",
    "a beautiful sunset over the ocean",
]
enc = tokenizer(captions, max_length=77, padding="max_length",
                truncation=True, return_tensors="pt").to("cuda")
cond = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

# 3) caption → image
images = model.generate_from_label(cond)           # [2, 3, 256, 256]
```

#### 公共能力（两种模式通用）

```python
# 图像重建（确定性）
x = torch.randn(2, 3, 256, 256, device="cuda")
recon = model.reconstruct_image(x)

# 拿到中间高斯分布
mu_img, sigma_img = model.encode_image_to_gaussian(x)
mu_cond, sigma_cond = model.encode_label_to_gaussian(cond_or_labels)

# Label 模式专属：标签间插值
# frames = model.label_interpolate(label1=5, label2=42, num_steps=8, device="cuda")
```

## 文件结构

```
.
├── model/
│   ├── vae.py               # VAE Encoder/Decoder（Flux2 风格）+ VAE_B/L/H
│   ├── label_encoder.py     # EmbeddingLabelEncoder（label 模式默认；nn.Embedding lookup）
│   ├── text_encoder.py      # 文本条件编码器（text 模式 / T2I）：
│   │                          - SpatialTextEncoder + TextEncoderWrapper
│   │                          - AdaLNBlock / Attention / SwiGLUFFN（Transformer 组件）
│   └── alignment_vae.py     # AlignmentVAE 组合模型（forward 接受 label / text dict）
├── dataset/
│   ├── label_image_dataset.py  # L2I 数据集（ImageNet ImageFolder / dummy）
│   └── text_image_dataset.py   # T2I 数据集（COCO / CSV / JSONL / dummy）
├── losses/
│   └── alignment_loss.py    # GaussianAlignmentLoss
├── util/
│   ├── misc.py              # 分布式工具（DDP 初始化、指标同步、checkpoint）
│   ├── lr_sched.py          # Cosine / Constant 学习率 + 线性预热
│   └── model_util.py        # RMSNorm、2D sin-cos PE、2D Vision RoPE
├── configs/
│   ├── imagenet_l2i.yaml    # ImageNet 类别条件 L2I 配置（默认）
│   └── coco_t2i.yaml        # COCO 文生图配置
├── train.py                 # 训练入口（DDP + AMP + FID，自动适配 label / text）
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
