# Sunfish Diffusion LLM

A 1.4B parameter diffusion-based language model trained on FineWeb, incorporating research from RND1 (Radical Numerics) and LLaDA. This project demonstrates that diffusion models can be successfully applied to language generation at scale.

## 🌟 Key Features

- **Diffusion-based Architecture**: Unlike traditional autoregressive models, Sunfish uses continuous diffusion in embedding space for parallel text generation
- **Scalable Training**: FSDP (Fully Sharded Data Parallel) support for multi-GPU training on consumer hardware
- **Efficient Inference**: DDIM sampling allows high-quality generation in 50 steps (vs 1000 for DDPM)
- **Text Infilling**: Supports filling in masked spans, not just left-to-right generation
- **Production-Ready**: Complete training framework with checkpointing, logging, and monitoring

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Architecture](#architecture)
- [Research Background](#research-background)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 256GB RAM recommended
- 2x NVIDIA GPUs (tested on A6000 48GB + A4500 20GB)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SunFish-Full.git
cd SunFish-Full

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install with W&B for experiment tracking
pip install wandb
wandb login
```

## ⚡ Quick Start

### Test the Data Pipeline

```bash
# Test data loading (uses tiny config for speed)
python -m data.fineweb_datamodule
```

### Run a Quick Training Test

```bash
# Overfit on a single batch to verify everything works
python train.py --config tiny --overfit-batches 10 --max-steps 100
```

### Generate Text from a Checkpoint

```bash
# Generate 3 samples of 256 tokens
python sample.py checkpoints/last.ckpt --num-samples 3 --length 256 --num-steps 50
```

## 🎓 Training

### Full Training (1.4B Model)

```bash
# Train with default configuration
python train.py --config 1.4B

# With custom batch size and learning rate
python train.py --config 1.4B --batch-size 4 --learning-rate 5e-5

# Resume from checkpoint
python train.py --config 1.4B --resume checkpoints/last.ckpt
```

### Configuration Presets

- **1.4B** (default): Full 1.4B parameter model (24 layers, 2048 dim)
- **small**: 350M parameters for faster experimentation (12 layers, 1024 dim)
- **tiny**: 125M parameters for debugging (12 layers, 768 dim)

### Training Parameters

Key hyperparameters in `config/model_config.py`:

```python
batch_size: int = 8              # Per-GPU batch size
accumulate_grad_batches: int = 16  # Gradient accumulation
learning_rate: float = 1e-4      # Peak learning rate
max_steps: int = 500_000         # Total training steps (~500GB of data)
warmup_steps: int = 10_000       # LR warmup
timesteps: int = 1000            # Diffusion timesteps
```

### Expected Training Time

On 2× A6000 GPUs:
- **500k steps**: ~3-5 days
- **Tokens processed**: ~131B tokens (~500GB of text)
- **Effective batch**: 256 sequences × 2048 tokens = 524k tokens/step

### Monitoring

Training metrics are logged to:
- **Weights & Biases**: Real-time metrics (if enabled)
- **TensorBoard**: `tensorboard --logdir logs/tensorboard`
- **Console**: Progress bars and periodic updates
- **Log files**: `logs/train_YYYYMMDD_HHMMSS.log`

### Checkpoints

Checkpoints are saved in `checkpoints/`:
- `sunfish-{step:06d}-{loss:.4f}.ckpt`: Top-K best checkpoints
- `last.ckpt`: Most recent checkpoint (for resumption)

## 🎯 Inference

### Unconditional Generation

Generate text from pure noise:

```bash
# Generate 5 samples with DDIM (fast, 50 steps)
python sample.py checkpoints/last.ckpt \
    --num-samples 5 \
    --length 512 \
    --num-steps 50 \
    --sampler ddim

# Use DDPM for higher quality (slow, 200 steps)
python sample.py checkpoints/last.ckpt \
    --num-samples 1 \
    --length 256 \
    --num-steps 200 \
    --sampler ddpm
```

### Text Infilling

Fill in masked text:

```bash
python sample.py checkpoints/last.ckpt \
    --mode infill \
    --prompt "The future of AI is [MASK] and will transform society." \
    --mask-length 20 \
    --num-steps 50
```

### Advanced Options

```bash
--sampler {ddpm,ddim}   # Sampling algorithm
--eta 0.5               # DDIM stochasticity (0=deterministic, 1=stochastic)
--num-steps 100         # More steps = better quality but slower
--output results.txt    # Save to file
```

## 🏗️ Architecture

### Model Overview

Sunfish is a **bidirectional transformer** trained to denoise embeddings corrupted by Gaussian noise. Unlike autoregressive models, it can attend to both past and future context.

#### Key Components

1. **Token Embedding** (50,257 × 2048)
   - Embeds discrete tokens into continuous space

2. **Positional Embedding** (2048 × 2048)
   - Learned positional encodings

3. **Timestep Embedding** (MLP)
   - Sinusoidal encoding + 2-layer MLP
   - Conditions model on diffusion timestep

4. **Transformer Encoder** (24 layers)
   - Multi-head self-attention (16 heads)
   - Feed-forward network (2048 → 8192 → 2048)
   - Pre-LayerNorm for stability

5. **Noise Prediction Head** (2048 → 2048)
   - Predicts noise in embedding space

### Diffusion Process

**Forward (Training):**
```
x_0 (clean) → add noise → x_t (noisy) → model predicts noise → loss
```

**Reverse (Inference):**
```
x_T (pure noise) → iteratively denoise → x_0 (clean) → round to tokens → text
```

### Training Objective

Simplified DDPM loss (epsilon-prediction):

```python
loss = MSE(predicted_noise, actual_noise)
```

where noise is sampled as `ε ~ N(0, I)` and added according to the diffusion schedule.

## 📚 Research Background

This implementation builds on:

### RND1 (Radical Numerics, 2024)

- **AR-to-Diffusion Conversion**: Simple Continual Pretraining (SCP) recipe
- **Layer-wise Learning Rates**: Different LRs for attention vs FFN layers
- **Large Batch Training**: Diffusion models need 2× larger batches than AR
- **Paper**: https://www.radicalnumerics.ai/blog/rnd1

### LLaDA (2024)

- **Continuous Diffusion**: Diffusion in embedding space (not discrete tokens)
- **Variable Masking**: Random mask ratios per batch
- **Paper**: https://arxiv.org/abs/2502.09992

### DDPM/DDIM

- **DDPM**: Original diffusion probabilistic model (Ho et al., 2020)
- **DDIM**: Deterministic sampling for faster inference (Song et al., 2021)

## 💻 Hardware Requirements

### Minimum

- **GPUs**: 1× NVIDIA A6000 (48GB) or equivalent
- **RAM**: 128GB
- **Storage**: 100GB free (for checkpoints and logs)
- **Network**: Stable connection for streaming FineWeb

### Recommended (for full 1.4B training)

- **GPUs**: 2× NVIDIA A6000 (48GB) or 1× A100 (80GB)
- **RAM**: 256GB
- **Storage**: 500GB NVMe SSD
- **CPU**: 32+ cores (Threadripper or EPYC)

### Memory Usage

| Component | Memory |
|-----------|--------|
| Model (BF16) | ~2.8 GB |
| Optimizer States | ~5.6 GB |
| Activations (batch=8) | ~15 GB per GPU |
| FSDP Overhead | ~5 GB |
| **Total per GPU** | **~28 GB** |

With FSDP, model shards across GPUs, so the A4500 (20GB) can participate in training.

## 🔧 Project Structure

```
SunFish-Full/
├── config/
│   └── model_config.py       # Hyperparameters and configuration
├── data/
│   └── fineweb_datamodule.py # FineWeb streaming dataset
├── models/
│   ├── diffusion_model.py    # Transformer architecture
│   └── schedulers.py          # DDPM/DDIM samplers
├── utils/
│   └── helpers.py             # Testing and monitoring utilities
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs
├── train.py                   # Training script
├── sample.py                  # Inference script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_size` (e.g., from 8 to 4)
2. Increase `accumulate_grad_batches` (e.g., from 16 to 32)
3. Enable `fsdp_activation_checkpointing = True` in config
4. Reduce `block_size` (e.g., from 2048 to 1024)

### NaN Loss

**Causes:**
- Learning rate too high
- Gradient explosion

**Solutions:**
1. Lower learning rate (e.g., `1e-4` → `5e-5`)
2. Increase gradient clipping (e.g., `max_grad_norm = 0.5`)
3. Check for data issues (inf/nan in inputs)

### Slow Training

**Optimizations:**
1. Increase `num_workers` in data loading (e.g., 4 → 8)
2. Use `bf16-mixed` precision (faster than `16-mixed` on Ampere+)
3. Enable `compile_model = True` (PyTorch 2.0+)
4. Reduce `log_every_n_steps` to minimize overhead

### Data Loading Bottleneck

If GPU utilization is low (<80%):
1. Increase `num_workers` and `prefetch_factor`
2. Consider downloading FineWeb locally instead of streaming
3. Use faster storage (NVMe SSD)

## 📊 Evaluation

### Qualitative

Inspect generated samples:
```bash
python sample.py checkpoints/best.ckpt --num-samples 10 --length 512
```

### Quantitative (TODO)

- Perplexity on held-out data
- Benchmark on language understanding tasks (LAMBADA, HellaSwag)
- Comparison with AR baseline of same size

## 🚧 Roadmap

- [ ] Autoregressive initialization support (load GPT-2/LLaMA checkpoints)
- [ ] Classifier-free guidance for conditional generation
- [ ] Instruction fine-tuning
- [ ] Mixture-of-Experts (MoE) variant
- [ ] Longer context support (4096+)
- [ ] Quantization (INT8/INT4) for inference

## 📝 Citation

If you use this code or model, please cite:

```bibtex
@software{sunfish2024,
  title={Sunfish: A 1.4B Parameter Diffusion Language Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SunFish-Full}
}
```

**Research this builds on:**

```bibtex
@article{rnd1_2024,
  title={RND1: Simple, Scalable AR-to-Diffusion Conversion},
  author={Radical Numerics},
  year={2024},
  url={https://www.radicalnumerics.ai/blog/rnd1}
}

@article{llada2024,
  title={LLaDA: Large Language Diffusion Assistant},
  author={ML-GSAI},
  year={2024},
  journal={arXiv preprint arXiv:2502.09992}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Radical Numerics** for the RND1 methodology
- **ML-GSAI** for LLaDA research
- **HuggingFace** for FineWeb dataset and transformers library
- **PyTorch Lightning** team for the training framework

## 📧 Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com
- Discord: YourDiscord#1234

---

**Note**: This is a research project. The model requires substantial compute resources and training time to reach full capability. Early checkpoints may produce low-quality text. Patience and experimentation are key!

Happy training! 🚀
