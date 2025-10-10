# 🐟 SunFish: Diffusion Language Model

SunFish is a diffusion-based language model that generates text by denoising in continuous embedding space. Unlike autoregressive models (GPT, LLaMA), SunFish uses iterative refinement similar to DALL-E and Stable Diffusion.

## 🌟 Key Features

- **Continuous Diffusion**: Operates in embedding space, not discrete tokens
- **Parallel Generation**: Can edit/infill arbitrary text positions
- **Flexible Sampling**: DDPM (1000 steps) or DDIM (50 steps)
- **Scalable Architecture**: From 500K (tiny) to 1.4B+ parameters
- **Modern Training**: FSDP, mixed precision, streaming data

## 📊 Model Sizes

| Config | Parameters | Layers | Heads | Dim | Use Case |
|--------|-----------|--------|-------|-----|----------|
| Tiny   | ~500K     | 2      | 2     | 128 | CPU testing |
| Full   | ~1.4B     | 24     | 16    | 2048| GPU training |

## 🚀 Quick Start

### 1. CPU Development (Current Machine)

Test the entire pipeline on CPU:

```bash
# Install dependencies
pip install -r requirements.txt

# Run validation suite
python validate_cpu.py
```

This will:
- ✅ Initialize tiny model
- ✅ Test data loading
- ✅ Verify forward/backward passes
- ✅ Overfit single batch (sanity check)
- ✅ Test sampling pipeline
- ✅ Run mini training loop

### 2. GPU Training (Main Machine)

Once CPU validation passes, train the full model:

```bash
# Full-scale training
python train.py --config full --name my-experiment

# Tiny config for quick GPU test
python train.py --config tiny --name gpu-test
```

### 3. Generate Text

```bash
# Generate samples
python sample.py checkpoints/last.ckpt --num_samples 5 --seq_len 512

# Infill text (fill in [MASK])
python sample.py checkpoints/last.ckpt --mode infill \
    --text "The future of AI is [MASK] and transformative."
```

## 📁 Project Structure

```
SunFish-Pretain/
├── config/                 # Model configurations
│   ├── model_config.py     # Full-scale config (1.4B params)
│   ├── tiny_config.py      # Tiny config for CPU testing
│   └── __init__.py
├── models/                 # Model implementations
│   ├── diffusion_transformer.py  # Core SunFish model
│   ├── schedulers.py       # DDPM, DDIM, Constrained DDIM
│   └── __init__.py
├── data/                   # Data pipeline
│   ├── fineweb_datamodule.py
│   └── __init__.py
├── utils/                  # Helper functions
│   ├── helpers.py
│   └── __init__.py
├── train.py               # Training script
├── sample.py              # Generation script
├── validate_cpu.py        # CPU validation suite
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔬 How It Works

### Diffusion Process

1. **Forward Process (Training)**
   ```
   Clean text embeddings → Add Gaussian noise → Noisy embeddings
   x_0 → x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
   ```

2. **Reverse Process (Sampling)**
   ```
   Pure noise → Iteratively denoise → Clean embeddings → Tokens
   x_T ~ N(0,I) → x_{T-1} → ... → x_0 → Discrete tokens
   ```

3. **Training Objective**
   ```
   Minimize: ||ε - ε_θ(x_t, t)||²
   (Predict the noise that was added)
   ```

### Architecture

```
Token IDs → Embeddings + Positional Encoding
                ↓
        Noisy Embeddings (x_t)
                ↓
        Timestep Embedding (t)
                ↓
        Transformer Encoder (24 layers)
                ↓
        Noise Prediction Head
                ↓
        Predicted Noise (ε_θ)
```

## 💻 Training Details

### CPU (Development)

```bash
python train.py --config tiny --cpu
```

- Model: 500K parameters
- Batch size: 2
- Precision: FP32
- Data: Synthetic
- Purpose: Validate pipeline

### GPU (Production)

```bash
python train.py --config full
```

- Model: 1.4B parameters
- Batch size: 8 × 16 = 128 (with gradient accumulation)
- Precision: BF16 mixed
- Strategy: FSDP (multi-GPU)
- Data: FineWeb (streaming)
- Steps: 500K
- Warmup: 5K steps

### Monitoring

Training metrics logged to WandB:
- Loss curve
- Learning rate
- Gradient norms
- Sampling quality

```bash
# Login to WandB (first time only)
wandb login

# Monitor at: https://wandb.ai
```

## 🎯 Sampling Strategies

### DDPM (Slow, Accurate)

```python
scheduler = DDPMScheduler(model)
embeddings = scheduler.sample(shape, num_steps=1000)
```

- Uses all 1000 diffusion steps
- Stochastic sampling
- Highest quality

### DDIM (Fast)

```python
scheduler = DDIMScheduler(model, eta=0.0)
embeddings = scheduler.sample(shape, num_steps=50)
```

- Skip most steps (50 instead of 1000)
- Deterministic (eta=0) or stochastic (eta=1)
- 20× faster, similar quality

### Constrained DDIM (Infilling)

```python
scheduler = ConstrainedDDIMScheduler(model, eta=0.0)
embeddings = scheduler.sample_with_constraint(
    shape, known_embeddings, mask, num_steps=50
)
```

- Fix certain positions
- Denoise only masked regions
- Perfect for text editing

## 🧪 Testing & Validation

### Pre-Training Checks

```bash
# 1. Test model initialization
python -c "from config import get_tiny_config; from models import SunFishTransformer; \
    model = SunFishTransformer(get_tiny_config()); \
    print(f'✅ {sum(p.numel() for p in model.parameters()):,} params')"

# 2. Test data pipeline
python -c "from config import get_tiny_config; from data import FineWebDataModule; \
    from utils import check_data_pipeline; \
    dm = FineWebDataModule(get_tiny_config()); \
    check_data_pipeline(dm, 10)"

# 3. Full validation suite
python validate_cpu.py
```

### Critical Sanity Check

**The model MUST overfit a single batch.** If it can't, the implementation is broken.

```python
from utils import overfit_single_batch

losses = overfit_single_batch(model, batch, num_steps=100)
# Expected: losses[-1] < 0.01
```

## 📈 Expected Results

### Loss Curve

- Initial loss: ~8-10
- After warmup: ~4-6
- Convergence: ~1-2
- Overfit single batch: < 0.01

### Generation Quality

- **Step 0**: Random noise → gibberish
- **Step 10K**: Some coherent phrases
- **Step 50K**: Mostly grammatical
- **Step 100K+**: High quality text

### Sampling Speed

| Method | Steps | Time (seq_len=512) |
|--------|-------|-------------------|
| DDPM   | 1000  | ~60s             |
| DDIM   | 50    | ~3s              |
| DDIM   | 10    | ~0.6s            |

## 🔧 Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce batch size
python train.py --config full  # Edit config: batch_size = 4

# Or use gradient checkpointing (in config)
activation_checkpointing = True
```

### Loss Spikes

- Check gradient clipping is enabled
- Reduce learning rate
- Increase warmup steps

### Slow Data Loading

- Increase `num_workers` in config
- Use local dataset instead of streaming
- Check disk I/O

### NaN Loss

- Use lower precision (FP32 instead of BF16)
- Reduce learning rate
- Check data for corrupted samples

## 📚 Implementation Notes

### Key Differences from Autoregressive LLMs

| Aspect | Autoregressive (GPT) | Diffusion (SunFish) |
|--------|---------------------|---------------------|
| Generation | Left-to-right, sequential | Iterative refinement |
| Speed | O(n) | O(n × steps) |
| Editing | Requires rewrite | Direct infilling |
| Training | Causal masking | Full context |
| Loss | Cross-entropy | MSE (noise prediction) |

### Design Decisions

1. **Pre-norm Transformer**: More stable than post-norm
2. **Linear beta schedule**: Simple and effective
3. **Cosine LR schedule**: Smooth convergence
4. **FSDP over DDP**: Better memory efficiency for large models
5. **Streaming data**: Infinite training data

## 🎓 Learning Resources

### Papers

- **DDPM**: Ho et al. 2020 - Denoising Diffusion Probabilistic Models
- **DDIM**: Song et al. 2020 - Denoising Diffusion Implicit Models
- **Diffusion-LM**: Li et al. 2022 - Diffusion-LM Improves Controllable Text Generation

### Code Structure

- `config/`: Hyperparameters (modify for experiments)
- `models/diffusion_transformer.py`: Core model (Section 6-7)
- `models/schedulers.py`: Sampling algorithms (Section 9)
- `data/`: Data pipeline (Section 5)
- `train.py`: Training loop (Section 8)
- `sample.py`: Generation (Section 10)

## 🚧 Roadmap

- [x] Basic diffusion transformer
- [x] DDPM/DDIM sampling
- [x] FineWeb data pipeline
- [x] CPU validation suite
- [ ] Multi-GPU FSDP training
- [ ] Checkpoint resumption
- [ ] Evaluation metrics (perplexity, MAUVE)
- [ ] Instruction fine-tuning
- [ ] LoRA support
- [ ] Quantization (INT8)

## 📝 Citation

```bibtex
@software{sunfish2025,
  title = {SunFish: Diffusion Language Model},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/yourusername/SunFish-Pretain}
}
```

## 🙏 Acknowledgments

- HuggingFace for FineWeb dataset
- PyTorch Lightning for training framework
- Anthropic for Claude (code generation)

## 📄 License

MIT License - See LICENSE file for details

---

**Built with 🐟 and diffusion magic** ✨
