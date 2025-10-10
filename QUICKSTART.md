# 🐟 SunFish Quick Start

## 🚀 Setup (5 minutes)

```bash
# Automated setup
./setup.sh

# OR manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧪 Validate Pipeline (Current Machine - CPU)

```bash
# Run full validation suite
python validate_cpu.py
```

This tests:
- Model initialization ✅
- Data loading ✅
- Forward/backward pass ✅
- Overfitting (critical!) ✅
- Sampling ✅
- Mini training loop ✅

## 🏋️ Training

### CPU (Testing)
```bash
# Tiny model on CPU (for development)
python train.py --config tiny --cpu --name cpu-test
```

### GPU (Production)
```bash
# Full model on GPU
python train.py --config full --name production-run

# Resume from checkpoint
python train.py --config full --resume checkpoints/last.ckpt
```

## 🎲 Generate Text

### Unconditional Generation
```bash
# Generate 5 samples of length 512
python sample.py checkpoints/last.ckpt \
    --num_samples 5 \
    --seq_len 512 \
    --num_steps 50

# Fast generation (10 steps)
python sample.py checkpoints/last.ckpt \
    --num_samples 3 \
    --seq_len 256 \
    --num_steps 10 \
    --scheduler ddim
```

### Text Infilling
```bash
# Fill in [MASK]
python sample.py checkpoints/last.ckpt \
    --mode infill \
    --text "The future of AI is [MASK] and revolutionary." \
    --num_steps 50
```

## 🔬 Testing & Debugging

### Test Model
```python
from config import get_tiny_config
from models import SunFishTransformer

config = get_tiny_config()
model = SunFishTransformer(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Test Data
```python
from config import get_tiny_config
from data import FineWebDataModule

dm = FineWebDataModule(get_tiny_config())
dm.setup()
batch = next(iter(dm.train_dataloader()))
print(f"Batch shape: {batch.shape}")
```

### Overfit Single Batch (Sanity Check)
```python
from utils import overfit_single_batch
import torch

batch = torch.randint(0, 1024, (2, 128))
losses = overfit_single_batch(model, batch, num_steps=100)
# losses[-1] should be < 0.01
```

## 📊 Monitoring

### WandB
```bash
# Login (first time)
wandb login

# View at: https://wandb.ai/<username>/<project>
```

### TensorBoard (fallback)
```bash
tensorboard --logdir logs/
```

## 🔧 Common Issues

### OOM Error
```python
# In config/model_config.py or config/tiny_config.py
batch_size = 4  # Reduce from 8
accumulate_grad_batches = 32  # Increase to maintain effective batch size
```

### Slow Data
```python
# In config
num_workers = 8  # Increase (CPU cores - 2)
```

### Loss NaN
- Use FP32 instead of BF16
- Reduce learning rate
- Check data for NaN values

## 📈 Expected Timeline

| Phase | Duration | Checkpoint |
|-------|----------|-----------|
| Setup & Validation | 30 min | `validate_cpu.py` passes |
| CPU Tiny Test | 1 hour | Loss < 1.0 |
| GPU Tiny Test | 2 hours | Loss < 0.5 |
| GPU Full Training | 3-7 days | Loss < 2.0 |
| Fine-tuning | 1-2 days | High quality text |

## 🎯 Model Configs

### Tiny (CPU Testing)
- Parameters: ~500K
- Layers: 2
- Heads: 2
- Embedding: 128
- Sequence: 128
- **Use**: Pipeline validation

### Full (GPU Training)
- Parameters: ~1.4B
- Layers: 24
- Heads: 16
- Embedding: 2048
- Sequence: 2048
- **Use**: Production model

## 💡 Pro Tips

1. **Always validate on CPU first** - Catch bugs early
2. **Overfit single batch** - Critical sanity check
3. **Start with tiny config on GPU** - Test infrastructure
4. **Monitor gradient norms** - Detect instabilities
5. **Use DDIM sampling** - 20x faster than DDPM
6. **Save checkpoints frequently** - Training can be interrupted
7. **Log to WandB** - Track experiments systematically

## 📚 File Overview

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `sample.py` | Text generation |
| `validate_cpu.py` | CPU validation suite |
| `config/model_config.py` | Full model config |
| `config/tiny_config.py` | Tiny model config |
| `models/diffusion_transformer.py` | Core model |
| `models/schedulers.py` | Sampling algorithms |
| `data/fineweb_datamodule.py` | Data pipeline |
| `utils/helpers.py` | Helper functions |

## 🐛 Debug Checklist

Before asking for help:

- [ ] CPU validation passes
- [ ] Model can overfit single batch (loss < 0.01)
- [ ] Forward pass completes without errors
- [ ] Gradients are computed (not None)
- [ ] Data loads without errors
- [ ] GPU memory usage is reasonable
- [ ] No NaN in loss or gradients

---

**Happy Diffusing! 🐟✨**
