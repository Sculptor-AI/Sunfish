# 🎯 SunFish with Prompt Conditioning - Ready to Train!

## ✨ What's New

Your SunFish diffusion model now supports **prompt-conditioned generation** using Classifier-Free Guidance (CFG)!

### New Capabilities

1. **Prompt Conditioning**: Generate text based on prompts like "Write about dogs"
2. **Guidance Control**: Adjust how strictly the model follows prompts (guidance_scale)
3. **Backward Compatible**: Still works unconditionally (without prompts)
4. **CPU-Optimized**: 41M parameter "micro" model runs on your 16-core CPU

## 🚀 Quick Start (3 Steps)

### 1. Start Training

```bash
./start_training.sh
```

This trains a **41.6M parameter model** with full prompt conditioning support.

### 2. Wait & Monitor

**Expected timeline**:
- **1K steps** (~30-60 min): Basic patterns
- **5K steps** (~2-4 hours): Token structure
- **10K steps** (~4-8 hours): Coherent words ← **First usable checkpoint**
- **20K+ steps** (~8-16 hours): Quality text, prompt following

Watch the loss in your terminal:
```
train_loss: 8.xxx  → Random (initial)
train_loss: 4.xxx  → Learning (500-2K steps)
train_loss: 2.xxx  → Structure (2K-10K steps)
train_loss: 1.xxx  → Coherent (10K+ steps)
```

### 3. Generate Samples

After 10K+ steps:

```bash
# Unconditional (no prompt)
./run.sh sample.py checkpoints/last.ckpt \
    --num_samples 3 \
    --seq_len 128 \
    --num_steps 20

# With prompt
./run.sh sample.py checkpoints/last.ckpt \
    --prompt "Write a story about a robot" \
    --guidance_scale 7.0 \
    --num_samples 3 \
    --seq_len 128 \
    --num_steps 20
```

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training instructions
- **[PROMPT_CONDITIONING.md](PROMPT_CONDITIONING.md)** - How CFG works and usage
- **[QUICKSTART.md](QUICKSTART.md)** - Original quick start guide

## 🔧 Your Setup

- **CPU**: AMD Ryzen 7 2700X (16 threads)
- **RAM**: 16GB
- **GPU**: None
- **Recommended Model**: Micro (41M params)

## 📊 Model Options

| Model | Params | Purpose | Training Time |
|-------|--------|---------|---------------|
| Nano | 300K | Testing only | Minutes |
| Tiny | 1M | Testing only | Minutes |
| **Micro** | **41M** | **CPU training** ← **Use this** | **Hours** |
| Full | 1.4B | GPU only | Days |

## 🎯 What to Expect

### Realistic Expectations

✅ **You CAN achieve**:
- Coherent words and phrases
- Basic grammatical structure
- Simple prompt following (keywords, topics)
- Interesting text patterns
- Fun experimental outputs

❌ **You CANNOT achieve** (with CPU/small model):
- GPT-4 level quality
- Complex reasoning
- Perfect grammar
- Consistent long-form content
- Strong prompt adherence

**This is a 41M parameter model on CPU** - think of it as a fun experiment and learning experience, not production-ready text generation!

## 🛠️ Common Commands

```bash
# Start training
./start_training.sh

# Resume training
./run.sh train.py --config micro --cpu --resume checkpoints/last.ckpt

# Generate unconditional
./run.sh sample.py checkpoints/last.ckpt --num_samples 3

# Generate with prompt
./run.sh sample.py checkpoints/last.ckpt \
    --prompt "Your prompt here" \
    --guidance_scale 7.0

# Test CFG implementation
./run.sh test_cfg.py

# Check model size
./run.sh -c "from config.micro_config import get_micro_config; get_micro_config()"
```

## 📁 Project Structure

```
SunFish-Pretain/
├── 🆕 TRAINING_GUIDE.md          # Comprehensive training guide
├── 🆕 PROMPT_CONDITIONING.md     # CFG usage and examples
├── 🆕 test_cfg.py                # Test CFG implementation
├── 🆕 start_training.sh          # Quick start script
├── 🆕 run.sh                     # Helper script (activates venv)
├──
├── config/
│   ├── micro_config.py           # 🆕 Updated with CFG params
│   ├── tiny_config.py            # 🆕 Updated with CFG params
│   └── model_config.py           # 🆕 Updated with CFG params
├── models/
│   ├── diffusion_transformer.py  # 🆕 CFG architecture added
│   └── schedulers.py             # 🆕 CFG sampling added
├── sample.py                      # 🆕 Added --prompt & --guidance_scale
├── train.py                       # (Unchanged - works with new model)
└── checkpoints/                   # Created during training
```

## 🔄 Migration from Old Checkpoints

**Old checkpoints are INCOMPATIBLE** with the new architecture because:
1. Changed from `nn.TransformerEncoder` to custom blocks with cross-attention
2. Added prompt encoder and projection layers
3. New model has more parameters

**Solution**: Train a fresh model with the new architecture!
- Old checkpoints backed up to: `checkpoints_old_architecture/`
- New training will create compatible checkpoints

## 💡 Tips for Success

1. **Be Patient**: First 5K steps are learning noise structure, not text
2. **Monitor Samples**: Generate every 1-2K steps to see progress
3. **Start Simple**: Try unconditional generation first
4. **Experiment**: Try different guidance_scale values (3.0, 7.0, 10.0)
5. **Save Checkpoints**: Keep multiple checkpoints, pick the best based on samples

## 🐛 Troubleshooting

### "No module named pytorch_lightning"
```bash
# Forgot to activate venv - use:
./run.sh <your_command>
# OR
source venv/bin/activate
python <your_command>
```

### "HuggingFace download failed"
- First time using prompts needs internet for GPT-2 embeddings (~500MB)
- Downloads once, cached for future use
- Can train/test without prompts if offline

### Out of Memory
```python
# In config/micro_config.py
batch_size: int = 2  # Reduce from 4
```

### Training too slow
- Expected! CPU training takes time
- Consider leaving overnight
- Or reduce model: n_layer=6, n_embd=384

## 🎉 You're Ready!

Everything is set up and ready to go:

```bash
# Just run this to start training:
./start_training.sh
```

Then grab a coffee and watch the magic happen! ☕✨

---

**Questions?** Check the documentation files or open an issue on GitHub.

**Happy training! 🐟🎯**
