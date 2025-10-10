# 🏋️ SunFish Training Guide - CPU Edition

## 🎯 Recommended: Micro Model (41M Parameters)

The **micro config** is the sweet spot for CPU training with prompt conditioning:
- **Size**: 41.6M parameters (with CFG)
- **Training time**: ~2-4 hours for 5K steps on 16-core CPU
- **Expected quality**: Coherent words/phrases after 10K+ steps
- **Memory**: Fits comfortably in 16GB RAM
- **Prompt support**: Full CFG with 64-token prompts

## 🚀 Quick Start

### 1. Start Training

```bash
# Activate environment
source venv/bin/activate

# Start training the micro model
python train.py --config micro --cpu --name my-first-model
```

This will:
- Create `checkpoints/` directory
- Save checkpoints every 1000 steps
- Log to WandB or TensorBoard
- Train for up to 50,000 steps

### 2. Monitor Progress

Watch the terminal for:
```
train_loss: 8.xxx  → Initial (random)
train_loss: 4.xxx  → Learning basic patterns (500-2K steps)
train_loss: 2.xxx  → Learning token structure (2K-10K steps)
train_loss: 1.xxx  → Coherent generation starts (10K+ steps)
```

### 3. Generate Samples

After 1000+ steps:

```bash
# Unconditional generation (no prompt)
python sample.py checkpoints/last.ckpt \
    --num_samples 3 \
    --seq_len 128 \
    --num_steps 20

# Conditional generation (with prompt)
python sample.py checkpoints/last.ckpt \
    --prompt "Write about nature" \
    --guidance_scale 7.0 \
    --num_samples 3 \
    --seq_len 128 \
    --num_steps 20
```

## 📊 Model Comparison

| Config | Parameters | Training Speed | Quality | CPU Recommended |
|--------|-----------|---------------|---------|-----------------|
| **Nano** | ~300K | Fast (min) | Gibberish | Testing only |
| **Tiny** | ~1M | Fast (min) | Weak patterns | Testing only |
| **Micro** | **~42M** | **Medium (hours)** | **Coherent words** | **✅ YES** |
| Full | ~1.4B | Slow (days/weeks) | High quality | ❌ NO (GPU only) |

## ⏱️ Estimated Training Times (16-core CPU)

| Steps | Time | Expected Output |
|-------|------|----------------|
| 1K | ~30-60 min | Random noise, basic patterns |
| 5K | ~2-4 hours | Some token structure, weak coherence |
| 10K | ~4-8 hours | Coherent words, simple phrases |
| 20K | ~8-16 hours | Good sentences, basic prompt following |
| 50K | ~20-40 hours | Quality text, decent prompt adherence |

## 🎛️ Training Options

### Resume Training

```bash
# Resume from checkpoint
python train.py --config micro --cpu --resume checkpoints/last.ckpt
```

### Adjust Settings

Edit `config/micro_config.py`:

```python
# Slower training, better quality
learning_rate: float = 3e-4  # Reduce from 5e-4
max_steps: int = 100000      # Train longer

# Faster training, lower quality
batch_size: int = 8          # Increase from 4
learning_rate: float = 7e-4  # Increase from 5e-4

# Prompt conditioning
conditioning_dropout: float = 0.15  # More dropout = better generalization
guidance_scale: float = 5.0         # Lower = less strict prompt following
```

## 💾 Checkpoints

Checkpoints are saved to `checkpoints/` every 1000 steps:

```
checkpoints/
├── last.ckpt                    # Most recent
├── sunfish-epoch=00-step=1000.ckpt
├── sunfish-epoch=00-step=2000.ckpt
└── ...
```

**Tip**: Keep the best checkpoint based on generated samples, not loss!

## 🔍 Monitoring

### Option 1: WandB (Recommended)

```bash
# Login (first time)
wandb login

# Training automatically logs to wandb.ai
python train.py --config micro --cpu --name experiment-1
```

View at: `https://wandb.ai/<your-username>/sunfish-micro`

### Option 2: TensorBoard (Offline)

```bash
# Start TensorBoard in another terminal
tensorboard --logdir logs/

# View at: http://localhost:6006
```

## 📈 What to Expect

### Training Progression

**Steps 0-1000**: Random Gibberish
```
"aslkdfj qwerty mnbvcx zxcvbn..."
```

**Steps 1000-5000**: Token Patterns
```
"the the the and of to in..."
```

**Steps 5000-10000**: Word Structure
```
"the cat dog and house big tree..."
```

**Steps 10000-20000**: Simple Sentences
```
"the dog is running in the park and the cat..."
```

**Steps 20000+**: Coherent Text
```
"The dog ran through the park, chasing after a ball..."
```

### Prompt Adherence

**Early Training (1K-5K steps)**:
- Prompt: "Write about dogs"
- Output: Random words, ignores prompt

**Mid Training (5K-10K steps)**:
- Prompt: "Write about dogs"
- Output: Some "dog"-related words appear

**Late Training (10K+ steps)**:
- Prompt: "Write about dogs"
- Output: Actually writes about dogs!

## 🐛 Troubleshooting

### Loss Not Decreasing

- **Normal**: Stays ~8-10 for first 500 steps
- **Check**: Should drop to ~4-6 by 2000 steps
- **If stuck**:
  - Increase learning_rate to 7e-4
  - Check data is loading (not all zeros)

### Out of Memory

```python
# Reduce batch size
batch_size: int = 2
accumulate_grad_batches: int = 16  # Keep effective batch = 32
```

### Training Too Slow

```python
# Reduce model size
n_layer: int = 6        # From 8
n_head: int = 6         # From 8
n_embd: int = 384       # From 512
```

### Prompts Not Working

- **Train longer**: Need 5K+ steps for basic adherence
- **Lower guidance_scale**: Try 3.0-5.0 instead of 7.0
- **Check internet**: First prompt download needs GPT-2 embeddings (~500MB)

## 🎯 Tips for Best Results

### 1. Start Unconditional

Train without worrying about prompts first:
```bash
# Just train the base model
python train.py --config micro --cpu --name base-model

# Test without prompts
python sample.py checkpoints/last.ckpt --num_samples 3
```

### 2. Patience Pays Off

- First 5K steps: Model learns noise structure
- 5K-10K steps: Learns token patterns
- 10K-20K steps: Starts making sense
- 20K+ steps: Quality improves noticeably

### 3. Monitor Samples, Not Loss

Generate samples every 1-2K steps:
```bash
# Quick test generation
python sample.py checkpoints/sunfish-epoch=00-step=5000.ckpt \
    --num_samples 1 \
    --seq_len 64 \
    --num_steps 10
```

### 4. Experiment with Sampling

```bash
# Fast (lower quality)
--num_steps 10 --scheduler ddim

# Balanced
--num_steps 20 --scheduler ddim

# High quality (slower)
--num_steps 50 --scheduler ddim

# Max quality (very slow)
--num_steps 100 --scheduler ddpm
```

## 🔬 Advanced: Fine-Tuning with Prompts

Once you have a base model (10K+ steps), you can fine-tune for prompts:

1. **Create prompt dataset** (beyond scope - needs custom data module)
2. **Fine-tune** with prompt data:
   ```bash
   python train.py --config micro --cpu \
       --resume checkpoints/best-base.ckpt \
       --name prompt-finetuning
   ```
3. **Test with prompts**:
   ```bash
   python sample.py checkpoints/last.ckpt \
       --prompt "Your prompt here" \
       --guidance_scale 7.0
   ```

## 📝 Recommended Workflow

### Day 1: Start Training
```bash
source venv/bin/activate
python train.py --config micro --cpu --name day1
# Leave running overnight
```

### Day 2: Check Progress
```bash
# Test at 5K steps
python sample.py checkpoints/sunfish-epoch=00-step=5000.ckpt \
    --num_samples 3 --seq_len 64

# Resume training
python train.py --config micro --cpu --resume checkpoints/last.ckpt
```

### Day 3+: Iterate
```bash
# Generate samples at different checkpoints
# Find the best one
# Resume training or adjust config
```

## 🎉 Success Metrics

You'll know it's working when:
- ✅ Loss drops below 2.0
- ✅ Samples contain real words
- ✅ Some grammatical structure appears
- ✅ Prompts have noticeable effect (at guidance_scale=10+)

Remember: **This is a tiny model running on CPU**. Don't expect GPT-4 quality, but DO expect:
- Coherent words and phrases
- Basic sentence structure
- Some prompt following
- Interesting text patterns

---

**Happy training! 🐟✨**
