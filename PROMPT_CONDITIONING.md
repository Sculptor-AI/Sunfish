# 🎯 Prompt Conditioning with Classifier-Free Guidance

SunFish now supports **prompt-conditioned generation** using Classifier-Free Guidance (CFG)!

## 🌟 What's New?

Your diffusion model can now:
- ✅ Generate text based on prompts (e.g., "Write about dogs")
- ✅ Control adherence strength via guidance_scale
- ✅ Train with both conditional and unconditional generation
- ✅ Use the same model for prompted and unprompted generation

## 🔬 How It Works

### Classifier-Free Guidance (CFG)

CFG trains a single model to support both conditional (with prompt) and unconditional (no prompt) generation:

```
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

**Training**: Randomly drop prompts 10-15% of the time (conditioning dropout)
**Sampling**: Run model twice (conditional + unconditional) and interpolate

### Architecture Changes

1. **Prompt Encoder**: Uses frozen GPT-2 embeddings + learnable projection
2. **Cross-Attention**: Each transformer block attends to prompt embeddings
3. **CFG Dropout**: Random conditioning dropout during training

## 🚀 Usage

### 1. Unconditional Generation (No Prompt)

```bash
# Standard unconditional generation
python sample.py checkpoints/last.ckpt \
    --num_samples 3 \
    --seq_len 256 \
    --num_steps 20
```

### 2. Conditional Generation (With Prompt)

```bash
# Generate with a prompt
python sample.py checkpoints/last.ckpt \
    --prompt "Write a story about a friendly robot" \
    --guidance_scale 7.5 \
    --num_samples 3 \
    --seq_len 256 \
    --num_steps 20
```

### 3. Adjusting Guidance Scale

```bash
# Low guidance (more creative, less adherent)
python sample.py checkpoints/last.ckpt \
    --prompt "Describe the ocean" \
    --guidance_scale 1.0

# Medium guidance (balanced) - RECOMMENDED
python sample.py checkpoints/last.ckpt \
    --prompt "Describe the ocean" \
    --guidance_scale 5.0

# High guidance (strict adherence, less diverse)
python sample.py checkpoints/last.ckpt \
    --prompt "Describe the ocean" \
    --guidance_scale 15.0
```

## ⚙️ Configuration

### Config Parameters (in `config/model_config.py`)

```python
# Classifier-Free Guidance (CFG) Parameters
conditioning_dropout: float = 0.1   # Training: % of time to drop prompts
guidance_scale: float = 7.5         # Sampling: CFG strength (1.0 = off)
prompt_max_length: int = 77         # Maximum prompt length in tokens
```

### Tiny Config Adjustments

The tiny config uses more aggressive settings for small model:

```python
conditioning_dropout: float = 0.15  # Higher dropout for small model
guidance_scale: float = 5.0         # Lower guidance for small model
prompt_max_length: int = 32         # Shorter prompts for small model
```

## 📚 Training with Prompts

### Option 1: Simple Synthetic Prompts

Create a custom data module that pairs text with simple prompts:

```python
# Example synthetic prompts
prompts = [
    "Write about animals",
    "Describe nature",
    "Tell a story",
    "Explain technology",
]

batch = {
    'tokens': token_ids,
    'prompts': [random.choice(prompts) for _ in range(batch_size)]
}
```

### Option 2: Generated Prompts from Text

Use the first few words as prompts:

```python
# Use first 5 words as prompt for the full text
text = "The cat sat on the mat and looked around."
prompt = " ".join(text.split()[:5])  # "The cat sat on the"
```

### Option 3: Continue with Unconditional

The model supports both modes! You can:
- Train unconditionally (no prompts) - model learns text distribution
- Add prompts later via fine-tuning
- Use unconditional mode during generation (just omit --prompt)

## 🎓 Expected Results by Training Stage

### CPU Training (Tiny Model)

| Steps | Expected Behavior |
|-------|-------------------|
| 0-500 | Gibberish (learning noise structure) |
| 500-2K | Basic token patterns, weak prompt correlation |
| 2K-5K | Some coherent words, basic keyword matching |
| 5K-10K | Simple sentences, noticeable prompt influence |
| 10K+ | Reasonable text, decent prompt adherence |

### GPU Training (Full Model)

| Steps | Expected Behavior |
|-------|-------------------|
| 0-5K | Warming up, basic patterns |
| 5K-50K | Grammatical text, moderate prompt adherence |
| 50K-100K | Good quality, strong prompt adherence |
| 100K+ | High quality, sophisticated prompt following |

## 💡 Tips & Tricks

### For CPU Training (Small Models)

1. **Use shorter prompts** (5-10 words max)
2. **Simple instructions** ("Write about X", "Describe Y")
3. **Lower guidance_scale** (3.0-5.0) - small models need less guidance
4. **More conditioning_dropout** (0.15-0.2) - helps generalization

### For GPU Training (Large Models)

1. **Longer, detailed prompts** work better
2. **Higher guidance_scale** (7.5-10.0) for strong adherence
3. **Standard conditioning_dropout** (0.1)
4. **Batch prompts together** - more efficient

### Debugging Prompts

If prompts aren't working:

1. **Check loss**: Should be similar to unconditional training
2. **Try guidance_scale=1.0**: Should match unconditional output
3. **Try guidance_scale=15.0**: Should show SOME prompt effect
4. **Train longer**: Prompt adherence improves with more training

## 🔧 Implementation Details

### Model Size Impact

Adding CFG increases parameters by:
- **Prompt projection**: ~200K params (768→dim×2→dim)
- **Cross-attention layers**: ~10-30% more params
- **Total**: Tiny model goes from ~800K to ~1.1M params

### Computational Cost

- **Training**: Same as unconditional (dropout is free)
- **Sampling with CFG**: 2x slower (two forward passes)
- **Sampling without CFG**: Same as before

### Memory Usage

- **Training**: +10-20% (prompt embeddings in memory)
- **Sampling**: +50% (two forward passes in parallel)

## 📊 Example Prompts to Try

### Simple (Good for Small Models)

```
"Write about dogs"
"Describe the ocean"
"Tell a short story"
"Explain computers"
```

### Medium (Better with More Training)

```
"Write a story about a friendly robot in a garden"
"Describe the feeling of watching a sunset"
"Explain how airplanes fly in simple terms"
```

### Complex (Large Models Only)

```
"Write a mysterious story set in Victorian London involving a lost artifact"
"Describe the philosophical implications of artificial intelligence"
```

## 🐛 Troubleshooting

### Error: "HuggingFace download failed"

The prompt encoder needs GPT-2 embeddings. First time you use prompts:
1. Ensure internet connection
2. Will download ~500MB (one-time)
3. Cached for future use

**Workaround**: Train/test without prompts first, add them later

### Prompts Have No Effect

- Model needs training on prompt data
- Try higher guidance_scale (10.0+)
- Train for more steps (5K+ minimum)

### Prompts Make Output Worse

- guidance_scale too high (try 3.0-7.5)
- conditioning_dropout too low (try 0.15)
- Model underfitted (train more)

## 📈 Next Steps

1. **Test basic functionality**: Run `python test_cfg.py`
2. **Try unconditional generation**: Train without prompts first
3. **Add prompts gradually**: Fine-tune with prompt data
4. **Experiment with guidance_scale**: Find sweet spot for your model
5. **Scale up**: Move to larger model when ready

---

**Happy prompting! 🎯✨**
