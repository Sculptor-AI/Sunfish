# Sunfish Diffusion LLM - Quick Start Guide

This guide gets you training in 5 minutes.

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Test installation
python test_setup.py
```

## Step 2: Verify Setup

The test script should show all green checkmarks:

```
✓ PASS: Imports
✓ PASS: CUDA
✓ PASS: Configuration
✓ PASS: Model
✓ PASS: Data
✓ PASS: Samplers

🎉 All tests passed! You're ready to start training.
```

## Step 3: Quick Training Test (5 minutes)

Test that training works with a tiny model:

```bash
python train.py --config tiny --fast-dev-run
```

This runs 1 batch to verify everything is working.

## Step 4: Overfit Test (10 minutes)

Verify the model can learn by overfitting on a few batches:

```bash
python train.py --config tiny --overfit-batches 10 --max-steps 200
```

Loss should drop from ~1.0 to <0.1.

## Step 5: Download Training Data (Recommended)

For reliable, fast training, download a local copy of the data first:

```bash
# Download 500GB of FineWeb (takes 2-6 hours depending on internet speed)
python download_dataset.py --size 500GB --output data/fineweb_local

# Or download 1TB for full training
python download_dataset.py --size 1TB --output data/fineweb_local
```

**Alternative:** You can skip this and stream data during training, but it's slower and requires constant internet.

## Step 6: Full Training (3-5 days)

Launch full training on the 1.4B model:

```bash
# With local dataset (RECOMMENDED - faster, more reliable)
python train.py --config 1.4B --local-dataset data/fineweb_local

# Or stream from HuggingFace (requires internet)
python train.py --config 1.4B

# Monitor with W&B
# Visit: https://wandb.ai/your-username/sunfish-diffusion-llm

# Or use TensorBoard
tensorboard --logdir logs/tensorboard
```

### Training Configuration

The default configuration targets:
- **Model**: 1.4B parameters (24 layers, 2048 dim)
- **Data**: 500GB of FineWeb (streaming)
- **Batch**: 256 sequences × 2048 tokens = 524k tokens/step
- **Steps**: 500,000 (approximately 3-5 days on 2× A6000)
- **Checkpoints**: Saved every 5,000 steps in `checkpoints/`

## Step 7: Monitor Training

### Via Weights & Biases

```bash
# View training metrics in browser
wandb login  # First time only
# Visit your W&B dashboard
```

### Via TensorBoard

```bash
tensorboard --logdir logs/tensorboard
# Visit http://localhost:6006
```

### Via Logs

```bash
# Follow training logs
tail -f logs/train_*.log
```

## Step 8: Generate Text

Once you have a checkpoint (even partially trained):

```bash
# Generate 5 samples
python sample.py checkpoints/last.ckpt \
    --num-samples 5 \
    --length 256 \
    --num-steps 50

# Fill in masked text
python sample.py checkpoints/last.ckpt \
    --mode infill \
    --prompt "The future of AI is [MASK] and revolutionary." \
    --mask-length 20
```

### Sampling Tips

- **Fewer steps (20-50)**: Faster, lower quality
- **More steps (100-200)**: Slower, higher quality
- **DDIM (eta=0)**: Deterministic, recommended
- **DDPM**: Stochastic, original method

## Common Commands

### Resume Training

```bash
python train.py --config 1.4B --resume checkpoints/last.ckpt
```

### Adjust Batch Size (if OOM)

```bash
python train.py --config 1.4B --batch-size 4
```

### Train on Smaller Model

```bash
# 350M parameters (faster, less memory)
python train.py --config small
```

### Disable W&B

```bash
python train.py --config 1.4B --no-wandb
```

## Expected Results

### Training Loss

- **Step 0**: ~1.0 (random)
- **Step 50k**: ~0.4-0.5
- **Step 200k**: ~0.25-0.3
- **Step 500k**: ~0.15-0.2

### Generation Quality

- **< 50k steps**: Mostly gibberish
- **50k-100k steps**: Some coherent phrases
- **100k-200k steps**: Short coherent sentences
- **200k+ steps**: Longer coherent text
- **500k steps**: Good quality, some knowledge

## Troubleshooting

### Out of Memory

```bash
# Solution 1: Reduce batch size
python train.py --config 1.4B --batch-size 4

# Solution 2: Enable activation checkpointing
# Edit config/model_config.py:
# fsdp_activation_checkpointing = True
```

### Slow Data Loading

```bash
# Increase workers
python train.py --config 1.4B --num-workers 8
```

### NaN Loss

```bash
# Lower learning rate
python train.py --config 1.4B --learning-rate 5e-5
```

## Next Steps

1. **Monitor**: Watch loss curves in W&B/TensorBoard
2. **Sample**: Generate text periodically to check quality
3. **Experiment**: Try different hyperparameters
4. **Fine-tune**: After pretraining, fine-tune on specific tasks

## File Locations

- **Checkpoints**: `checkpoints/*.ckpt`
- **Logs**: `logs/train_*.log`
- **TensorBoard**: `logs/tensorboard/`
- **W&B**: Online at wandb.ai

## Need Help?

- Check `README.md` for full documentation
- Run `python test_setup.py` to verify installation
- Check logs in `logs/` directory
- Open an issue on GitHub

## Hardware Requirements

**Minimum** (for training):
- 1× NVIDIA A6000 (48GB) or equivalent
- 128GB RAM
- 100GB free disk space

**Recommended** (for full speed):
- 2× NVIDIA A6000 (48GB)
- 256GB RAM
- 500GB NVMe SSD

**For inference only**:
- 1× NVIDIA GPU with 8GB+ VRAM
- 32GB RAM

---

Happy training! 🚀
