# SunFish Usage Guide

## Quick Start

### Generate Text (Simple)
```bash
./generate.sh
```

### Generate Text (Custom)
```bash
./generate.sh checkpoints/sunfish-epoch=00-step=500-v1.ckpt 10 128 50
#               └─ checkpoint path                           │  │   │
#                                                            │  │   └─ DDIM steps (quality)
#                                                            │  └───── Length (tokens)
#                                                            └──────── Number of samples
```

### Generate Text (Advanced)
```bash
venv/bin/python sample.py checkpoints/sunfish-epoch=00-step=500-v1.ckpt \
    --num_samples 5 \
    --seq_len 128 \
    --num_steps 50 \
    --scheduler ddim
```

## Training

### Nano Model (4.3M params, ~30 min on CPU)
```bash
./run_nano.sh
```

### Micro Model (32M params, needs GPU)
```bash
./run_micro.sh
```

### Custom Training
```bash
venv/bin/python train.py --config nano --cpu --name my-experiment
```

## Available Checkpoints

After training, find checkpoints in `checkpoints/`:
- `sunfish-epoch=00-step=500-v1.ckpt` - Final checkpoint (500 steps)
- `sunfish-epoch=00-step=400-v1.ckpt` - Checkpoint at step 400
- `last-v1.ckpt` - Latest checkpoint (same as final)

## Tips

**Better Quality:**
- Use more DDIM steps: `--num_steps 100` (slower but better)
- Use longer training: increase `max_steps` in config
- Use larger model: try `micro` config

**Faster Generation:**
- Use fewer DDIM steps: `--num_steps 10` (faster but worse)
- Use shorter sequences: `--seq_len 32`

**More Coherent Text:**
- Train longer (current: 500 steps → try 5000+)
- Use more training data
- Use larger model with more parameters
