#!/bin/bash
# SunFish Text Generation - Simple Wrapper

CHECKPOINT="${1:-checkpoints/sunfish-epoch=00-step=500-v1.ckpt}"
NUM_SAMPLES="${2:-5}"
LENGTH="${3:-64}"
STEPS="${4:-20}"

echo "🐟 SunFish Text Generator"
echo "========================="
echo "Checkpoint: $CHECKPOINT"
echo "Samples: $NUM_SAMPLES"
echo "Length: $LENGTH tokens"
echo "DDIM steps: $STEPS"
echo ""

venv/bin/python sample.py \
    "$CHECKPOINT" \
    --num_samples "$NUM_SAMPLES" \
    --seq_len "$LENGTH" \
    --num_steps "$STEPS" \
    --scheduler ddim
