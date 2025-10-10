#!/bin/bash
# Quick sampling script - NO PROMPTS (works without HuggingFace)

CHECKPOINT="${1:-checkpoints/last.ckpt}"
NUM_SAMPLES="${2:-3}"
SEQ_LEN="${3:-128}"
NUM_STEPS="${4:-20}"

echo "🐟 Quick Sampling (Unconditional)"
echo "================================="
echo "Checkpoint: $CHECKPOINT"
echo "Samples: $NUM_SAMPLES"
echo "Length: $SEQ_LEN tokens"
echo "Steps: $NUM_STEPS"
echo ""

source venv/bin/activate
python sample.py "$CHECKPOINT" \
    --num_samples "$NUM_SAMPLES" \
    --seq_len "$SEQ_LEN" \
    --num_steps "$NUM_STEPS"
