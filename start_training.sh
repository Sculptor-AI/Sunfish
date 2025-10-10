#!/bin/bash
# Quick start script for training SunFish with prompt conditioning

echo "🐟 SunFish Training - Micro Model (41M params)"
echo "=============================================="
echo ""
echo "This will train a model with prompt conditioning support"
echo "Expected: ~2-4 hours for first 5K steps on 16-core CPU"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

# Activate virtual environment
source venv/bin/activate

# Set experiment name
EXPERIMENT_NAME="micro-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "🚀 Starting training..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Config: micro (41M params)"
echo "Device: CPU (16 threads)"
echo ""
echo "📊 Monitor progress:"
echo "  - Watch terminal for train_loss"
echo "  - Checkpoints saved to: checkpoints/"
echo "  - Test samples: ./run.sh sample.py checkpoints/last.ckpt"
echo ""
echo "⏹️  To stop: Press Ctrl+C (progress will be saved)"
echo ""
sleep 2

# Start training
python train.py --config micro --cpu --name "$EXPERIMENT_NAME"

echo ""
echo "✅ Training session complete!"
echo "📁 Checkpoints saved in: checkpoints/"
echo ""
echo "🎲 Try generating samples:"
echo "  ./run.sh sample.py checkpoints/last.ckpt --num_samples 3"
echo ""
echo "📝 With prompts (after 5K+ steps):"
echo "  ./run.sh sample.py checkpoints/last.ckpt --prompt \"Write about nature\" --guidance_scale 7.0"
echo ""
