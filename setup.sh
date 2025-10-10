#!/bin/bash

# SunFish Setup Script
# Automated setup for development

set -e

echo "🐟 SunFish Setup Script"
echo "======================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Unknown"
fi

echo "Detected OS: $OS"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi

echo "✅ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
    print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints
mkdir -p logs
echo "✅ Directories created"
echo ""

# Run validation
echo "Running CPU validation..."
python3 validate_cpu.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Train tiny model: python train.py --config tiny --cpu"
echo "  3. Check README.md for more options"
echo ""
