#!/usr/bin/env python3
"""
Quick import test - verify all modules load correctly
"""

import sys

print("🐟 SunFish Import Test")
print("=" * 50)

# Test core imports
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

try:
    import pytorch_lightning as pl
    print(f"✅ PyTorch Lightning {pl.__version__}")
except ImportError as e:
    print(f"❌ PyTorch Lightning: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")
    sys.exit(1)

try:
    from datasets import load_dataset
    print(f"✅ Datasets")
except ImportError as e:
    print(f"❌ Datasets: {e}")
    sys.exit(1)

print()

# Test project imports
try:
    from config import SunFishConfig, TinySunFishConfig, get_tiny_config
    print("✅ Config module")
except ImportError as e:
    print(f"❌ Config: {e}")
    sys.exit(1)

try:
    from models import SunFishTransformer, DDPMScheduler, DDIMScheduler
    print("✅ Models module")
except ImportError as e:
    print(f"❌ Models: {e}")
    sys.exit(1)

try:
    from data import FineWebDataModule, TinyTextDataset
    print("✅ Data module")
except ImportError as e:
    print(f"❌ Data: {e}")
    sys.exit(1)

try:
    from utils import count_parameters, check_data_pipeline
    print("✅ Utils module")
except ImportError as e:
    print(f"❌ Utils: {e}")
    sys.exit(1)

print()

# Quick functionality test
try:
    config = get_tiny_config()
    model = SunFishTransformer(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model creation: {n_params:,} parameters")
except Exception as e:
    print(f"❌ Model creation: {e}")
    sys.exit(1)

try:
    from data import TinyTextDataset
    dataset = TinyTextDataset(num_samples=10)
    batch = dataset[0]
    print(f"✅ Data loading: batch shape {batch.shape}")
except Exception as e:
    print(f"❌ Data loading: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("🎉 All imports successful!")
print("=" * 50)
print()
print("Next steps:")
print("  1. Run: python validate_cpu.py")
print("  2. Check: QUICKSTART.md")
print("  3. Train: python train.py --config tiny --cpu")
print()
