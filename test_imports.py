#!/usr/bin/env python3
"""
Quick import test - verify masked diffusion modules load correctly.
"""

import sys

print("Sunfish Masked Diffusion Import Test")
print("=" * 50)

# Core dependencies
try:
    import torch
    print(f"OK PyTorch {torch.__version__}")
except ImportError as e:
    print(f"FAIL PyTorch: {e}")
    sys.exit(1)

try:
    import pytorch_lightning as pl
    print(f"OK PyTorch Lightning {pl.__version__}")
except ImportError as e:
    print(f"FAIL PyTorch Lightning: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"OK Transformers {transformers.__version__}")
except ImportError as e:
    print(f"FAIL Transformers: {e}")
    sys.exit(1)

try:
    from datasets import load_dataset
    print("OK Datasets")
except ImportError as e:
    print(f"FAIL Datasets: {e}")
    sys.exit(1)

print()

# Project imports
try:
    from config import QwenMaskedDiffusionConfig, get_qwen_masked_config_cpu
    print("OK Config module")
except ImportError as e:
    print(f"FAIL Config: {e}")
    sys.exit(1)

try:
    from models import MaskedDiffusionLM, DiscreteDiffusionSampler
    print("OK Models module")
except ImportError as e:
    print(f"FAIL Models: {e}")
    sys.exit(1)

try:
    from data import QwenDataModule, QwenStreamDataset, QwenTextDataset
    print("OK Data module")
except ImportError as e:
    print(f"FAIL Data: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("All imports successful!")
print("=" * 50)
