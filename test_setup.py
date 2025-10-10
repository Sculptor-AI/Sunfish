#!/usr/bin/env python3
"""
Quick setup test script to verify installation and basic functionality.
Run this after installation to check if everything is working.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")

    try:
        import torch

        logger.info(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ PyTorch: {e}")
        return False

    try:
        import pytorch_lightning as pl

        logger.info(f"  ✓ PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ PyTorch Lightning: {e}")
        return False

    try:
        import transformers

        logger.info(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ Transformers: {e}")
        return False

    try:
        import datasets

        logger.info(f"  ✓ Datasets {datasets.__version__}")
    except ImportError as e:
        logger.error(f"  ✗ Datasets: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    logger.info("\nTesting CUDA...")

    import torch

    if torch.cuda.is_available():
        logger.info(f"  ✓ CUDA available")
        logger.info(f"  ✓ CUDA version: {torch.version.cuda}")
        logger.info(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            logger.info(f"  ✓ GPU {i}: {props.name} ({memory_gb:.1f} GB)")

        return True
    else:
        logger.warning("  ⚠ CUDA not available - training will use CPU (very slow)")
        return False


def test_config():
    """Test configuration loading."""
    logger.info("\nTesting configuration...")

    try:
        from config.model_config import get_config_tiny, get_config_1_4B

        config = get_config_tiny()
        logger.info(f"  ✓ Config loaded successfully")
        logger.info(f"  ✓ Estimated parameters: {config.get_parameter_count():,}")

        return True
    except Exception as e:
        logger.error(f"  ✗ Config loading failed: {e}")
        return False


def test_model():
    """Test model initialization."""
    logger.info("\nTesting model initialization...")

    try:
        import torch
        from config.model_config import get_config_tiny
        from models.diffusion_model import DiffusionTransformer

        config = get_config_tiny()
        config.n_layer = 2  # Very small for quick test

        model = DiffusionTransformer(config)
        logger.info(f"  ✓ Model created: {model.count_parameters():,} parameters")

        # Test forward pass
        batch_size = 2
        seq_len = 64
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, config.timesteps, (batch_size,))

        noisy_emb, noise = model.get_noisy_embeddings(token_ids, timesteps)
        pred_noise = model(noisy_emb, timesteps)

        logger.info(f"  ✓ Forward pass successful")
        logger.info(f"  ✓ Input shape: {token_ids.shape}")
        logger.info(f"  ✓ Output shape: {pred_noise.shape}")

        return True
    except Exception as e:
        logger.error(f"  ✗ Model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data():
    """Test data loading (without actually downloading data)."""
    logger.info("\nTesting data module...")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info(f"  ✓ Tokenizer loaded: {len(tokenizer)} tokens")

        # Test tokenization
        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer(text)
        logger.info(f"  ✓ Tokenization works: {len(tokens['input_ids'])} tokens")

        return True
    except Exception as e:
        logger.error(f"  ✗ Data test failed: {e}")
        return False


def test_schedulers():
    """Test sampling schedulers."""
    logger.info("\nTesting samplers...")

    try:
        import torch
        from config.model_config import get_config_tiny
        from models.diffusion_model import DiffusionTransformer
        from models.schedulers import DDIMSampler, round_embeddings_to_tokens

        config = get_config_tiny()
        config.n_layer = 2

        model = DiffusionTransformer(config)
        model.eval()

        # Test DDIM sampler
        sampler = DDIMSampler(model, eta=0.0)
        shape = (1, 16, config.n_embd)

        with torch.no_grad():
            samples = sampler.sample(shape, num_steps=5, show_progress=False)
            token_ids = round_embeddings_to_tokens(samples, model.token_embedding)

        logger.info(f"  ✓ DDIM sampling works")
        logger.info(f"  ✓ Generated tokens: {token_ids.shape}")

        return True
    except Exception as e:
        logger.error(f"  ✗ Sampler test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Sunfish Setup Test")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Configuration", test_config),
        ("Model", test_model),
        ("Data", test_data),
        ("Samplers", test_schedulers),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"\nTest '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed! You're ready to start training.")
        return 0
    else:
        print("\n  ⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
