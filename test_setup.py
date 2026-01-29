#!/usr/bin/env python3
"""
Quick diagnostic script to verify the masked diffusion setup.
Run this to check GPU usage, model loading, and basic functionality.
"""

import torch
import time

def check_cuda():
    """Check CUDA setup."""
    print("=" * 60)
    print("CUDA DIAGNOSTIC")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"Free memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

    return torch.cuda.is_available()


def test_model_loading():
    """Test model loads properly on GPU."""
    print("\n" + "=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)

    from config.qwen_masked_config import get_qwen_masked_config
    from models.masked_diffusion_lm import MaskedDiffusionLM

    config = get_qwen_masked_config()
    print(f"Config: batch_size={config.batch_size}, block_size={config.block_size}")

    print("\nLoading model (this downloads Qwen3-0.6B if not cached)...")
    start = time.time()
    model = MaskedDiffusionLM(config)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nMoving model to {device}...")
    model = model.to(device)
    model.eval()

    if torch.cuda.is_available():
        print(f"GPU memory after loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    print(f"\nModel info:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Mask token ID: {model.mask_token_id}")

    return model, device


def test_forward_pass(model, device):
    """Test a single forward pass."""
    print("\n" + "=" * 60)
    print("FORWARD PASS TEST")
    print("=" * 60)

    batch_size = 2
    seq_len = 128

    # Create test input
    test_input = torch.randint(0, model.vocab_size - 1, (batch_size, seq_len), device=device)

    print(f"Input shape: {test_input.shape}")

    start = time.time()
    with torch.no_grad():
        # Test masking
        t = torch.tensor([500, 250], device=device)
        masked_tokens, mask = model.forward_mask(test_input, t)
        print(f"Mask rate: {mask.float().mean():.2%}")

        # Test forward
        logits = model.forward(masked_tokens)
        print(f"Output logits shape: {logits.shape}")

    forward_time = time.time() - start
    print(f"Forward pass time: {forward_time:.3f}s")

    if torch.cuda.is_available():
        print(f"GPU memory after forward: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    return True


def test_generation(model, device):
    """Test text generation."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    from models.discrete_sampler import DiscreteDiffusionSampler

    sampler = DiscreteDiffusionSampler(model)

    print("Generating 1 sample with 64 tokens, 10 steps...")
    start = time.time()

    tokens = sampler.sample(
        batch_size=1,
        seq_len=64,
        num_steps=10,  # Very few steps for quick test
        temperature=1.0,
        top_k=50,
        show_progress=True,
    )

    gen_time = time.time() - start
    print(f"Generation time: {gen_time:.1f}s")

    # Decode
    text = model.decode(tokens)[0]
    print(f"\nGenerated text (untrained model, will be garbage):")
    print("-" * 40)
    # Handle Windows console encoding issues
    try:
        display_text = text[:200] + "..." if len(text) > 200 else text
        print(display_text)
    except UnicodeEncodeError:
        # Fall back to ASCII representation
        print(text[:200].encode('ascii', 'replace').decode('ascii') + "...")
    print("-" * 40)

    return True


def test_training_step(model, device):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("TRAINING STEP TEST")
    print("=" * 60)

    batch_size = 2
    seq_len = 128

    # Create test batch
    test_batch = {
        "input_ids": torch.randint(0, model.vocab_size - 1, (batch_size, seq_len), device=device)
    }

    # Enable training mode
    model.train()

    # Run one step
    print("Running single training step...")
    start = time.time()
    loss = model.training_step(test_batch, 0)
    step_time = time.time() - start

    print(f"Loss: {loss.item():.4f}")
    print(f"Step time: {step_time:.3f}s")

    if torch.cuda.is_available():
        print(f"GPU memory during training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    return loss.item()


def main():
    print("\n" + "=" * 60)
    print("SUNFISH MASKED DIFFUSION - DIAGNOSTIC TEST")
    print("=" * 60 + "\n")

    # Check CUDA
    cuda_ok = check_cuda()
    if not cuda_ok:
        print("\nWARNING: CUDA not available, training will be very slow!")

    # Load model
    try:
        model, device = test_model_loading()
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        return False

    # Test forward pass
    try:
        test_forward_pass(model, device)
    except Exception as e:
        print(f"\nERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test generation
    try:
        test_generation(model, device)
    except Exception as e:
        print(f"\nERROR in generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test training step
    try:
        test_training_step(model, device)
    except Exception as e:
        print(f"\nERROR in training step: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    print("\nKey insights:")
    print("1. The model IS loading on GPU if CUDA is available")
    print("2. Generation output will be garbage until trained")
    print("3. You need THOUSANDS of steps, not 50")
    print("4. Each 'step' with accumulate=64 is 64 forward/backward passes")
    print("\nRecommended: Train for at least 5000-10000 steps to see coherence")
    print("Use --synthetic flag for faster testing with random data")

    return True


if __name__ == "__main__":
    main()
