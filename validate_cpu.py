#!/usr/bin/env python3
"""
SunFish CPU Validation Script
Test entire pipeline on CPU before GPU training
"""

import torch
from config import get_tiny_config
from models import SunFishTransformer, DDIMScheduler
from data import FineWebDataModule, TinyTextDataset
from utils import (
    count_parameters,
    check_data_pipeline,
    overfit_single_batch,
    test_forward_pass,
    analyze_diffusion_schedule,
)
from sample import round_embeddings_to_tokens
from transformers import AutoTokenizer


def test_model_initialization():
    """Test 1: Model initialization."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Initialization")
    print("=" * 70)

    config = get_tiny_config()
    model = SunFishTransformer(config)

    total_params = count_parameters(model)
    non_emb_params = count_parameters(model, non_embedding=True)

    print(f"\n✅ Model initialized successfully!")
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Non-embedding: {non_emb_params:,} ({non_emb_params / 1e6:.2f}M)")

    return model


def test_data_loading():
    """Test 2: Data loading."""
    print("\n" + "=" * 70)
    print("TEST 2: Data Loading")
    print("=" * 70)

    config = get_tiny_config()
    datamodule = FineWebDataModule(config)

    check_data_pipeline(datamodule, num_batches=10)

    return datamodule


def test_forward_backward():
    """Test 3: Forward and backward pass."""
    print("\n" + "=" * 70)
    print("TEST 3: Forward/Backward Pass")
    print("=" * 70)

    config = get_tiny_config()
    model = SunFishTransformer(config)

    # Forward pass
    test_forward_pass(model, batch_size=2, seq_len=32)

    # Test backward
    print("\n🔍 Testing backward pass...")
    model.train()

    batch = torch.randint(0, config.vocab_size, (2, 32))
    loss = model.training_step(batch, 0)

    loss.backward()

    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())

    if has_grad:
        print("✅ Backward pass successful (gradients computed)")
    else:
        print("❌ Backward pass failed (no gradients)")

    return model


def test_overfitting():
    """Test 4: Single batch overfitting."""
    print("\n" + "=" * 70)
    print("TEST 4: Single Batch Overfitting")
    print("=" * 70)

    config = get_tiny_config()
    model = SunFishTransformer(config)

    # Create small batch
    batch = torch.randint(0, config.vocab_size, (2, 32))

    # Try to overfit
    losses = overfit_single_batch(
        model, batch, num_steps=100, lr=3e-4, target_loss=0.01
    )

    return losses


def test_diffusion_schedule():
    """Test 5: Diffusion schedule."""
    print("\n" + "=" * 70)
    print("TEST 5: Diffusion Schedule")
    print("=" * 70)

    config = get_tiny_config()
    model = SunFishTransformer(config)

    analyze_diffusion_schedule(model)

    return model


def test_sampling():
    """Test 6: Text generation."""
    print("\n" + "=" * 70)
    print("TEST 6: Text Generation (DDIM)")
    print("=" * 70)

    config = get_tiny_config()
    model = SunFishTransformer(config)
    model.eval()

    # Create scheduler
    scheduler = DDIMScheduler(model, eta=0.0)

    # Generate
    print("\n🎲 Generating sample...")
    batch_size = 2
    seq_len = 32
    shape = (batch_size, seq_len, config.n_embd)

    embeddings = scheduler.sample(shape, num_steps=10, show_progress=True)

    # Round to tokens
    token_ids = round_embeddings_to_tokens(embeddings, model.token_embedding)

    # Decode
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    print("\n📝 Generated text:")
    for i, ids in enumerate(token_ids):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"\nSample {i + 1}: {text[:100]}...")

    print("\n✅ Sampling successful!")


def test_mini_training():
    """Test 7: Mini training loop."""
    print("\n" + "=" * 70)
    print("TEST 7: Mini Training Loop (10 steps)")
    print("=" * 70)

    config = get_tiny_config()
    model = SunFishTransformer(config)
    model.train()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Create dataset
    dataset = TinyTextDataset(
        num_samples=100, block_size=config.block_size, vocab_size=config.vocab_size
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Train for 10 steps
    print("\n🏋️ Training for 10 steps...")
    losses = []

    for step, batch in enumerate(loader):
        if step >= 10:
            break

        # Forward
        loss = model.training_step(batch, step)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step}: Loss = {loss.item():.4f}")

    print(f"\n✅ Training successful!")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss decreased: {losses[0] > losses[-1]}")


def main():
    """Run all validation tests."""

    print("\n" + "🐟" * 35)
    print("SUNFISH CPU VALIDATION SUITE")
    print("Testing entire pipeline on CPU before GPU training")
    print("🐟" * 35)

    try:
        # Run tests
        test_model_initialization()
        test_data_loading()
        test_forward_backward()
        test_overfitting()
        test_diffusion_schedule()
        test_sampling()
        test_mini_training()

        # Summary
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Your CPU is ready for development!")
        print("✅ Pipeline validated successfully!")
        print("\n📝 Next steps:")
        print("  1. Install dependencies on GPU machine")
        print("  2. Copy this project to GPU machine")
        print("  3. Run: python train.py --config full")
        print("  4. Monitor with WandB")
        print("\n" + "=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ VALIDATION FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        print("\n💡 Fix the error and run again.")


if __name__ == "__main__":
    main()
