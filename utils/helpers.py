"""
SunFish Utility Functions
Helper functions for testing, debugging, and analysis
"""

import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from typing import Optional


def count_parameters(model, non_embedding: bool = False):
    """
    Count model parameters.

    Args:
        model: PyTorch model
        non_embedding: If True, exclude embedding parameters

    Returns:
        Number of parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if non_embedding and hasattr(model, "token_embedding"):
        total -= model.token_embedding.weight.numel()
        if hasattr(model, "pos_embedding"):
            total -= model.pos_embedding.numel()

    return total


def check_data_pipeline(datamodule, num_batches: int = 10):
    """
    Test data pipeline speed and correctness.

    Args:
        datamodule: PyTorch Lightning DataModule
        num_batches: Number of batches to test
    """
    print("\n🔍 Testing data pipeline...")

    datamodule.setup()
    loader = datamodule.train_dataloader()

    start = time.time()
    total_samples = 0

    for i, batch in enumerate(tqdm(loader, total=num_batches, desc="Loading batches")):
        if i >= num_batches:
            break

        # Validate batch
        assert batch.dtype == torch.long, f"Expected dtype long, got {batch.dtype}"
        assert len(batch.shape) == 2, f"Expected 2D batch, got {batch.shape}"

        total_samples += batch.shape[0]

    elapsed = time.time() - start
    throughput = total_samples / elapsed

    print(f"\n✅ Data pipeline test complete!")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Batch shape: {batch.shape}")
    print(f"  Token range: [{batch.min()}, {batch.max()}]")


def overfit_single_batch(
    model,
    batch: torch.Tensor,
    num_steps: int = 100,
    lr: float = 1e-3,
    target_loss: float = 0.01,
):
    """
    Sanity check: overfit a single batch (Section 8).

    The model should be able to perfectly memorize a single batch.
    If loss doesn't go near zero, something is wrong with the model.

    Args:
        model: SunFish model
        batch: Single batch of token IDs
        num_steps: Number of optimization steps
        lr: Learning rate
        target_loss: Target loss (should reach this or lower)

    Returns:
        List of losses
    """
    print("\n🧪 Single batch overfitting test...")
    print(f"  Target: Loss < {target_loss}")

    device = next(model.parameters()).device
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    pbar = tqdm(range(num_steps), desc="Overfitting")

    for step in pbar:
        # Forward pass
        loss = model.training_step(batch.to(device), 0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Early stopping if target reached
        if loss.item() < target_loss:
            print(f"\n✅ Reached target loss at step {step}")
            break

    final_loss = losses[-1]
    print(f"\nFinal loss: {final_loss:.6f}")

    if final_loss < target_loss:
        print("✅ PASS: Model can overfit (implementation correct)")
    else:
        print(f"❌ FAIL: Model cannot overfit (expected < {target_loss}, got {final_loss})")

    return losses


def test_forward_pass(model, batch_size: int = 2, seq_len: int = 32):
    """
    Test model forward pass.

    Args:
        model: SunFish model
        batch_size: Batch size
        seq_len: Sequence length
    """
    print("\n🔍 Testing forward pass...")

    device = next(model.parameters()).device
    model.eval()

    # Create dummy input
    token_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        # Get embeddings
        x_0 = model.token_embedding(token_ids) + model.pos_embedding[:, :seq_len, :]

        # Sample timestep
        t = torch.randint(0, model.config.timesteps, (batch_size,), device=device)

        # Add noise
        noise = torch.randn_like(x_0)
        x_t = model.q_sample(x_0, t, noise)

        # Predict noise
        pred_noise = model(x_t, t)

        # Compute loss
        loss = F.mse_loss(noise, pred_noise)

    print(f"✅ Forward pass successful!")
    print(f"  Input shape: {x_t.shape}")
    print(f"  Output shape: {pred_noise.shape}")
    print(f"  Loss: {loss.item():.4f}")

    return loss.item()


def analyze_diffusion_schedule(model):
    """
    Analyze the diffusion schedule.

    Args:
        model: SunFish model
    """
    print("\n📊 Diffusion Schedule Analysis")
    print("=" * 60)

    T = model.config.timesteps

    print(f"\nSchedule Parameters:")
    print(f"  Total timesteps: {T}")
    print(f"  Beta range: [{model.config.beta_start}, {model.config.beta_end}]")

    # Analyze key timesteps
    key_timesteps = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    print(f"\nKey Timestep Values:")
    print(f"  {'t':>5} | {'beta':>8} | {'alpha':>8} | {'alpha_bar':>10} | {'SNR':>8}")
    print(f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 8}")

    for t in key_timesteps:
        beta = model.betas[t].item()
        alpha = model.alphas[t].item()
        alpha_bar = model.alphas_cumprod[t].item()
        snr = alpha_bar / (1 - alpha_bar)

        print(f"  {t:5d} | {beta:8.6f} | {alpha:8.6f} | {alpha_bar:10.8f} | {snr:8.4f}")

    print("=" * 60)


def memory_usage_summary():
    """Print GPU memory usage summary."""
    if torch.cuda.is_available():
        print("\n💾 GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9

            print(f"  GPU {i}:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved:  {reserved:.2f} GB")
            print(f"    Total:     {total:.2f} GB")
            print(f"    Usage:     {allocated / total * 100:.1f}%")
    else:
        print("\n💾 No GPU available")


def gradient_stats(model):
    """
    Compute gradient statistics.

    Useful for debugging training instabilities.
    """
    total_norm = 0.0
    max_grad = 0.0
    min_grad = float("inf")
    num_params = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_grad = max(max_grad, p.grad.abs().max().item())
            min_grad = min(min_grad, p.grad.abs().min().item())
            num_params += 1

    total_norm = total_norm**0.5

    return {
        "total_norm": total_norm,
        "max_grad": max_grad,
        "min_grad": min_grad,
        "num_params_with_grad": num_params,
    }


if __name__ == "__main__":
    print("✅ Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("  - count_parameters(): Count model parameters")
    print("  - check_data_pipeline(): Test data loading")
    print("  - overfit_single_batch(): Sanity check model can learn")
    print("  - test_forward_pass(): Test model forward pass")
    print("  - analyze_diffusion_schedule(): Analyze noise schedule")
    print("  - memory_usage_summary(): Check GPU memory")
    print("  - gradient_stats(): Compute gradient statistics")
