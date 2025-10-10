"""
Utility Functions for Sunfish Training
Includes testing, debugging, and monitoring helpers
"""

import torch
import time
import logging
from tqdm import tqdm
from typing import Optional
import psutil
import GPUtil

logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module):
    """Print detailed model summary."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable = total_params - trainable_params

    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print(f"Non-trainable params:   {non_trainable:,}")
    print(f"Model size (FP32):      {total_params * 4 / 1e9:.2f} GB")
    print(f"Model size (FP16/BF16): {total_params * 2 / 1e9:.2f} GB")
    print("=" * 80 + "\n")


def check_data_pipeline(datamodule, num_batches: int = 10):
    """
    Test data pipeline speed and correctness.

    Args:
        datamodule: PyTorch Lightning DataModule
        num_batches: Number of batches to test
    """
    logger.info(f"Testing data pipeline with {num_batches} batches...")

    datamodule.setup()
    loader = datamodule.train_dataloader()

    start_time = time.time()
    total_tokens = 0

    for i, batch in enumerate(tqdm(loader, total=num_batches, desc="Loading batches")):
        if i >= num_batches:
            break

        # Validate batch shape
        assert batch.ndim == 2, f"Expected 2D batch, got {batch.ndim}D"
        assert batch.shape[1] == datamodule.config.block_size, (
            f"Expected seq_len={datamodule.config.block_size}, got {batch.shape[1]}"
        )

        total_tokens += batch.numel()

        if i < 3:
            logger.info(
                f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}, "
                f"min={batch.min().item()}, max={batch.max().item()}"
            )

    elapsed = time.time() - start_time
    sequences_per_sec = (num_batches * datamodule.config.batch_size) / elapsed
    tokens_per_sec = total_tokens / elapsed

    print("\n" + "=" * 80)
    print("Data Pipeline Test Results")
    print("=" * 80)
    print(f"  Batches loaded:     {num_batches}")
    print(f"  Time:               {elapsed:.2f}s")
    print(f"  Sequences/sec:      {sequences_per_sec:.2f}")
    print(f"  Tokens/sec:         {tokens_per_sec:,.0f}")
    print(f"  Batch size:         {datamodule.config.batch_size}")
    print(f"  Sequence length:    {datamodule.config.block_size}")
    print("=" * 80 + "\n")


def overfit_single_batch_test(
    model,
    batch,
    steps: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    Sanity check: overfit on a single batch.

    This tests that:
    1. The model can learn (forward/backward work)
    2. Loss decreases (gradients are correct)
    3. Model can memorize (capacity is sufficient)

    Args:
        model: PyTorch model with training_step method
        batch: Single batch of data
        steps: Number of training steps
        lr: Learning rate
        device: Device to use
    """
    logger.info(f"Running overfit test on single batch for {steps} steps...")

    model = model.to(device)
    batch = batch.to(device)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for step in tqdm(range(steps), desc="Overfitting"):
        optimizer.zero_grad()

        # Forward pass
        loss = model.training_step(batch, step)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Record loss
        loss_val = loss.item()
        losses.append(loss_val)

        # Log every 10 steps
        if step % 10 == 0:
            logger.info(f"Step {step:3d}: Loss = {loss_val:.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print("\n" + "=" * 80)
    print("Overfit Test Results")
    print("=" * 80)
    print(f"  Initial loss:  {initial_loss:.6f}")
    print(f"  Final loss:    {final_loss:.6f}")
    print(f"  Reduction:     {reduction:.1f}%")
    print(f"  Status:        {'PASS ✓' if final_loss < 0.1 else 'FAIL ✗'}")
    print("=" * 80 + "\n")

    if final_loss > 0.1:
        logger.warning(
            "Overfit test failed! Model should reach loss < 0.1 when overfitting. "
            "Check model architecture and training loop."
        )

    return losses


def monitor_gpu_memory():
    """Monitor GPU memory usage."""
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(
                f"GPU {gpu.id}: {gpu.name} - "
                f"Memory: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB "
                f"({gpu.memoryUtil * 100:.1f}%), "
                f"Load: {gpu.load * 100:.0f}%"
            )
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")


def monitor_system_resources():
    """Monitor CPU and RAM usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(
        f"System Resources - CPU: {cpu_percent:.1f}%, "
        f"RAM: {memory.used / 1e9:.1f}/{memory.total / 1e9:.1f} GB "
        f"({memory.percent:.1f}%)"
    )


def estimate_training_time(
    total_steps: int,
    steps_completed: int,
    time_elapsed: float,
):
    """
    Estimate remaining training time.

    Args:
        total_steps: Total training steps
        steps_completed: Steps completed so far
        time_elapsed: Time elapsed (seconds)

    Returns:
        Formatted time estimate string
    """
    if steps_completed == 0:
        return "Unknown"

    avg_time_per_step = time_elapsed / steps_completed
    remaining_steps = total_steps - steps_completed
    remaining_time = avg_time_per_step * remaining_steps

    # Format time
    hours = int(remaining_time // 3600)
    minutes = int((remaining_time % 3600) // 60)
    seconds = int(remaining_time % 60)

    return f"{hours}h {minutes}m {seconds}s"


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Note: For diffusion models, this isn't directly applicable since we're
    not doing next-token prediction. This is here for reference.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """Check if tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        logger.error(f"{name} contains NaN values!")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"{name} contains Inf values!")
        return False
    return True


def gradient_stats(model: torch.nn.Module):
    """
    Compute gradient statistics for debugging.

    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    num_params = 0
    min_grad = float("inf")
    max_grad = float("-inf")

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm**2
            num_params += 1

            min_grad = min(min_grad, p.grad.data.min().item())
            max_grad = max(max_grad, p.grad.data.max().item())

    total_norm = total_norm**0.5

    return {
        "grad_norm": total_norm,
        "num_params_with_grad": num_params,
        "min_grad": min_grad,
        "max_grad": max_grad,
    }


def save_generated_samples(samples: list, output_path: str):
    """
    Save generated text samples to file.

    Args:
        samples: List of generated text strings
        output_path: Path to output file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            f.write(f"=== Sample {i + 1} ===\n")
            f.write(sample)
            f.write("\n\n")
    logger.info(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    # Test utilities
    import sys

    sys.path.append("..")
    from config.model_config import get_config_tiny
    from models.diffusion_model import DiffusionTransformer

    logging.basicConfig(level=logging.INFO)

    # Test model summary
    config = get_config_tiny()
    model = DiffusionTransformer(config)
    print_model_summary(model)

    # Test system monitoring
    monitor_system_resources()
    monitor_gpu_memory()
