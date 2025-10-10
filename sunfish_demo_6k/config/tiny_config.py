"""
Tiny SunFish Configuration for CPU Testing
Ultra-small model for validating pipeline on CPU
"""

from dataclasses import dataclass
from config.model_config import SunFishConfig


@dataclass
class TinySunFishConfig(SunFishConfig):
    """
    Tiny configuration for CPU-based development and testing.

    This config creates a model with ~500K parameters that can run on CPU
    to validate the entire pipeline before GPU training.
    """

    # ============================================================================
    # Tiny Model Architecture
    # ============================================================================
    vocab_size: int = 1024  # Reduced vocab
    n_layer: int = 2  # Just 2 layers
    n_head: int = 2  # 2 attention heads
    n_embd: int = 128  # Small embedding dimension
    intermediate_size: int = 512  # 4x n_embd
    block_size: int = 128  # Short sequences
    dropout: float = 0.0  # No dropout for testing

    # ============================================================================
    # Tiny Diffusion Parameters
    # ============================================================================
    timesteps: int = 100  # Fewer diffusion steps
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # ============================================================================
    # Classifier-Free Guidance (CFG) Parameters
    # ============================================================================
    conditioning_dropout: float = 0.15  # Higher dropout for small model
    guidance_scale: float = 5.0  # Lower guidance for small model
    prompt_max_length: int = 32  # Shorter prompts for small model

    # ============================================================================
    # Tiny Training Parameters
    # ============================================================================
    batch_size: int = 2  # Very small batches
    accumulate_grad_batches: int = 2  # Effective batch size = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000  # Just 1000 steps for testing
    warmup_steps: int = 100

    # ============================================================================
    # CPU-specific Settings
    # ============================================================================
    precision: str = "32"  # Full precision on CPU
    strategy: str = "auto"  # No distributed training
    gradient_clip_val: float = 1.0
    devices: int = 1
    accelerator: str = "cpu"  # CPU only

    # ============================================================================
    # Fast Logging for Testing
    # ============================================================================
    log_every_n_steps: int = 10
    val_check_interval: int = 100
    checkpoint_every_n_steps: int = 500
    save_top_k: int = 1

    # ============================================================================
    # Data
    # ============================================================================
    num_workers: int = 0  # No multiprocessing on CPU

    # ============================================================================
    # Experiment
    # ============================================================================
    project_name: str = "sunfish-tiny-test"
    experiment_name: str = "cpu-validation"


def get_tiny_config():
    """Factory function to get tiny config."""
    config = TinySunFishConfig()
    print("=" * 60)
    print("🐟 TINY SUNFISH CONFIG FOR CPU TESTING")
    print("=" * 60)
    print(f"Model Size: ~{config.model_size:.3f}B parameters ({config.model_size * 1000:.1f}M)")
    print(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"Sequence Length: {config.block_size}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Diffusion Steps: {config.timesteps}")
    print(f"Batch Size: {config.batch_size} (effective: {config.batch_size * config.accumulate_grad_batches})")
    print(f"Max Training Steps: {config.max_steps}")
    print(f"Device: {config.accelerator.upper()}")
    print("=" * 60)
    return config


if __name__ == "__main__":
    config = get_tiny_config()
