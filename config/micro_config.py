"""
Micro SunFish Configuration - Sweet Spot for Coherent Generation
~100M parameters - small enough for CPU, large enough to learn language
"""

from dataclasses import dataclass
from config.model_config import SunFishConfig


@dataclass
class MicroSunFishConfig(SunFishConfig):
    """
    Micro configuration for actual text generation on CPU.

    ~100M parameters - the sweet spot:
    - Small enough to train on CPU in hours
    - Large enough to generate coherent words/phrases
    - Uses real text data (not synthetic)
    """

    # ============================================================================
    # Micro Model Architecture (~100M params)
    # ============================================================================
    vocab_size: int = 8192  # GPT-2 tokenizer subset
    n_layer: int = 8  # 8 transformer layers
    n_head: int = 8  # 8 attention heads
    n_embd: int = 512  # 512 embedding dimension
    intermediate_size: int = 2048  # 4x n_embd
    block_size: int = 256  # Medium sequences
    dropout: float = 0.1  # Some dropout for regularization

    # ============================================================================
    # Diffusion Parameters (fewer steps for faster training)
    # ============================================================================
    timesteps: int = 500  # Reduced from 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # ============================================================================
    # Classifier-Free Guidance (CFG) Parameters
    # ============================================================================
    conditioning_dropout: float = 0.1  # Standard dropout for medium model
    guidance_scale: float = 7.0  # Good balance for 100M model
    prompt_max_length: int = 64  # Reasonable prompt length

    # ============================================================================
    # Aggressive Training for Fast Convergence
    # ============================================================================
    batch_size: int = 4  # Small batches for CPU
    accumulate_grad_batches: int = 8  # Effective batch = 32
    learning_rate: float = 5e-4  # Higher LR for faster learning
    weight_decay: float = 0.01
    max_steps: int = 10000  # 10K steps is enough for CPU training
    warmup_steps: int = 500

    # ============================================================================
    # CPU-Optimized Settings
    # ============================================================================
    precision: str = "32"
    strategy: str = "auto"
    gradient_clip_val: float = 1.0
    devices: int = 1
    accelerator: str = "cpu"

    # ============================================================================
    # Frequent Logging (monitor progress)
    # ============================================================================
    log_every_n_steps: int = 50
    val_check_interval: int = 500
    checkpoint_every_n_steps: int = 1000
    save_top_k: int = 3

    # ============================================================================
    # Data
    # ============================================================================
    tokenizer_name: str = "gpt2"
    num_workers: int = 2  # Some parallelism

    # ============================================================================
    # Experiment
    # ============================================================================
    project_name: str = "sunfish-micro"
    experiment_name: str = "100m-coherent-text"


def get_micro_config():
    """Factory function to get micro config."""
    config = MicroSunFishConfig()
    print("=" * 60)
    print("🐟 MICRO SUNFISH - 100M PARAMETER MODEL")
    print("=" * 60)
    print(f"Model Size: ~{config.model_size:.3f}B parameters ({config.model_size * 1000:.1f}M)")
    print(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"Sequence Length: {config.block_size}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Diffusion Steps: {config.timesteps}")
    print(f"Batch Size: {config.batch_size} (effective: {config.batch_size * config.accumulate_grad_batches})")
    print(f"Max Training Steps: {config.max_steps:,}")
    print(f"Device: {config.accelerator.upper()}")
    print(f"\n🎯 Goal: Generate coherent words after training!")
    print("=" * 60)
    return config


if __name__ == "__main__":
    config = get_micro_config()

    # Calculate actual params
    from models import SunFishTransformer
    model = SunFishTransformer(config)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Actual parameters: {actual_params:,} ({actual_params/1e6:.1f}M)")
