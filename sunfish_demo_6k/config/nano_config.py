"""
Nano SunFish - ULTRA TINY for fast CPU validation
~5M parameters - shows results in 30 minutes!
"""

from dataclasses import dataclass
from config.model_config import SunFishConfig


@dataclass
class NanoSunFishConfig(SunFishConfig):
    """
    Nano config - smallest possible model that still learns.

    ~5M parameters - can train and show results in 30 minutes on CPU.
    Goal: Validate the entire pipeline works end-to-end.
    """

    # ============================================================================
    # ULTRA Tiny Architecture (~5M params)
    # ============================================================================
    vocab_size: int = 2048  # Very small vocab
    n_layer: int = 4  # Just 4 layers
    n_head: int = 4  # 4 attention heads
    n_embd: int = 256  # Small embedding
    intermediate_size: int = 1024  # 4x n_embd
    block_size: int = 64  # Very short sequences
    dropout: float = 0.0

    # ============================================================================
    # Fast Diffusion
    # ============================================================================
    timesteps: int = 100  # Fewer steps
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # ============================================================================
    # Aggressive Training for FAST results
    # ============================================================================
    batch_size: int = 8  # Larger batches
    accumulate_grad_batches: int = 1  # No accumulation
    learning_rate: float = 1e-3  # High LR for fast learning
    weight_decay: float = 0.01
    max_steps: int = 500  # Just 500 steps!
    warmup_steps: int = 50

    # ============================================================================
    # CPU Settings
    # ============================================================================
    precision: str = "32"
    strategy: str = "auto"
    gradient_clip_val: float = 1.0
    devices: int = 1
    accelerator: str = "cpu"

    # ============================================================================
    # Frequent Checkpointing
    # ============================================================================
    log_every_n_steps: int = 10
    val_check_interval: int = 50
    checkpoint_every_n_steps: int = 100  # Checkpoint every 100 steps
    save_top_k: int = 2

    # ============================================================================
    # Data
    # ============================================================================
    num_workers: int = 0
    tokenizer_name: str = "gpt2"

    # ============================================================================
    # Experiment
    # ============================================================================
    project_name: str = "sunfish-nano-validation"
    experiment_name: str = "5m-params-fast-test"


def get_nano_config():
    """Get nano config with info display."""
    config = NanoSunFishConfig()
    print("=" * 60)
    print("🐟 NANO SUNFISH - ULTRA FAST CPU VALIDATION")
    print("=" * 60)
    print(f"Model Size: ~{config.model_size:.3f}B ({config.model_size * 1000:.1f}M)")
    print(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"Sequence Length: {config.block_size}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Max Steps: {config.max_steps} (expect ~30 min on CPU)")
    print(f"Checkpoints every: {config.checkpoint_every_n_steps} steps")
    print("=" * 60)
    return config


if __name__ == "__main__":
    from models import SunFishTransformer
    config = get_nano_config()
    model = SunFishTransformer(config)
    actual = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Actual parameters: {actual:,} ({actual/1e6:.1f}M)")
