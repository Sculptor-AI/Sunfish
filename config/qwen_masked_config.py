"""
Qwen Masked Diffusion Configuration
Optimized for RTX 5080 (~16GB VRAM)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QwenMaskedDiffusionConfig:
    """Configuration for Qwen3-based masked diffusion language model."""

    # ============================================================================
    # Base Model
    # ============================================================================
    base_model: str = "Qwen/Qwen3-0.6B"

    # ============================================================================
    # Masking Parameters
    # ============================================================================
    timesteps: int = 1000
    mask_schedule: str = "linear"  # linear or cosine
    bidirectional: bool = True  # disable causal masking when possible

    # ============================================================================
    # Training Parameters (RTX 5080 optimized - ~16GB VRAM)
    # ============================================================================
    learning_rate: float = 5e-5      # Higher to adapt to bidirectional masking
    batch_size: int = 2              # Small due to VRAM constraints
    accumulate_grad_batches: int = 64  # Effective batch = 128 sequences
    max_steps: int = 100000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    gradient_checkpointing: bool = True  # Required for 5080

    # ============================================================================
    # A2D Warmup Protocol
    # ============================================================================
    a2d_warmup_steps: int = 2000     # Phase 1: 10% LR, gradual bidirectional
    warmup_steps: int = 10000         # Phase 2: Linear warmup to full LR

    # ============================================================================
    # Shift Operation (add in Phase 2)
    # ============================================================================
    use_shift: bool = False  # Start without, add later for Dream-style prediction

    # ============================================================================
    # Sequence Parameters
    # ============================================================================
    block_size: int = 512  # Shorter for VRAM, increase later

    # ============================================================================
    # Precision and Compute
    # ============================================================================
    precision: str = "bf16-mixed"  # RTX 5080 supports bf16
    accelerator: str = "gpu"
    devices: str = "auto"
    strategy: str = "auto"  # Use auto for single GPU

    # ============================================================================
    # Data Parameters
    # ============================================================================
    num_workers: int = 4  # Use workers for faster data loading
    use_synthetic_data: bool = False  # Use random tokens for fast testing
    dataset_name: str = "wikitext-103"  # Options: wikitext-2, wikitext-103, openwebtext, fineweb

    # ============================================================================
    # Logging and Checkpointing
    # ============================================================================
    log_every_n_steps: int = 50
    val_check_interval: int = 5000
    checkpoint_every_n_steps: int = 5000
    save_top_k: int = 3

    # ============================================================================
    # Experiment Tracking
    # ============================================================================
    project_name: str = "sunfish-masked-diffusion"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        assert self.timesteps > 0, "timesteps must be positive"
        assert self.mask_schedule in ["linear", "cosine"], "Invalid mask schedule"
        assert self.block_size > 0, "block_size must be positive"
        assert self.batch_size > 0, "batch_size must be positive"

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size with gradient accumulation."""
        return self.batch_size * self.accumulate_grad_batches

    @property
    def effective_tokens_per_batch(self) -> int:
        """Compute effective tokens per batch."""
        return self.effective_batch_size * self.block_size

    def __repr__(self):
        return (
            f"QwenMaskedDiffusionConfig(\n"
            f"  Base Model: {self.base_model}\n"
            f"  Block Size: {self.block_size}\n"
            f"  Timesteps: {self.timesteps}\n"
            f"  Mask Schedule: {self.mask_schedule}\n"
            f"  Bidirectional: {self.bidirectional}\n"
            f"  Effective Batch Size: {self.effective_batch_size} sequences\n"
            f"  Effective Tokens/Batch: {self.effective_tokens_per_batch:,}\n"
            f"  Gradient Checkpointing: {self.gradient_checkpointing}\n"
            f"  Use Shift: {self.use_shift}\n"
            f")"
        )


def get_qwen_masked_config() -> QwenMaskedDiffusionConfig:
    """Get default Qwen masked diffusion configuration."""
    return QwenMaskedDiffusionConfig()


def get_qwen_masked_config_cpu() -> QwenMaskedDiffusionConfig:
    """Get CPU-optimized configuration for testing."""
    config = QwenMaskedDiffusionConfig()
    config.accelerator = "cpu"
    config.precision = "32"
    config.batch_size = 1
    config.accumulate_grad_batches = 4
    config.block_size = 128
    config.gradient_checkpointing = False
    return config
