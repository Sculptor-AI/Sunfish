"""
SunFish Diffusion LLM Configuration
Based on technical report Section 6-8
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SunFishConfig:
    """Configuration for SunFish diffusion language model."""

    # ============================================================================
    # Model Architecture (Section 6)
    # ============================================================================
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 2048
    intermediate_size: int = 8192  # FFN hidden size (4x n_embd)
    block_size: int = 2048  # Maximum sequence length
    dropout: float = 0.1

    # ============================================================================
    # Diffusion Parameters (Section 2)
    # ============================================================================
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = "linear"  # linear, cosine, or quadratic

    # ============================================================================
    # Classifier-Free Guidance (CFG) Parameters
    # ============================================================================
    conditioning_dropout: float = 0.1  # Probability of dropping prompt during training
    guidance_scale: float = 7.5  # CFG guidance scale for sampling (1.0 = no guidance)
    prompt_max_length: int = 77  # Maximum prompt length in tokens

    # ============================================================================
    # Training Parameters (Section 7-8)
    # ============================================================================
    batch_size: int = 8
    accumulate_grad_batches: int = 16  # Effective batch size = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 500000
    warmup_steps: int = 5000

    # ============================================================================
    # Data Parameters
    # ============================================================================
    tokenizer_name: str = "gpt2"
    num_workers: int = 4

    # ============================================================================
    # Logging and Checkpointing
    # ============================================================================
    log_every_n_steps: int = 50
    val_check_interval: int = 5000
    checkpoint_every_n_steps: int = 5000
    save_top_k: int = 3

    # ============================================================================
    # Compute Configuration
    # ============================================================================
    precision: str = "bf16-mixed"  # bf16-mixed, 16-mixed, or 32
    strategy: str = "fsdp"  # fsdp, ddp, or auto
    gradient_clip_val: float = 1.0
    devices: str = "auto"  # Number of GPUs or "auto"
    accelerator: str = "gpu"  # gpu or cpu

    # ============================================================================
    # Experiment Tracking
    # ============================================================================
    project_name: str = "sunfish-diffusion-llm"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.intermediate_size >= self.n_embd, "intermediate_size should be >= n_embd"
        assert self.timesteps > 0, "timesteps must be positive"
        assert 0 < self.beta_start < self.beta_end < 1, "Invalid beta schedule"

    @property
    def model_size(self):
        """Estimate model size in billions of parameters."""
        # Rough estimate: 12 * n_layer * n_embd^2 for transformer
        # Plus embedding tables
        params = 12 * self.n_layer * (self.n_embd ** 2)
        params += self.vocab_size * self.n_embd * 2  # input + output embeddings
        return params / 1e9

    def __repr__(self):
        return (
            f"SunFishConfig(\n"
            f"  Model Size: ~{self.model_size:.2f}B parameters\n"
            f"  Layers: {self.n_layer}, Heads: {self.n_head}, Dim: {self.n_embd}\n"
            f"  Block Size: {self.block_size}, Vocab: {self.vocab_size}\n"
            f"  Diffusion Steps: {self.timesteps}\n"
            f"  Effective Batch Size: {self.batch_size * self.accumulate_grad_batches}\n"
            f")"
        )
