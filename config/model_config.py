"""
Configuration for Sunfish Diffusion LLM (1.4B parameters)
Based on RND1 research and optimized for A6000 + A4500 GPU setup
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DiffusionConfig:
    """Complete configuration for diffusion language model training and inference."""

    # ============================================================================
    # Model Architecture (targeting ~1.4B parameters)
    # ============================================================================
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size (will be updated from tokenizer)
    n_layer: int = 24  # Number of transformer layers
    n_head: int = 16  # Number of attention heads
    n_embd: int = 2048  # Embedding dimension (d_model)
    intermediate_size: int = 8192  # FFN intermediate dimension (4x n_embd)
    block_size: int = 2048  # Maximum sequence length / context window
    dropout: float = 0.1  # Dropout probability
    attention_dropout: float = 0.1  # Attention dropout
    residual_dropout: float = 0.1  # Residual connection dropout

    # Layer normalization
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = True  # Whether to use bias in linear layers

    # ============================================================================
    # Diffusion Process Hyperparameters (following DDPM)
    # ============================================================================
    timesteps: int = 1000  # Number of diffusion timesteps (T)
    beta_start: float = 0.0001  # Start of beta schedule
    beta_end: float = 0.02  # End of beta schedule
    beta_schedule: Literal["linear", "cosine", "sqrt"] = "linear"  # Noise schedule type

    # Noise prediction target
    prediction_type: Literal["epsilon", "v_prediction", "sample"] = "epsilon"  # What model predicts

    # ============================================================================
    # Training Hyperparameters (optimized for limited hardware)
    # ============================================================================
    # Batch configuration
    batch_size: int = 8  # Per-GPU batch size (start conservative for memory)
    accumulate_grad_batches: int = 16  # Gradient accumulation steps
    # Effective global batch = batch_size × num_gpus × accumulate_grad_batches
    # = 8 × 2 × 16 = 256 sequences = 524,288 tokens per optimization step

    # Optimization
    learning_rate: float = 1e-4  # Peak learning rate
    min_lr: float = 1e-6  # Minimum learning rate for cosine decay
    weight_decay: float = 0.01  # AdamW weight decay
    adam_beta1: float = 0.9  # Adam beta1
    adam_beta2: float = 0.95  # Adam beta2
    adam_epsilon: float = 1e-8  # Adam epsilon
    max_grad_norm: float = 1.0  # Gradient clipping threshold

    # Learning rate schedule
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = 10000  # LR warmup steps (~2% of total)
    max_steps: int = 500000  # Total training steps

    # RND1-specific: Layer-wise learning rates for AR initialization
    use_layer_wise_lr: bool = False  # Enable different LR for attention vs FFN
    attention_lr_multiplier: float = 1.0  # Multiplier for attention layer LR
    ffn_lr_multiplier: float = 0.2  # Multiplier for FFN layer LR (preserve knowledge)

    # ============================================================================
    # Data Configuration
    # ============================================================================
    tokenizer_name: str = "gpt2"  # HuggingFace tokenizer identifier
    dataset_name: str = "HuggingFaceFW/fineweb"  # FineWeb dataset
    dataset_config: Optional[str] = "default"  # Dataset configuration/subset
    dataset_split: str = "train"  # Dataset split to use
    stream_dataset: bool = True  # Use streaming mode (essential for large datasets)

    # Data preprocessing
    num_workers: int = 4  # DataLoader worker processes
    prefetch_factor: int = 2  # Batches to prefetch per worker
    pin_memory: bool = True  # Pin memory for faster GPU transfer

    # ============================================================================
    # Distributed Training & Hardware Optimization
    # ============================================================================
    # Precision
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "bf16-mixed"  # Use bfloat16 mixed precision

    # Parallelism strategy
    strategy: Literal["ddp", "fsdp", "deepspeed"] = "fsdp"  # Fully Sharded Data Parallel

    # FSDP configuration
    fsdp_sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"] = "FULL_SHARD"
    fsdp_cpu_offload: bool = False  # Offload to CPU (slower but saves VRAM)
    fsdp_activation_checkpointing: bool = False  # Recompute activations (saves memory, slower)

    # Compilation (PyTorch 2.0+)
    compile_model: bool = False  # Use torch.compile (can cause issues with some ops)

    # ============================================================================
    # Logging & Checkpointing
    # ============================================================================
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "sunfish-diffusion-llm"
    wandb_entity: Optional[str] = None  # Your W&B username/team
    wandb_run_name: Optional[str] = None  # Auto-generated if None

    # Logging frequency
    log_every_n_steps: int = 50  # Log metrics every N steps
    val_check_interval: int = 5000  # Validation/sampling interval

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 5000  # Save checkpoint frequency
    save_top_k: int = 3  # Keep best K checkpoints
    save_last: bool = True  # Always save latest checkpoint

    # ============================================================================
    # Inference & Sampling
    # ============================================================================
    # Sampling configuration
    default_num_inference_steps: int = 50  # DDIM steps for fast sampling
    default_eta: float = 0.0  # DDIM eta (0=deterministic, 1=stochastic like DDPM)
    guidance_scale: float = 1.0  # Classifier-free guidance scale (1=no guidance)

    # Generation
    max_gen_length: int = 512  # Maximum generation length
    temperature: float = 1.0  # Sampling temperature (applied after denoising)
    top_k: int = 0  # Top-k filtering (0=disabled)
    top_p: float = 1.0  # Nucleus sampling threshold

    # ============================================================================
    # Experimental / Advanced Features
    # ============================================================================
    # Curriculum learning
    use_curriculum: bool = False  # Gradually increase sequence length during training
    curriculum_start_length: int = 512  # Starting sequence length
    curriculum_end_step: int = 100000  # Step when curriculum ends

    # Mixed masking strategies (inspired by LLaDA)
    use_variable_masking: bool = False  # Random mask ratio per batch
    min_mask_ratio: float = 0.15  # Minimum masking ratio
    max_mask_ratio: float = 0.85  # Maximum masking ratio

    # Debugging
    detect_anomaly: bool = False  # Enable PyTorch anomaly detection (slow!)
    fast_dev_run: bool = False  # Run 1 batch for quick testing
    overfit_batches: int = 0  # Overfit on N batches for debugging

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Validation and derived parameters."""
        # Calculate effective batch size
        self.effective_batch_size = self.batch_size * self.accumulate_grad_batches

        # Validate model dimensions
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        # Validate paths exist
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def get_parameter_count(self) -> int:
        """
        Estimate total parameter count.

        Formula for transformer with tied embeddings:
        - Token embeddings: vocab_size × n_embd
        - Position embeddings: block_size × n_embd
        - N layers × (
            - Multi-head attention: 4 × n_embd² (Q, K, V, O projections)
            - FFN: 2 × n_embd × intermediate_size
            - Layer norms: 4 × n_embd (2 per layer)
          )
        - Final layer norm: n_embd
        - Output projection: n_embd² (if not tied)
        """
        # Embeddings
        token_emb = self.vocab_size * self.n_embd
        pos_emb = self.block_size * self.n_embd

        # Per-layer parameters
        attn_params = 4 * self.n_embd * self.n_embd  # Q, K, V, O
        ffn_params = 2 * self.n_embd * self.intermediate_size  # Up and down projection
        ln_params = 4 * self.n_embd  # 2 layer norms × 2 params (weight, bias)

        layer_params = self.n_layer * (attn_params + ffn_params + ln_params)

        # Output
        final_ln = self.n_embd
        output_proj = self.n_embd * self.n_embd  # Noise prediction head

        total = token_emb + pos_emb + layer_params + final_ln + output_proj

        return total

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        params = self.get_parameter_count()
        effective_batch = self.batch_size * self.accumulate_grad_batches * 2  # Assuming 2 GPUs
        effective_tokens = effective_batch * self.block_size

        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Sunfish Diffusion LLM Configuration                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model Architecture:                                                          ║
║   • Parameters: {params:,} (~{params/1e9:.2f}B)                              ║
║   • Layers: {self.n_layer}  |  Heads: {self.n_head}  |  Dim: {self.n_embd}  ║
║   • Context Length: {self.block_size} tokens                                 ║
║   • Vocab Size: {self.vocab_size:,}                                          ║
║                                                                              ║
║ Training Configuration:                                                      ║
║   • Total Steps: {self.max_steps:,}                                          ║
║   • Batch Size: {self.batch_size} × 2 GPUs × {self.accumulate_grad_batches} accum = {effective_batch} seq  ║
║   • Effective Tokens/Step: {effective_tokens:,} (~{effective_tokens/1e3:.0f}K)  ║
║   • Learning Rate: {self.learning_rate} (warmup: {self.warmup_steps} steps) ║
║   • Precision: {self.precision}  |  Strategy: {self.strategy}                ║
║                                                                              ║
║ Diffusion Process:                                                           ║
║   • Timesteps: {self.timesteps}                                              ║
║   • Beta Schedule: {self.beta_schedule} ({self.beta_start} → {self.beta_end})  ║
║   • Prediction Type: {self.prediction_type}                                  ║
║                                                                              ║
║ Data Pipeline:                                                               ║
║   • Dataset: {self.dataset_name}                                             ║
║   • Tokenizer: {self.tokenizer_name}                                         ║
║   • Streaming: {self.stream_dataset}                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        return summary.strip()


# Default configuration instances for different scales
def get_config_1_4B() -> DiffusionConfig:
    """Get default 1.4B parameter configuration (recommended for A6000 + A4500)."""
    return DiffusionConfig()


def get_config_small() -> DiffusionConfig:
    """Get smaller configuration for testing (350M parameters)."""
    return DiffusionConfig(
        n_layer=12,
        n_head=12,
        n_embd=1024,
        intermediate_size=4096,
        block_size=1024,
        batch_size=16,
        accumulate_grad_batches=8,
    )


def get_config_tiny() -> DiffusionConfig:
    """Get tiny configuration for rapid prototyping (125M parameters)."""
    return DiffusionConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        intermediate_size=3072,
        block_size=512,
        batch_size=32,
        accumulate_grad_batches=4,
        max_steps=100000,
    )


if __name__ == "__main__":
    # Test configuration
    config = get_config_1_4B()
    print(config.summary())
    print(f"\nEstimated parameter count: {config.get_parameter_count():,}")
