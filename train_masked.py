#!/usr/bin/env python3
"""
Masked Diffusion Training Script
Train the discrete masked diffusion language model.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import FSDPStrategy

from config.qwen_masked_config import (
    QwenMaskedDiffusionConfig,
    get_qwen_masked_config,
    get_qwen_masked_config_cpu,
)
from models.masked_diffusion_lm import MaskedDiffusionLM
from data.qwen_datamodule import QwenDataModule


def setup_callbacks(config, overwrite_last: bool = False):
    """Set up training callbacks."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/masked/",
        filename="masked-diffusion-{epoch:02d}-{step}",
        save_top_k=config.save_top_k,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=config.checkpoint_every_n_steps,
        save_last=True,
        enable_version_counter=not overwrite_last,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def setup_logger(config):
    """Set up experiment logger."""
    wandb_disabled = os.environ.get("WANDB_DISABLED") == "true"
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    wandb_key = os.environ.get("WANDB_API_KEY")

    if wandb_disabled or wandb_mode in {"disabled", "offline"} or not wandb_key:
        print("WandB disabled or not configured, using TensorBoard")
        return TensorBoardLogger("logs/", name="masked-diffusion")

    try:
        logger = WandbLogger(
            project=config.project_name,
            name=config.experiment_name or "masked-diffusion-run",
            log_model=False,
        )
        print("Using WandB logger")
        return logger
    except Exception:
        print("WandB not available, using TensorBoard")
        return TensorBoardLogger("logs/", name="masked-diffusion")


def setup_strategy(config):
    """Set up distributed training strategy."""
    if config.strategy == "fsdp" and config.accelerator == "gpu":
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
        )
        print("Using FSDP strategy")
        return strategy
    elif config.strategy == "ddp" and torch.cuda.device_count() > 1:
        return "ddp"
    else:
        return "auto"


def print_model_info(model, config):
    """Print model information."""
    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)

    print("\n" + "=" * 70)
    print("SUNFISH MASKED DIFFUSION LM")
    print("=" * 70)

    print(f"\nModel Statistics:")
    print(f"  Base Model:           {config.base_model}")
    print(f"  Total Parameters:     {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Non-Embedding Params: {non_emb_params:,} ({non_emb_params/1e9:.2f}B)")
    print(f"  Vocabulary Size:      {model.vocab_size:,}")
    print(f"  Mask Token ID:        {model.mask_token_id}")

    print(f"\nTraining Configuration:")
    print(f"  Batch Size:           {config.batch_size}")
    print(f"  Gradient Accumulation: {config.accumulate_grad_batches}")
    print(f"  Effective Batch Size: {config.effective_batch_size} sequences")
    print(f"  Tokens per Batch:     {config.effective_tokens_per_batch:,}")
    print(f"  Learning Rate:        {config.learning_rate}")
    print(f"  Weight Decay:         {config.weight_decay}")
    print(f"  Max Steps:            {config.max_steps:,}")
    print(f"  A2D Warmup Steps:     {config.a2d_warmup_steps:,}")
    print(f"  Warmup Steps:         {config.warmup_steps:,}")

    print(f"\nMasked Diffusion Configuration:")
    print(f"  Timesteps:            {config.timesteps}")
    print(f"  Mask Schedule:        {config.mask_schedule}")
    print(f"  Block Size:           {config.block_size}")
    print(f"  Bidirectional:        {config.bidirectional}")
    print(f"  Use Shift:            {config.use_shift}")
    print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")

    print(f"\nCompute Configuration:")
    print(f"  Device:               {config.accelerator.upper()}")
    print(f"  Precision:            {config.precision}")
    print(f"  Strategy:             {config.strategy}")

    if config.accelerator == "gpu" and torch.cuda.is_available():
        print(f"  GPUs Available:       {torch.cuda.device_count()}")
        print(f"  GPU Memory:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Masked Diffusion LM")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model (HuggingFace model id or local path)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (for testing)",
    )
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Use TPU (XLA) training",
    )
    parser.add_argument(
        "--tpu-cores",
        type=int,
        default=None,
        help="Number of TPU cores to use (default: auto; Colab v5e often uses 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override sequence block size",
    )
    parser.add_argument(
        "--use-shift",
        action="store_true",
        help="Enable shift operation (predict i+1 from hidden i)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random data for fast testing",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["wikitext-2", "wikitext-103", "openwebtext", "fineweb"],
        help="Dataset to use (default: wikitext-103)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Override checkpoint interval in steps",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=None,
        help="Override how many best checkpoints to keep",
    )
    parser.add_argument(
        "--overwrite-last",
        action="store_true",
        help="Overwrite last.ckpt instead of versioning (last-v1.ckpt)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers (Windows: 0 is safest)",
    )

    args = parser.parse_args()

    # Load configuration
    if args.tpu:
        config = get_qwen_masked_config()
        config.accelerator = "tpu"
        config.devices = "auto" if args.tpu_cores is None else int(args.tpu_cores)
        config.precision = "bf16-mixed"
        config.num_workers = 0
        config.strategy = "auto"
        print("Using TPU configuration")
    elif args.cpu:
        config = get_qwen_masked_config_cpu()
        print("Using CPU configuration for testing")
    else:
        config = get_qwen_masked_config()
        print("Using GPU configuration")

    # Improve Tensor Core utilization on NVIDIA GPUs
    if not args.cpu and not args.tpu and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Apply overrides
    if args.name:
        config.experiment_name = args.name
    if args.base_model:
        config.base_model = args.base_model
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.accumulate:
        config.accumulate_grad_batches = args.accumulate
    if args.block_size:
        config.block_size = args.block_size
    if args.use_shift:
        config.use_shift = True
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    if args.synthetic:
        config.use_synthetic_data = True
    if args.dataset:
        config.dataset_name = args.dataset
    if args.checkpoint_every:
        config.checkpoint_every_n_steps = args.checkpoint_every
    if args.save_top_k is not None:
        config.save_top_k = args.save_top_k
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    # Initialize model
    print("\nInitializing model...")
    model = MaskedDiffusionLM(config)

    # Initialize data
    print("Initializing data module...")
    datamodule = QwenDataModule(config)

    # Print model info
    print_model_info(model, config)

    # Set up callbacks and logger
    callbacks = setup_callbacks(config, overwrite_last=args.overwrite_last)
    logger = setup_logger(config)
    strategy = setup_strategy(config)

    # Initialize trainer
    # Note: val_check_interval might be larger than training data, so use None for epoch-based validation
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices="auto" if config.devices == "auto" else int(config.devices),
        strategy=strategy,
        precision=config.precision,
        max_steps=config.max_steps,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=None,  # Validate at end of each epoch (safer for small datasets)
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        deterministic=False,
        enable_progress_bar=True,
    )

    if args.resume:
        try:
            torch.serialization.add_safe_globals([QwenMaskedDiffusionConfig])
        except Exception as exc:
            print(f"Warning: failed to allowlist config class for resume ({exc})")

    # Start training
    print("\nStarting training...\n")

    try:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if callbacks:
            print(f"Last checkpoint: {callbacks[0].last_model_path}")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise

    print("\nTraining complete!")
    if callbacks:
        print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
