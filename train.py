#!/usr/bin/env python3
"""
Training Script for Sunfish Diffusion LLM

This script handles:
- Model initialization
- Data loading
- Distributed training with FSDP
- Checkpointing and logging
- Mixed precision training
"""

import os
import sys
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from pytorch_lightning.strategies import FSDPStrategy
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.model_config import get_config_1_4B, get_config_small, get_config_tiny
from models.diffusion_model import DiffusionTransformer
from data.fineweb_datamodule import FineWebDataModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Sunfish Diffusion LLM")

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="1.4B",
        choices=["1.4B", "small", "tiny"],
        help="Model configuration preset",
    )

    # Checkpoint resumption
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Override config options
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--num-workers", type=int, help="Override num data workers")

    # Data configuration
    parser.add_argument(
        "--local-dataset",
        type=str,
        default=None,
        help="Path to locally downloaded dataset (default: stream from HuggingFace)",
    )

    # Logging
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")

    # Debugging
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 batch for quick testing",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="Overfit on N batches (debugging)",
    )

    # Hardware
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )

    return parser.parse_args()


def get_config(args):
    """Get configuration based on arguments."""
    # Load base config
    if args.config == "1.4B":
        config = get_config_1_4B()
    elif args.config == "small":
        config = get_config_small()
    elif args.config == "tiny":
        config = get_config_tiny()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Apply overrides
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.precision is not None:
        config.precision = args.precision

    # W&B config
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name

    # Debugging
    if args.fast_dev_run:
        config.fast_dev_run = True
    if args.overfit_batches > 0:
        config.overfit_batches = args.overfit_batches

    return config


def setup_loggers(config):
    """Set up logging (W&B and TensorBoard)."""
    loggers = []

    # W&B logger
    if config.use_wandb:
        run_name = config.wandb_run_name or f"sunfish-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            log_model=False,  # Don't log full model to W&B (too large)
            config=vars(config),
        )
        loggers.append(wandb_logger)
        logger.info(f"W&B logging enabled: {config.wandb_project}/{run_name}")

    # TensorBoard logger (always enable as backup)
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="tensorboard",
        version=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    loggers.append(tb_logger)

    return loggers


def setup_callbacks(config):
    """Set up training callbacks."""
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="sunfish-{step:06d}-{train/loss:.4f}",
        save_top_k=config.save_top_k,
        save_last=config.save_last,
        monitor="train/loss",
        mode="min",
        every_n_train_steps=config.save_every_n_steps,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Rich progress bar (better than default)
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    # Model summary
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)

    return callbacks


def setup_strategy(config, num_devices):
    """Set up distributed training strategy."""
    if config.strategy == "fsdp" and num_devices > 1:
        # FSDP configuration for multi-GPU training
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        from functools import partial

        # Determine sharding strategy
        if config.fsdp_sharding_strategy == "FULL_SHARD":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif config.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD

        # Auto wrap policy (wrap layers larger than 1e8 parameters)
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e8)

        strategy = FSDPStrategy(
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=None,  # Could add if memory constrained
            cpu_offload=config.fsdp_cpu_offload,
        )

        logger.info(f"Using FSDP strategy: {config.fsdp_sharding_strategy}")
        return strategy

    elif config.strategy == "ddp" and num_devices > 1:
        return "ddp"

    else:
        return "auto"


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Get configuration
    config = get_config(args)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed, workers=True)

    # Print configuration
    logger.info("\n" + "=" * 80)
    logger.info("Starting Sunfish Diffusion LLM Training")
    logger.info("=" * 80)
    print(config.summary())

    # ========================================================================
    # Initialize Data Module
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("Initializing data module...")
    logger.info("-" * 80)

    # Use local dataset if provided
    local_data_dir = args.local_dataset if hasattr(args, 'local_dataset') else None
    data_module = FineWebDataModule(config, local_data_dir=local_data_dir)
    data_module.setup()

    # ========================================================================
    # Initialize Model
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("Initializing model...")
    logger.info("-" * 80)

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = DiffusionTransformer.load_from_checkpoint(args.resume, config=config)
    else:
        model = DiffusionTransformer(config)

    logger.info(f"Model initialized: {model.count_parameters():,} parameters")

    # ========================================================================
    # Setup Training
    # ========================================================================
    loggers = setup_loggers(config)
    callbacks = setup_callbacks(config)

    # Determine number of devices
    if args.gpus is not None:
        num_devices = args.gpus
    else:
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    strategy = setup_strategy(config, num_devices)

    # ========================================================================
    # Initialize Trainer
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("Initializing trainer...")
    logger.info("-" * 80)

    trainer = pl.Trainer(
        # Hardware
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices if num_devices > 0 else "auto",
        strategy=strategy,
        precision=config.precision,
        # Training loop
        max_steps=config.max_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.max_grad_norm,
        gradient_clip_algorithm="norm",
        # Logging
        logger=loggers,
        log_every_n_steps=config.log_every_n_steps,
        # Callbacks
        callbacks=callbacks,
        # Checkpointing
        enable_checkpointing=True,
        # Validation
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=None,  # We use steps, not epochs
        # Performance
        deterministic=False,  # For best performance
        benchmark=True,  # Use cuDNN benchmark for speed
        # Debugging
        fast_dev_run=config.fast_dev_run,
        overfit_batches=config.overfit_batches,
        detect_anomaly=config.detect_anomaly,
        # Profiling (disabled by default)
        profiler=None,
    )

    # Print training info
    effective_batch_size = config.batch_size * num_devices * config.accumulate_grad_batches
    total_tokens_per_step = effective_batch_size * config.block_size
    total_tokens = total_tokens_per_step * config.max_steps

    logger.info("\n" + "=" * 80)
    logger.info("Training Configuration:")
    logger.info("-" * 80)
    logger.info(f"  Devices: {num_devices} × {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Precision: {config.precision}")
    logger.info(f"  Batch size per device: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.accumulate_grad_batches}")
    logger.info(f"  Effective batch size: {effective_batch_size} sequences")
    logger.info(f"  Tokens per step: {total_tokens_per_step:,}")
    logger.info(f"  Total training steps: {config.max_steps:,}")
    logger.info(f"  Total tokens: {total_tokens:,} (~{total_tokens / 1e9:.1f}B)")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Warmup steps: {config.warmup_steps}")
    logger.info("=" * 80 + "\n")

    # ========================================================================
    # Start Training
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80 + "\n")

    try:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

        # Print final checkpoint path
        if trainer.checkpoint_callback:
            logger.info(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
            logger.info(f"Last checkpoint: {trainer.checkpoint_callback.last_model_path}")

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("Training interrupted by user")
        logger.warning("=" * 80)

        if trainer.checkpoint_callback:
            logger.info(f"\nLast checkpoint saved at: {trainer.checkpoint_callback.last_model_path}")

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"Training failed with error: {e}")
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()
