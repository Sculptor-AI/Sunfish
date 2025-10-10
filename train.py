#!/usr/bin/env python3
"""
SunFish Training Script
Train the diffusion language model (Section 7-8)
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import FSDPStrategy

from config import SunFishConfig, TinySunFishConfig, MicroSunFishConfig, NanoSunFishConfig
from models import SunFishTransformer
from data import FineWebDataModule


def setup_callbacks(config):
    """Set up training callbacks."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="sunfish-{epoch:02d}-{step}",
        save_top_k=config.save_top_k,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=config.checkpoint_every_n_steps,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def setup_logger(config):
    """Set up experiment logger."""

    # Try WandB first, fallback to TensorBoard
    try:
        logger = WandbLogger(
            project=config.project_name,
            name=config.experiment_name or "sunfish-run",
            log_model=False,
        )
        print("✅ Using WandB logger")
        return logger
    except:
        print("⚠️  WandB not available, using TensorBoard")
        logger = TensorBoardLogger("logs/", name="sunfish")
        return logger


def setup_strategy(config):
    """Set up distributed training strategy."""

    if config.strategy == "fsdp" and config.accelerator == "gpu":
        # FSDP for large models on multi-GPU
        strategy = FSDPStrategy(
            auto_wrap_policy={torch.nn.TransformerEncoderLayer},
            activation_checkpointing_policy={torch.nn.TransformerEncoderLayer},
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
        )
        print("✅ Using FSDP strategy")
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
    print("🐟 SUNFISH DIFFUSION LLM")
    print("=" * 70)
    print(f"\n📊 Model Statistics:")
    print(f"  Total Parameters:     {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Non-Embedding Params: {non_emb_params:,} ({non_emb_params/1e9:.2f}B)")
    print(f"  Layers:               {config.n_layer}")
    print(f"  Attention Heads:      {config.n_head}")
    print(f"  Embedding Dimension:  {config.n_embd}")
    print(f"  Sequence Length:      {config.block_size}")
    print(f"  Vocabulary Size:      {config.vocab_size}")

    print(f"\n⚙️  Training Configuration:")
    print(f"  Batch Size:           {config.batch_size}")
    print(f"  Gradient Accumulation: {config.accumulate_grad_batches}")
    print(f"  Effective Batch Size: {config.batch_size * config.accumulate_grad_batches}")
    print(f"  Learning Rate:        {config.learning_rate}")
    print(f"  Weight Decay:         {config.weight_decay}")
    print(f"  Max Steps:            {config.max_steps:,}")
    print(f"  Warmup Steps:         {config.warmup_steps:,}")

    print(f"\n🔬 Diffusion Configuration:")
    print(f"  Timesteps:            {config.timesteps}")
    print(f"  Beta Schedule:        [{config.beta_start}, {config.beta_end}]")

    print(f"\n💻 Compute Configuration:")
    print(f"  Device:               {config.accelerator.upper()}")
    print(f"  Precision:            {config.precision}")
    print(f"  Strategy:             {config.strategy}")

    if config.accelerator == "gpu":
        print(f"  GPUs Available:       {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"  GPU Memory:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train SunFish Diffusion LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="full",
        choices=["full", "tiny", "micro", "nano"],
        help="Model configuration (full, tiny, micro, or nano)",
    )
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
        "--cpu",
        action="store_true",
        help="Force CPU training (for testing)",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config == "tiny":
        config = TinySunFishConfig()
        print("🐟 Using TINY configuration for testing")
    elif args.config == "micro":
        config = MicroSunFishConfig()
        print("🐟 Using MICRO configuration for coherent text generation")
    elif args.config == "nano":
        config = NanoSunFishConfig()
        print("🐟 Using NANO configuration for fast CPU validation")
    else:
        config = SunFishConfig()
        print("🐟 Using FULL configuration for production training")

    if args.cpu:
        config.accelerator = "cpu"
        config.precision = "32"
        config.strategy = "auto"
        print("⚠️  Forcing CPU training")

    if args.name:
        config.experiment_name = args.name

    # Initialize model
    model = SunFishTransformer(config)

    # Initialize data
    datamodule = FineWebDataModule(config)

    # Print model info
    print_model_info(model, config)

    # Set up callbacks and logger
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    strategy = setup_strategy(config)

    # Initialize trainer
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
        val_check_interval=config.val_check_interval,
        enable_checkpointing=True,
        deterministic=False,  # For performance
        enable_progress_bar=True,
    )

    # Start training
    print("\n🚀 Starting training...\n")

    try:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Last checkpoint saved at: {callbacks[0].last_model_path}")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        raise

    print("\n✅ Training complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
