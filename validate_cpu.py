#!/usr/bin/env python3
"""
Sunfish masked diffusion CPU validation script.
Runs lightweight checks on config, data pipeline, forward/backward, and sampling.
"""

import argparse
import sys
import torch

from config.qwen_masked_config import get_qwen_masked_config_cpu
from data.qwen_datamodule import QwenDataModule
from models.masked_diffusion_lm import MaskedDiffusionLM
from models.discrete_sampler import DiscreteDiffusionSampler


def build_config(args):
    config = get_qwen_masked_config_cpu()
    if args.base_model:
        config.base_model = args.base_model
    if args.block_size:
        config.block_size = args.block_size
    if args.use_shift:
        config.use_shift = True
    return config


def test_datamodule(config):
    print("\n" + "=" * 70)
    print("TEST 1: Data Module")
    print("=" * 70)

    dm = QwenDataModule(config)
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    print(f"Batch shape: {input_ids.shape}")
    print(f"Dtype: {input_ids.dtype}")
    print(f"Token range: [{input_ids.min()}, {input_ids.max()}]")

    assert input_ids.dtype == torch.long, "Expected input_ids dtype torch.long"
    assert input_ids.ndim == 2, "Expected 2D batch [batch, seq]"

    return batch


def test_model_forward_backward(config, batch):
    print("\n" + "=" * 70)
    print("TEST 2: Model Forward/Backward")
    print("=" * 70)

    model = MaskedDiffusionLM(config)
    model.train()

    token_ids = batch["input_ids"] if isinstance(batch, dict) else batch
    batch_size = token_ids.shape[0]
    device = token_ids.device

    t = torch.randint(
        1,
        config.timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    masked_tokens, mask = model.forward_mask(token_ids, t)
    logits = model.forward(masked_tokens)

    if config.use_shift:
        targets = token_ids[:, 1:]
        mask_shifted = mask[:, 1:]
    else:
        targets = token_ids
        mask_shifted = mask

    logits_flat = logits.reshape(-1, model.vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = mask_shifted.reshape(-1).float()

    loss_per_token = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
    )
    masked_loss = loss_per_token * mask_flat
    num_masked = mask_flat.sum().clamp(min=1)
    loss = masked_loss.sum() / num_masked
    print(f"Training loss: {loss.item():.6f}")

    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    if not has_grad:
        raise RuntimeError("Backward pass failed (no gradients found)")
    print("Backward pass: OK")

    return model


@torch.no_grad()
def test_sampler(model, seq_len, num_steps):
    print("\n" + "=" * 70)
    print("TEST 3: Discrete Sampler")
    print("=" * 70)

    sampler = DiscreteDiffusionSampler(model)
    tokens = sampler.sample(
        batch_size=1,
        seq_len=seq_len,
        num_steps=num_steps,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        show_progress=False,
    )

    print(f"Sampled tokens shape: {tokens.shape}")
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Validate masked diffusion on CPU")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model (HuggingFace model id or local path)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Override block size for CPU validation",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Unmasking steps for sampler test",
    )
    parser.add_argument(
        "--use-shift",
        action="store_true",
        help="Enable shift operation during validation",
    )
    args = parser.parse_args()

    config = build_config(args)
    config.block_size = args.block_size

    print("\nSunfish Masked Diffusion CPU Validation")
    print(f"Base model: {config.base_model}")
    print(f"Block size: {config.block_size}")
    print(f"Use shift: {config.use_shift}")

    try:
        batch = test_datamodule(config)
        model = test_model_forward_backward(config, batch)
        test_sampler(model, seq_len=min(32, config.block_size), num_steps=args.num_steps)
    except Exception as exc:
        print(f"\nValidation failed: {exc}")
        print("If this is a download error, ensure the base model is available locally or online.")
        sys.exit(1)

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
