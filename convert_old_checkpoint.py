#!/usr/bin/env python3
"""
Convert old checkpoint (without prompt conditioning) to new architecture.
This allows loading old checkpoints into the new model.
"""

import torch
import argparse
from config import SunFishConfig, TinySunFishConfig
from models import SunFishTransformer


def convert_checkpoint(old_checkpoint_path: str, output_path: str, config_type: str = "tiny"):
    """Convert old checkpoint to new architecture."""

    print(f"🔄 Converting checkpoint: {old_checkpoint_path}")

    # Load old checkpoint
    old_checkpoint = torch.load(old_checkpoint_path, map_location="cpu", weights_only=False)

    # Get config
    if config_type == "tiny":
        config = TinySunFishConfig()
    else:
        config = SunFishConfig()

    # Create new model
    print("✨ Creating new model with prompt conditioning...")
    new_model = SunFishTransformer(config)

    # Map old keys to new keys
    old_state = old_checkpoint["state_dict"]
    new_state = new_model.state_dict()

    converted_state = {}

    print("🔧 Mapping old transformer layers to new transformer blocks...")

    # Copy embeddings and time encoders (these didn't change)
    for key in old_state.keys():
        if key.startswith(("token_embedding", "pos_embedding", "time_encoder", "time_mlp", "ln_f", "noise_head")):
            if key in new_state:
                converted_state[key] = old_state[key]
                print(f"  ✓ Copied: {key}")

    # Map transformer layers to transformer blocks
    # Old: transformer.layers.{i}.*
    # New: transformer_blocks.{i}.*
    num_layers = config.n_layer

    for i in range(num_layers):
        old_prefix = f"transformer.layers.{i}"
        new_prefix = f"transformer_blocks.{i}"

        # Map self-attention (same structure)
        for suffix in [".self_attn.in_proj_weight", ".self_attn.in_proj_bias",
                       ".self_attn.out_proj.weight", ".self_attn.out_proj.bias"]:
            old_key = old_prefix + suffix
            new_key = new_prefix + suffix
            if old_key in old_state:
                converted_state[new_key] = old_state[old_key]
                print(f"  ✓ Mapped: {old_key} → {new_key}")

        # Map norms
        for old_norm, new_norm in [("norm1", "norm1"), ("norm2", "norm3")]:
            for suffix in [".weight", ".bias"]:
                old_key = f"{old_prefix}.{old_norm}{suffix}"
                new_key = f"{new_prefix}.{new_norm}{suffix}"
                if old_key in old_state:
                    converted_state[new_key] = old_state[old_key]
                    print(f"  ✓ Mapped: {old_key} → {new_key}")

        # Map FFN (linear1/linear2 -> ffn.0/ffn.3)
        for old_linear, new_linear in [("linear1", "ffn.0"), ("linear2", "ffn.3")]:
            for suffix in [".weight", ".bias"]:
                old_key = f"{old_prefix}.{old_linear}{suffix}"
                new_key = f"{new_prefix}.{new_linear}{suffix}"
                if old_key in old_state:
                    converted_state[new_key] = old_state[old_key]
                    print(f"  ✓ Mapped: {old_key} → {new_key}")

        # Initialize cross-attention and norm2 with random values
        # (they'll be trained when you add prompts)
        print(f"  ⚠️  Initializing new cross-attention for layer {i} (will need training)")

    # Load converted state (strict=False to allow missing cross-attention weights)
    new_model.load_state_dict(converted_state, strict=False)

    # Create new checkpoint
    new_checkpoint = {
        "state_dict": new_model.state_dict(),
        "hyper_parameters": old_checkpoint.get("hyper_parameters", {}),
        "epoch": old_checkpoint.get("epoch", 0),
        "global_step": old_checkpoint.get("global_step", 0),
    }

    # Save
    torch.save(new_checkpoint, output_path)
    print(f"\n✅ Converted checkpoint saved to: {output_path}")
    print(f"\n📝 Note: Cross-attention layers are randomly initialized.")
    print(f"   - Model will work for unconditional generation")
    print(f"   - For prompt conditioning, you'll need to fine-tune with prompts")

    return new_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old checkpoint to new architecture")
    parser.add_argument("input", type=str, help="Path to old checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: input_converted.ckpt)")
    parser.add_argument("--config", type=str, default="tiny", choices=["tiny", "full"], help="Config type")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".ckpt", "_converted.ckpt")

    convert_checkpoint(args.input, args.output, args.config)
