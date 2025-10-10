#!/usr/bin/env python3
"""
Export SunFish model to ONNX format for deployment
"""

import torch
import argparse
from models import SunFishTransformer


def export_to_onnx(checkpoint_path: str, output_path: str):
    """Export model to ONNX format."""

    print(f"🐟 Loading SunFish model from: {checkpoint_path}")
    model = SunFishTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    # Get model dimensions
    batch_size = 1
    seq_len = 128
    n_embd = model.config.n_embd

    print(f"📐 Model dimensions:")
    print(f"   - Embedding dim: {n_embd}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    x_t = torch.randn(batch_size, seq_len, n_embd)
    t = torch.randint(0, model.config.timesteps, (batch_size,))

    # Optional: context for prompt conditioning
    # context = torch.randn(batch_size, 64, n_embd)  # Prompt embeddings

    print(f"\n📦 Exporting to ONNX...")

    # Export with dynamic axes for flexibility
    torch.onnx.export(
        model,
        (x_t, t),  # Add context if using prompts
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['noisy_embeddings', 'timesteps'],
        output_names=['predicted_noise'],
        dynamic_axes={
            'noisy_embeddings': {0: 'batch_size', 1: 'seq_len'},
            'timesteps': {0: 'batch_size'},
            'predicted_noise': {0: 'batch_size', 1: 'seq_len'}
        }
    )

    print(f"✅ Model exported to: {output_path}")
    print(f"\n📝 Model size: {torch.onnx._get_model_size(output_path) / 1024 / 1024:.1f} MB")

    print(f"\n🚀 To use this model:")
    print(f"   1. Load with ONNX Runtime: pip install onnxruntime")
    print(f"   2. Run inference with custom sampling loop")
    print(f"   3. Convert embeddings to tokens using your tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SunFish to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="sunfish_model.onnx", help="Output ONNX path")

    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
