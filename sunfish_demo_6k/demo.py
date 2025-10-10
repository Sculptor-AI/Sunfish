#!/usr/bin/env python3
"""
SunFish Demo - Portable Text Generation
"""

import torch
from models import SunFishTransformer, DDIMScheduler
from transformers import AutoTokenizer

def generate_text(prompt=None, num_samples=3, seq_len=128, guidance_scale=7.0):
    """Generate text with the SunFish model."""

    # Load model
    print("🐟 Loading SunFish model...")
    model = SunFishTransformer.load_from_checkpoint("model.ckpt", map_location="cpu")
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create sampler
    sampler = DDIMScheduler(model, eta=0.0)

    print(f"🔮 Generating {num_samples} samples...")

    # Generate embeddings
    shape = (num_samples, seq_len, model.config.n_embd)
    embeddings = sampler.sample(
        shape,
        num_steps=20,
        show_progress=True,
        prompt=prompt,
        guidance_scale=guidance_scale if prompt else 1.0
    )

    # Round to tokens
    vocab_embeddings = model.token_embedding.weight
    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)
    vocab_norm = torch.nn.functional.normalize(vocab_embeddings, dim=-1)
    similarities = torch.matmul(embeddings_norm, vocab_norm.T)
    token_ids = similarities.argmax(dim=-1)

    # Decode
    print("\n" + "="*80)
    for i, tokens in enumerate(token_ids):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\nSample {i+1}:")
        print("-"*80)
        print(text)
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SunFish Demo")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--length", type=int, default=128, help="Sequence length")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")

    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        num_samples=args.samples,
        seq_len=args.length,
        guidance_scale=args.guidance
    )
