#!/usr/bin/env python3
"""
SunFish Sampling Script
Generate text using trained diffusion model (Section 9-10)
"""

import argparse
import torch
from transformers import AutoTokenizer

from models import SunFishTransformer, DDIMScheduler, DDPMScheduler, ConstrainedDDIMScheduler
from config import SunFishConfig


def round_embeddings_to_tokens(
    embeddings: torch.Tensor,
    token_embedding_layer: torch.nn.Embedding,
    method: str = "nearest",
) -> torch.Tensor:
    """
    Round continuous embeddings to discrete tokens (Section 3).

    Args:
        embeddings: [batch, seq_len, n_embd]
        token_embedding_layer: Embedding layer from model
        method: 'nearest' or 'gumbel' (nearest is faster)

    Returns:
        token_ids: [batch, seq_len]
    """
    batch_size, seq_len, _ = embeddings.shape

    # Get all token embeddings
    vocab_embeddings = token_embedding_layer.weight  # [vocab_size, n_embd]

    if method == "nearest":
        # Nearest neighbor using cosine similarity
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)
        vocab_norm = torch.nn.functional.normalize(vocab_embeddings, dim=-1)

        # [batch, seq_len, vocab_size]
        similarities = torch.matmul(embeddings_norm, vocab_norm.T)

        # Get nearest tokens
        token_ids = similarities.argmax(dim=-1)

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    return token_ids


@torch.no_grad()
def generate_text(
    checkpoint_path: str,
    num_samples: int = 5,
    seq_len: int = 256,
    num_steps: int = 50,
    scheduler: str = "ddim",
    temperature: float = 1.0,
    show_progress: bool = True,
    prompt: str = None,
    guidance_scale: float = None,
):
    """
    Generate text using trained diffusion model with optional prompt conditioning.

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to generate
        seq_len: Sequence length
        num_steps: Number of denoising steps
        scheduler: 'ddim' or 'ddpm'
        temperature: Sampling temperature (for noise scaling)
        show_progress: Show progress bar
        prompt: Optional text prompt for conditional generation
        guidance_scale: Classifier-free guidance scale (None = use config default)
    """
    print(f"\n🐟 Loading SunFish model from: {checkpoint_path}")

    # Load model
    model = SunFishTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device.upper()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_name)

    # Use config guidance scale if not specified
    if guidance_scale is None:
        guidance_scale = getattr(model.config, "guidance_scale", 7.5)

    # Create scheduler
    if scheduler == "ddim":
        sampler = DDIMScheduler(model, eta=0.0)  # Deterministic
        print(f"Using DDIM scheduler with {num_steps} steps")
    else:
        sampler = DDPMScheduler(model)
        print(f"Using DDPM scheduler with {num_steps} steps")

    # Print generation settings
    if prompt:
        print(f"\n📝 Prompt: \"{prompt}\"")
        print(f"🎯 Guidance Scale: {guidance_scale}")
    else:
        print(f"\n🎲 Unconditional generation (no prompt)")

    # Generate embeddings
    print(f"\n🔮 Generating {num_samples} samples of length {seq_len}...")
    shape = (num_samples, seq_len, model.config.n_embd)

    embeddings = sampler.sample(
        shape,
        num_steps=num_steps,
        show_progress=show_progress,
        prompt=prompt,
        guidance_scale=guidance_scale
    )

    # Scale by temperature
    if temperature != 1.0:
        embeddings = embeddings / temperature

    # Round to tokens
    print("\n🔄 Rounding embeddings to tokens...")
    token_ids = round_embeddings_to_tokens(embeddings, model.token_embedding)

    # Decode and display
    print("\n" + "=" * 80)
    print("📝 GENERATED TEXT")
    print("=" * 80)

    for i, ids in enumerate(token_ids):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"\n{'─' * 80}")
        print(f"Sample {i + 1}:")
        print(f"{'─' * 80}")
        print(text)

    print("\n" + "=" * 80)


@torch.no_grad()
def infill_text(
    checkpoint_path: str,
    text: str,
    num_steps: int = 50,
    mask_token: str = "[MASK]",
    show_progress: bool = True,
):
    """
    Fill in masked text (Section 10).

    Args:
        checkpoint_path: Path to model checkpoint
        text: String with mask token
        num_steps: Number of denoising steps
        mask_token: Token indicating where to infill
        show_progress: Show progress bar
    """
    print(f"\n🐟 Loading SunFish model from: {checkpoint_path}")

    # Load model
    model = SunFishTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_name)

    # Parse text
    if mask_token not in text:
        print(f"❌ Error: '{mask_token}' not found in text")
        return

    parts = text.split(mask_token)
    if len(parts) != 2:
        print(f"❌ Error: Expected exactly one '{mask_token}' token")
        return

    prefix_text, suffix_text = parts

    # Tokenize
    prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"][0]
    suffix_ids = tokenizer(suffix_text, return_tensors="pt")["input_ids"][0]

    # Create full sequence
    mask_len = 10  # Generate 10 tokens for mask
    seq_len = len(prefix_ids) + mask_len + len(suffix_ids)

    print(f"\n📊 Infilling:")
    print(f"  Prefix tokens: {len(prefix_ids)}")
    print(f"  Mask tokens: {mask_len}")
    print(f"  Suffix tokens: {len(suffix_ids)}")
    print(f"  Total length: {seq_len}")

    # Get embeddings for known parts
    prefix_emb = model.token_embedding(prefix_ids.to(device))
    suffix_emb = model.token_embedding(suffix_ids.to(device))

    # Initialize middle with noise
    middle_emb = torch.randn(mask_len, model.config.n_embd, device=device)

    # Combine
    known_embeddings = torch.cat([prefix_emb, middle_emb, suffix_emb], dim=0).unsqueeze(0)

    # Create mask (True = known, False = unknown)
    mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    mask[:, len(prefix_ids) : len(prefix_ids) + mask_len] = False

    # Sample with constraint
    print(f"\n🔄 Infilling with constrained DDIM...")
    scheduler = ConstrainedDDIMScheduler(model, eta=0.0)

    shape = (1, seq_len, model.config.n_embd)
    embeddings = scheduler.sample_with_constraint(
        shape,
        known_embeddings,
        mask,
        num_steps=num_steps,
        show_progress=show_progress,
    )

    # Round to tokens
    token_ids = round_embeddings_to_tokens(embeddings, model.token_embedding)

    # Decode
    result_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("📝 INFILLED TEXT")
    print("=" * 80)
    print(result_text)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate text with SunFish")

    # Common arguments
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "infill"],
        help="Generation mode",
    )

    # Generation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Sequence length",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Sampling scheduler",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )

    # Classifier-free guidance arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for conditional generation (CFG)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (default: use config value)",
    )

    # Infilling arguments
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text with [MASK] token for infilling",
    )

    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Hide progress bar",
    )

    args = parser.parse_args()

    # Execute
    if args.mode == "generate":
        generate_text(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            scheduler=args.scheduler,
            temperature=args.temperature,
            show_progress=not args.no_progress,
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
        )
    elif args.mode == "infill":
        if args.text is None:
            print("❌ Error: --text required for infill mode")
            return

        infill_text(
            checkpoint_path=args.checkpoint,
            text=args.text,
            num_steps=args.num_steps,
            show_progress=not args.no_progress,
        )


if __name__ == "__main__":
    main()
