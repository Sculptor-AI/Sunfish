#!/usr/bin/env python3
"""
Inference Script for Sunfish Diffusion LLM

Supports:
- Unconditional generation (from pure noise)
- Text infilling (fill in [MASK] tokens)
- DDPM or DDIM sampling
"""

import os
import sys
import torch
import argparse
import logging
from transformers import AutoTokenizer
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.model_config import DiffusionConfig
from models.diffusion_model import DiffusionTransformer
from models.schedulers import DDPMSampler, DDIMSampler, round_embeddings_to_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text using Sunfish Diffusion LLM")

    # Model checkpoint
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.ckpt file)",
    )

    # Generation mode
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "infill"],
        help="Generation mode: generate from scratch or infill masked text",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=256,
        help="Generation length (number of tokens)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of diffusion steps (more = slower but better)",
    )

    # Sampling method
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddpm", "ddim"],
        help="Sampling algorithm",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0=deterministic, 1=stochastic)",
    )

    # Infilling parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for infilling (must contain [MASK])",
    )
    parser.add_argument(
        "--mask-length",
        type=int,
        default=20,
        help="Number of tokens to generate for [MASK]",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: print to console)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (not yet implemented)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create a minimal config for loading
    # The actual config will be loaded from checkpoint
    config = DiffusionConfig()

    # Load model
    model = DiffusionTransformer.load_from_checkpoint(checkpoint_path, config=config, map_location=device)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {model.count_parameters():,} parameters")
    logger.info(f"Vocab size: {model.config.vocab_size}")
    logger.info(f"Context length: {model.config.block_size}")

    return model


def generate_unconditional(
    model: DiffusionTransformer,
    tokenizer: AutoTokenizer,
    num_samples: int,
    length: int,
    num_steps: int,
    sampler_type: str = "ddim",
    eta: float = 0.0,
    device: str = "cuda",
) -> list:
    """
    Generate text unconditionally (from pure noise).

    Args:
        model: Trained diffusion model
        tokenizer: Tokenizer for decoding
        num_samples: Number of samples to generate
        length: Length of each sample (in tokens)
        num_steps: Number of diffusion steps
        sampler_type: 'ddpm' or 'ddim'
        eta: DDIM eta parameter
        device: Device to use

    Returns:
        List of generated text strings
    """
    logger.info(f"\nGenerating {num_samples} samples of length {length} tokens...")
    logger.info(f"Using {sampler_type.upper()} sampler with {num_steps} steps")

    # Create sampler
    if sampler_type == "ddpm":
        sampler = DDPMSampler(model)
    else:
        sampler = DDIMSampler(model, eta=eta)

    # Sample in embedding space
    shape = (num_samples, length, model.config.n_embd)
    with torch.no_grad():
        generated_embeddings = sampler.sample(shape, num_steps=num_steps, show_progress=True)

    # Round embeddings to tokens
    logger.info("Converting embeddings to tokens...")
    token_ids = round_embeddings_to_tokens(generated_embeddings, model.token_embedding, method="cosine")

    # Decode tokens to text
    texts = []
    for i, ids in enumerate(token_ids):
        text = tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True)
        texts.append(text)

    return texts


def generate_infill(
    model: DiffusionTransformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    mask_length: int,
    num_steps: int,
    sampler_type: str = "ddim",
    eta: float = 0.0,
    device: str = "cuda",
) -> str:
    """
    Fill in masked text.

    Args:
        model: Trained diffusion model
        tokenizer: Tokenizer
        prompt: Text with [MASK] placeholder
        mask_length: Number of tokens to generate for mask
        num_steps: Number of diffusion steps
        sampler_type: 'ddpm' or 'ddim'
        eta: DDIM eta parameter
        device: Device

    Returns:
        Completed text with mask filled in
    """
    if "[MASK]" not in prompt:
        raise ValueError("Prompt must contain [MASK] token")

    logger.info(f"\nInfilling masked text...")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Mask length: {mask_length} tokens")

    # Split prompt at [MASK]
    parts = prompt.split("[MASK]", 1)
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""

    # Tokenize prefix and suffix
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    logger.info(f"Prefix: {len(prefix_ids)} tokens, Suffix: {len(suffix_ids)} tokens")

    # Get embeddings for prefix and suffix
    with torch.no_grad():
        prefix_emb = model.token_embedding(torch.tensor(prefix_ids, device=device)).unsqueeze(0)
        suffix_emb = model.token_embedding(torch.tensor(suffix_ids, device=device)).unsqueeze(0)

    # Initialize masked region with random noise
    masked_emb = torch.randn(1, mask_length, model.config.n_embd, device=device)

    # Concatenate
    full_sequence = torch.cat([prefix_emb, masked_emb, suffix_emb], dim=1)
    seq_len = full_sequence.shape[1]

    # Create mask indicating which positions are fixed
    fixed_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    fixed_mask[: len(prefix_ids)] = True
    fixed_mask[len(prefix_ids) + mask_length :] = True

    # Create sampler
    if sampler_type == "ddpm":
        sampler = DDPMSampler(model)
    else:
        sampler = DDIMSampler(model, eta=eta)

    # Denoise with constraint (keep prefix/suffix fixed)
    logger.info("Denoising...")
    timesteps = torch.linspace(
        model.config.timesteps - 1, 0, num_steps, dtype=torch.long, device=device
    )

    x = full_sequence
    for i, t in enumerate(timesteps):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            # Predict noise
            predicted_noise = model(x, t_batch)

            # Get alpha values
            alpha_bar_t = model.alphas_cumprod[t]

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_t_next = model.alphas_cumprod[t_next]

                # DDIM update
                direction = torch.sqrt(1 - alpha_bar_t_next) * predicted_noise
                x = torch.sqrt(alpha_bar_t_next) * pred_x0 + direction
            else:
                x = pred_x0

            # Re-inject fixed tokens (keep prefix and suffix unchanged)
            x[0, fixed_mask] = torch.cat([prefix_emb[0], suffix_emb[0]], dim=0)

    # Convert to tokens
    token_ids = round_embeddings_to_tokens(x, model.token_embedding, method="cosine")

    # Decode
    completed_text = tokenizer.decode(token_ids[0].cpu().tolist(), skip_special_tokens=True)

    return completed_text


def main():
    """Main inference function."""
    args = parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model.config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_name)

    # Generate
    if args.mode == "generate":
        # Unconditional generation
        texts = generate_unconditional(
            model=model,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            length=args.length,
            num_steps=args.num_steps,
            sampler_type=args.sampler,
            eta=args.eta,
            device=args.device,
        )

        # Print results
        print("\n" + "=" * 80)
        print("Generated Samples:")
        print("=" * 80)
        for i, text in enumerate(texts):
            print(f"\n--- Sample {i+1} ---")
            print(text)
            print()

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                for i, text in enumerate(texts):
                    f.write(f"=== Sample {i+1} ===\n")
                    f.write(text)
                    f.write("\n\n")
            logger.info(f"Saved to: {args.output}")

    elif args.mode == "infill":
        # Infilling
        if args.prompt is None:
            raise ValueError("Must provide --prompt for infill mode")

        completed_text = generate_infill(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            mask_length=args.mask_length,
            num_steps=args.num_steps,
            sampler_type=args.sampler,
            eta=args.eta,
            device=args.device,
        )

        # Print result
        print("\n" + "=" * 80)
        print("Infilled Text:")
        print("=" * 80)
        print(f"\nOriginal: {args.prompt}")
        print(f"Completed: {completed_text}\n")

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                f.write(f"Original: {args.prompt}\n")
                f.write(f"Completed: {completed_text}\n")
            logger.info(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
