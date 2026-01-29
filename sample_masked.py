#!/usr/bin/env python3
"""
Masked Diffusion Sampling Script
Generate text using trained masked diffusion model.
"""

import argparse
import torch

from models.masked_diffusion_lm import MaskedDiffusionLM
from models.discrete_sampler import DiscreteDiffusionSampler


@torch.no_grad()
def generate_text(
    checkpoint_path: str,
    num_samples: int = 5,
    seq_len: int = 128,
    num_steps: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    show_progress: bool = True,
    prompt: str = None,
):
    """
    Generate text using trained masked diffusion model.

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to generate
        seq_len: Sequence length to generate
        num_steps: Number of unmasking steps
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p filtering
        show_progress: Show progress bar
        prompt: Optional text prompt to condition on
    """
    print(f"\nLoading model from: {checkpoint_path}")

    # Load model
    model = MaskedDiffusionLM.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device.upper()}")

    # Create sampler
    sampler = DiscreteDiffusionSampler(model)

    # Handle prompt
    prefix_ids = None
    if prompt:
        print(f"\nPrompt: \"{prompt}\"")
        prefix_tokens = model.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        prefix_ids = prefix_tokens.expand(num_samples, -1).to(device)
        actual_gen_len = seq_len
        print(f"Prefix length: {prefix_ids.shape[1]} tokens")
    else:
        print("\nUnconditional generation")
        actual_gen_len = seq_len

    # Generate
    print(f"\nGenerating {num_samples} samples of length {actual_gen_len}...")
    print(f"Steps: {num_steps}, Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")

    try:
        tokens = sampler.sample(
            batch_size=num_samples,
            seq_len=actual_gen_len,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            show_progress=show_progress,
            prefix_ids=prefix_ids,
        )
    except ValueError as exc:
        print(f"Generation error: {exc}")
        return None, None

    # Decode and display
    print("\n" + "=" * 80)
    print("GENERATED TEXT")
    print("=" * 80)

    texts = model.decode(tokens)
    for i, text in enumerate(texts):
        print(f"\n{'-' * 80}")
        print(f"Sample {i + 1}:")
        print(f"{'-' * 80}")
        # Handle Windows console encoding issues
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'))

    print("\n" + "=" * 80)

    return tokens, texts


@torch.no_grad()
def infill_text(
    checkpoint_path: str,
    text: str,
    infill_len: int = 20,
    num_steps: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    mask_token: str = "[MASK]",
    show_progress: bool = True,
):
    """
    Fill in masked sections of text.

    Args:
        checkpoint_path: Path to model checkpoint
        text: Text with [MASK] token indicating where to infill
        infill_len: Number of tokens to generate for infill
        num_steps: Number of unmasking steps
        temperature: Sampling temperature
        top_k: Top-k filtering
        mask_token: Token indicating infill position
        show_progress: Show progress bar
    """
    print(f"\nLoading model from: {checkpoint_path}")

    # Load model
    model = MaskedDiffusionLM.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Parse text
    if mask_token not in text:
        print(f"Error: '{mask_token}' not found in text")
        return

    parts = text.split(mask_token)
    if len(parts) != 2:
        print(f"Error: Expected exactly one '{mask_token}' token")
        return

    prefix_text, suffix_text = parts

    print(f"\nPrefix: \"{prefix_text}\"")
    print(f"Suffix: \"{suffix_text}\"")
    print(f"Infill length: {infill_len} tokens")

    # Tokenize
    prefix_ids = model.tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    suffix_ids = model.tokenizer(
        suffix_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    print(f"Prefix tokens: {prefix_ids.shape[1]}")
    print(f"Suffix tokens: {suffix_ids.shape[1]}")

    # Create sampler and infill
    sampler = DiscreteDiffusionSampler(model)

    tokens = sampler.infill(
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        infill_len=infill_len,
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        show_progress=show_progress,
    )

    # Decode
    result_text = model.decode(tokens)[0]

    print("\n" + "=" * 80)
    print("INFILLED TEXT")
    print("=" * 80)
    try:
        print(result_text)
    except UnicodeEncodeError:
        print(result_text.encode('ascii', 'replace').decode('ascii'))
    print("=" * 80)

    return tokens, result_text


@torch.no_grad()
def interactive_mode(checkpoint_path: str):
    """
    Interactive text generation mode.
    """
    print(f"\nLoading model from: {checkpoint_path}")

    model = MaskedDiffusionLM.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    sampler = DiscreteDiffusionSampler(model)

    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Commands:")
    print("  Enter text prompt to generate continuation")
    print("  Type 'quit' or 'exit' to stop")
    print("  Type 'settings' to change generation settings")
    print("=" * 80 + "\n")

    # Default settings
    settings = {
        "seq_len": 128,
        "num_steps": 50,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
    }

    while True:
        try:
            user_input = input("\nPrompt> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "settings":
                print("\nCurrent settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print("\nEnter setting to change (e.g., 'temperature 0.8') or 'back':")

                while True:
                    setting_input = input("Setting> ").strip()
                    if setting_input.lower() == "back":
                        break
                    parts = setting_input.split()
                    if len(parts) == 2 and parts[0] in settings:
                        try:
                            if parts[0] in ["seq_len", "num_steps", "top_k"]:
                                settings[parts[0]] = int(parts[1])
                            else:
                                settings[parts[0]] = float(parts[1])
                            print(f"Set {parts[0]} = {settings[parts[0]]}")
                        except ValueError:
                            print("Invalid value")
                    else:
                        print("Invalid setting")
                continue

            # Generate with prompt
            prefix_ids = model.tokenizer(
                user_input,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)

            print(f"\nGenerating ({prefix_ids.shape[1]} prefix tokens)...")

            try:
                tokens = sampler.sample(
                    batch_size=1,
                    seq_len=settings["seq_len"],
                    num_steps=settings["num_steps"],
                    temperature=settings["temperature"],
                    top_k=settings["top_k"],
                    top_p=settings["top_p"],
                    show_progress=True,
                    prefix_ids=prefix_ids,
                )
            except ValueError as exc:
                print(f"Generation error: {exc}")
                continue

            text = model.decode(tokens)[0]
            print(f"\n{'-' * 60}")
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode('ascii', 'replace').decode('ascii'))
            print(f"{'-' * 60}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue


def main():
    parser = argparse.ArgumentParser(description="Generate text with Masked Diffusion")

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "infill", "interactive"],
        help="Generation mode",
    )

    # Generation arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length to generate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of unmasking steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering (0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) filtering",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for conditional generation",
    )

    # Infilling arguments
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text with [MASK] token for infilling",
    )
    parser.add_argument(
        "--infill-len",
        type=int,
        default=20,
        help="Number of tokens to generate for infill",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bar",
    )

    args = parser.parse_args()

    if args.mode == "generate":
        generate_text(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            show_progress=not args.no_progress,
            prompt=args.prompt,
        )

    elif args.mode == "infill":
        if args.text is None:
            print("Error: --text required for infill mode")
            return

        infill_text(
            checkpoint_path=args.checkpoint,
            text=args.text,
            infill_len=args.infill_len,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            show_progress=not args.no_progress,
        )

    elif args.mode == "interactive":
        interactive_mode(args.checkpoint)


if __name__ == "__main__":
    main()
