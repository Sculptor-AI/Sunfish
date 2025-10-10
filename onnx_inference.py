#!/usr/bin/env python3
"""
ONNX Inference Script for SunFish Diffusion Model
Deploy this on your AI platform for production use
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm


class SunFishONNXInference:
    """ONNX inference wrapper for SunFish diffusion model."""

    def __init__(self, onnx_path: str, timesteps: int = 500):
        """
        Initialize ONNX inference.

        Args:
            onnx_path: Path to ONNX model
            timesteps: Number of diffusion timesteps (from training config)
        """
        print(f"🐟 Loading SunFish ONNX model from: {onnx_path}")

        # Load ONNX model
        self.session = ort.InferenceSession(onnx_path)

        # Get model info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Get embedding dimension from model
        input_shape = self.session.get_inputs()[0].shape
        self.n_embd = input_shape[2]  # [batch, seq, embd]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Diffusion schedule (linear beta schedule)
        self.timesteps = timesteps
        betas = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        print(f"✅ Model loaded successfully")
        print(f"   - Embedding dimension: {self.n_embd}")
        print(f"   - Timesteps: {timesteps}")

    def predict_noise(self, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Run single denoising step."""
        inputs = {
            self.input_names[0]: x_t.astype(np.float32),
            self.input_names[1]: t.astype(np.int64)
        }
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]

    def ddim_sample(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_steps: int = 20,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        DDIM sampling for fast generation.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            num_steps: Number of denoising steps (lower = faster)
            show_progress: Show progress bar

        Returns:
            embeddings: [batch, seq_len, n_embd]
        """
        # Start with pure noise
        x = np.random.randn(batch_size, seq_len, self.n_embd).astype(np.float32)

        # Create timestep schedule (uniformly spaced)
        timesteps = np.linspace(self.timesteps - 1, 0, num_steps).astype(int)

        iterator = tqdm(enumerate(timesteps), total=len(timesteps), desc="DDIM Sampling") if show_progress else enumerate(timesteps)

        for i, t in iterator:
            t_batch = np.array([t] * batch_size, dtype=np.int64)

            # Predict noise
            predicted_noise = self.predict_noise(x, t_batch)

            # Get alpha values
            alpha_cumprod_t = self.sqrt_alphas_cumprod[t] ** 2

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.sqrt_alphas_cumprod[t_prev] ** 2
            else:
                alpha_cumprod_t_prev = 1.0

            # Predict x_0
            pred_x0 = (x - np.sqrt(1 - alpha_cumprod_t) * predicted_noise) / np.sqrt(alpha_cumprod_t)

            # Deterministic direction
            direction = np.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise

            # Compute x_{t-1}
            x = np.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction

        return x

    def embeddings_to_tokens(self, embeddings: np.ndarray, vocab_size: int = 8192) -> np.ndarray:
        """
        Convert embeddings to tokens using nearest neighbor.

        Note: This requires the token embedding matrix. For ONNX deployment,
        you need to either:
        1. Export token embeddings separately
        2. Use a separate tokenization model
        3. Load from PyTorch checkpoint

        Args:
            embeddings: [batch, seq_len, n_embd]
            vocab_size: Vocabulary size

        Returns:
            token_ids: [batch, seq_len]
        """
        # For now, return placeholder - you'll need to provide vocab embeddings
        raise NotImplementedError(
            "Token embedding matrix needed for rounding. "
            "Either load from PyTorch checkpoint or export separately."
        )

    def generate(
        self,
        num_samples: int = 1,
        seq_len: int = 128,
        num_steps: int = 20,
        return_embeddings: bool = False
    ):
        """
        Generate text samples.

        Args:
            num_samples: Number of samples to generate
            seq_len: Sequence length
            num_steps: Number of denoising steps
            return_embeddings: If True, return embeddings instead of text

        Returns:
            embeddings if return_embeddings=True, else list of text strings
        """
        print(f"\n🔮 Generating {num_samples} samples of length {seq_len}...")

        # Generate embeddings
        embeddings = self.ddim_sample(
            batch_size=num_samples,
            seq_len=seq_len,
            num_steps=num_steps,
            show_progress=True
        )

        if return_embeddings:
            return embeddings

        # For text output, you need token embeddings
        # This is a limitation of ONNX - the full pipeline needs the embedding matrix
        print("⚠️  Token conversion requires embedding matrix (not in ONNX export)")
        print("   Use return_embeddings=True and post-process with PyTorch model")
        return embeddings


def main():
    """Demo usage."""
    import argparse

    parser = argparse.ArgumentParser(description="SunFish ONNX Inference")
    parser.add_argument("model", type=str, help="Path to ONNX model")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--length", type=int, default=128, help="Sequence length")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps")

    args = parser.parse_args()

    # Create inference engine
    engine = SunFishONNXInference(args.model)

    # Generate embeddings
    embeddings = engine.generate(
        num_samples=args.samples,
        seq_len=args.length,
        num_steps=args.steps,
        return_embeddings=True
    )

    print(f"\n✅ Generated embeddings shape: {embeddings.shape}")
    print(f"   Use PyTorch model to convert to text, or integrate token embeddings")


if __name__ == "__main__":
    main()
