#!/usr/bin/env python3
"""
Complete ONNX inference with token conversion
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm
import json


class SunFishONNX:
    def __init__(self, model_dir="."):
        # Load ONNX model
        self.session = ort.InferenceSession(f"{model_dir}/sunfish_model.onnx")

        # Load config
        with open(f"{model_dir}/config.json") as f:
            self.config = json.load(f)

        # Load token embeddings
        self.token_embeddings = np.load(f"{model_dir}/token_embeddings.npy")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_name'])

        # Setup diffusion schedule
        timesteps = self.config['timesteps']
        betas = np.linspace(self.config['beta_start'], self.config['beta_end'], timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)

        print(f"✅ Model loaded: {self.config['n_embd']}D, {timesteps} timesteps")

    def sample_ddim(self, batch_size=1, seq_len=128, num_steps=20):
        """DDIM sampling."""
        x = np.random.randn(batch_size, seq_len, self.config['n_embd']).astype(np.float32)
        timesteps = np.linspace(self.config['timesteps'] - 1, 0, num_steps).astype(int)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = np.array([t] * batch_size, dtype=np.int64)

            # Predict noise
            noise = self.session.run(
                ['predicted_noise'],
                {'noisy_embeddings': x, 'timesteps': t_batch}
            )[0]

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else 1.0

            pred_x0 = (x - np.sqrt(1 - alpha_t) * noise) / np.sqrt(alpha_t)
            direction = np.sqrt(1 - alpha_prev) * noise
            x = np.sqrt(alpha_prev) * pred_x0 + direction

        return x

    def embeddings_to_tokens(self, embeddings):
        """Convert embeddings to tokens using nearest neighbor."""
        # Normalize
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        vocab_norm = self.token_embeddings / (np.linalg.norm(self.token_embeddings, axis=-1, keepdims=True) + 1e-8)

        # Compute similarities
        similarities = np.matmul(emb_norm, vocab_norm.T)
        token_ids = np.argmax(similarities, axis=-1)

        return token_ids

    def generate(self, num_samples=3, seq_len=128, num_steps=20):
        """Generate text."""
        print(f"\n🔮 Generating {num_samples} samples...")

        # Sample embeddings
        embeddings = self.sample_ddim(num_samples, seq_len, num_steps)

        # Convert to tokens
        token_ids = self.embeddings_to_tokens(embeddings)

        # Decode to text
        texts = []
        for tokens in token_ids:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)

        return texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    # Load model
    model = SunFishONNX()

    # Generate
    texts = model.generate(args.samples, args.length, args.steps)

    # Display
    print("\n" + "="*80)
    for i, text in enumerate(texts):
        print(f"\nSample {i+1}:")
        print("-"*80)
        print(text)
    print("="*80)
