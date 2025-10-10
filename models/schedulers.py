"""
Diffusion Sampling Schedulers for Inference
Implements DDPM (slow) and DDIM (fast) sampling
"""

import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DDPMSampler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) Sampler.

    This is the original ancestral sampling method from DDPM paper.
    Requires T steps (e.g., 1000) for best quality but is very slow.
    """

    def __init__(self, model):
        """
        Args:
            model: DiffusionTransformer model with diffusion schedule buffers
        """
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples from pure noise using DDPM sampling.

        Args:
            shape: (batch_size, seq_len, n_embd) shape of samples to generate
            num_steps: Number of denoising steps (defaults to model.config.timesteps)
            show_progress: Show progress bar

        Returns:
            Denoised embeddings of shape [batch_size, seq_len, n_embd]
        """
        device = next(self.model.parameters()).device

        if num_steps is None:
            num_steps = self.model.config.timesteps
        else:
            num_steps = min(num_steps, self.model.config.timesteps)

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Denoise step by step from T-1 to 0
        timesteps = range(num_steps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")

        for t in timesteps:
            # Create batch of timestep indices
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise at this timestep
            predicted_noise = self.model(x, t_batch)

            # Get schedule values
            alpha_t = self.model.alphas[t]
            alpha_bar_t = self.model.alphas_cumprod[t]
            beta_t = self.model.betas[t]

            # Compute predicted x_0 (clean sample)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            if t > 0:
                # Get previous alpha_bar
                alpha_bar_t_prev = self.model.alphas_cumprod[t - 1]

                # Compute mean of p(x_{t-1} | x_t, x_0)
                # Using the formula from DDPM paper
                coeff_pred_x0 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t)
                coeff_x_t = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t)

                mean = coeff_pred_x0 * pred_x0 + coeff_x_t * x

                # Compute variance
                # Using beta_t as variance (simplified, could use posterior variance)
                variance = beta_t

                # Sample noise
                noise = torch.randn_like(x)

                # Sample x_{t-1}
                x = mean + torch.sqrt(variance) * noise
            else:
                # At t=0, just use predicted x_0 (no noise)
                x = pred_x0

        return x


class DDIMSampler:
    """
    Denoising Diffusion Implicit Model (DDIM) Sampler.

    DDIM allows deterministic sampling with much fewer steps (e.g., 50 instead of 1000).
    The eta parameter controls stochasticity:
    - eta=0: fully deterministic (recommended for language)
    - eta=1: equivalent to DDPM (stochastic)
    """

    def __init__(self, model, eta: float = 0.0):
        """
        Args:
            model: DiffusionTransformer model
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        """
        self.model = model
        self.eta = eta
        self.model.eval()

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM with fewer steps.

        Args:
            shape: (batch_size, seq_len, n_embd)
            num_steps: Number of denoising steps (much less than training steps)
            show_progress: Show progress bar

        Returns:
            Denoised embeddings [batch_size, seq_len, n_embd]
        """
        device = next(self.model.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Create a schedule of timesteps to step through
        # We uniformly sample num_steps timesteps from [0, T-1]
        timesteps = torch.linspace(
            self.model.config.timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=device,
        )

        if show_progress:
            timesteps = tqdm(timesteps, desc=f"DDIM Sampling (eta={self.eta})")

        for i, t in enumerate(timesteps):
            t = t.long()

            # Create batch of timesteps
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Get alpha values
            alpha_bar_t = self.model.alphas_cumprod[t]

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            if i == len(timesteps) - 1:
                # Last step: return predicted clean sample
                x = pred_x0
            else:
                # Get next timestep
                t_next = timesteps[i + 1].long()
                alpha_bar_t_next = self.model.alphas_cumprod[t_next]

                # DDIM update formula
                # Compute sigma (controls stochasticity)
                sigma = (
                    self.eta
                    * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t))
                    * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)
                )

                # Direction pointing to x_t
                direction = torch.sqrt(1 - alpha_bar_t_next - sigma**2) * predicted_noise

                # Compute x_{t-1}
                x = torch.sqrt(alpha_bar_t_next) * pred_x0 + direction

                # Add noise if eta > 0
                if self.eta > 0 and i < len(timesteps) - 1:
                    noise = torch.randn_like(x)
                    x = x + sigma * noise

        return x


def round_embeddings_to_tokens(
    embeddings: torch.Tensor,
    token_embedding: nn.Embedding,
    method: str = "cosine",
) -> torch.Tensor:
    """
    Convert continuous embeddings to discrete tokens.

    This is the crucial step for diffusion LMs: after generating continuous
    embeddings, we need to map them back to the discrete token vocabulary.

    Args:
        embeddings: [batch, seq_len, d_model] continuous embeddings
        token_embedding: The model's token embedding layer
        method: 'cosine' or 'euclidean' distance metric

    Returns:
        token_ids: [batch, seq_len] discrete token indices
    """
    batch_size, seq_len, d_model = embeddings.shape

    # Get all vocabulary embeddings
    vocab_embeddings = token_embedding.weight  # [vocab_size, d_model]

    if method == "cosine":
        # Normalize for cosine similarity
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)  # [B, L, D]
        vocab_norm = torch.nn.functional.normalize(vocab_embeddings, dim=-1)  # [V, D]

        # Compute cosine similarity: [B, L, V]
        # We use einsum for efficiency instead of matmul
        similarities = torch.einsum("bld,vd->blv", embeddings_norm, vocab_norm)

        # Get token with highest similarity
        token_ids = similarities.argmax(dim=-1)  # [B, L]

    elif method == "euclidean":
        # Compute L2 distance (negative for argmax)
        # [B, L, D] - [V, D] -> [B, L, V]
        embeddings_expanded = embeddings.unsqueeze(2)  # [B, L, 1, D]
        vocab_expanded = vocab_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, V, D]

        distances = -torch.norm(embeddings_expanded - vocab_expanded, dim=-1, p=2)  # [B, L, V]
        token_ids = distances.argmax(dim=-1)  # [B, L]

    else:
        raise ValueError(f"Unknown method: {method}")

    return token_ids


if __name__ == "__main__":
    # Test samplers
    import sys

    sys.path.append("..")
    from config.model_config import get_config_tiny
    from models.diffusion_model import DiffusionTransformer

    logging.basicConfig(level=logging.INFO)

    # Create tiny model for testing
    config = get_config_tiny()
    config.n_layer = 2  # Very small for quick testing
    model = DiffusionTransformer(config).cuda()

    print("Testing DDPM sampler...")
    ddpm_sampler = DDPMSampler(model)
    samples_ddpm = ddpm_sampler.sample(
        shape=(2, 64, config.n_embd), num_steps=50, show_progress=True
    )
    print(f"DDPM samples shape: {samples_ddpm.shape}")

    # Round to tokens
    token_ids = round_embeddings_to_tokens(samples_ddpm, model.token_embedding)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Token IDs range: [{token_ids.min()}, {token_ids.max()}]")

    print("\nTesting DDIM sampler...")
    ddim_sampler = DDIMSampler(model, eta=0.0)
    samples_ddim = ddim_sampler.sample(shape=(2, 64, config.n_embd), num_steps=20, show_progress=True)
    print(f"DDIM samples shape: {samples_ddim.shape}")

    print("\nSamplers test complete!")
