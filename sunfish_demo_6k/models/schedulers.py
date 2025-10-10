"""
Diffusion Schedulers for SunFish
Implements DDPM and DDIM sampling (Section 9)
"""

import torch
from typing import Optional
from tqdm import tqdm


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) Scheduler.

    Implements ancestral sampling algorithm (Section 9).
    This is the "standard" diffusion sampling but requires T steps (slow).

    Reference: Ho et al. 2020 - Denoising Diffusion Probabilistic Models
    """

    def __init__(self, model):
        """
        Args:
            model: SunFishTransformer model with diffusion schedule
        """
        self.model = model

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: Optional[int] = None,
        show_progress: bool = True,
        prompt: Optional[str] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using DDPM ancestral sampling with optional CFG.

        Algorithm:
        1. Start from pure noise x_T ~ N(0, I)
        2. For t = T, ..., 1:
            - Predict noise: ε_θ(x_t, t)
            - If using CFG: ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
            - Compute predicted x_0: (x_t - sqrt(1-ᾱ_t)ε) / sqrt(ᾱ_t)
            - Sample x_{t-1} using reverse process

        Args:
            shape: Tuple (batch_size, seq_len, n_embd)
            num_steps: Number of denoising steps (default: model.config.timesteps)
            show_progress: Whether to show progress bar
            prompt: Optional text prompt for conditional generation
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            x_0: Generated embeddings [batch, seq_len, n_embd]
        """
        if num_steps is None:
            num_steps = self.model.config.timesteps

        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)

        # Encode prompt if provided
        context = None
        if prompt is not None and guidance_scale != 1.0:
            # Encode prompt for conditional generation
            prompts = [prompt] * shape[0]  # Repeat for batch
            context = self.model.prompt_encoder.encode(prompts)

        timesteps = reversed(range(num_steps))

        iterator = tqdm(timesteps, desc="DDPM Sampling") if show_progress else timesteps

        for t in iterator:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise with classifier-free guidance
            if context is not None and guidance_scale != 1.0:
                # Conditional prediction
                noise_cond = self.model(x, t_batch, context=context)
                # Unconditional prediction
                noise_uncond = self.model(x, t_batch, context=None)
                # Apply CFG formula
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # Standard unconditional prediction
                predicted_noise = self.model(x, t_batch, context=context)

            # Get schedule values
            alpha_t = self.model.alphas[t]
            alpha_cumprod_t = self.model.alphas_cumprod[t]

            if t > 0:
                alpha_cumprod_t_prev = self.model.alphas_cumprod[t - 1]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(
                alpha_cumprod_t
            )

            # Direction pointing to x_t
            direction = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise

            # Compute mean of p(x_{t-1} | x_t)
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction

            # Add noise (except at last step)
            if t > 0:
                noise = torch.randn_like(x)
                beta_t = self.model.betas[t]
                x = x + torch.sqrt(beta_t) * noise

        return x


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Model (DDIM) Scheduler.

    Implements deterministic/semi-deterministic sampling (Section 9).
    Much faster than DDPM - can use 50 steps instead of 1000!

    Key advantage: Skip timesteps without loss of quality.

    Reference: Song et al. 2020 - Denoising Diffusion Implicit Models
    """

    def __init__(self, model, eta: float = 0.0):
        """
        Args:
            model: SunFishTransformer model with diffusion schedule
            eta: Stochasticity parameter
                - 0.0 = fully deterministic (DDIM)
                - 1.0 = equivalent to DDPM
        """
        self.model = model
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 50,
        show_progress: bool = True,
        prompt: Optional[str] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM with optional CFG.

        Algorithm:
        1. Create timestep schedule (can skip steps!)
        2. Start from pure noise x_T ~ N(0, I)
        3. For each timestep:
            - Predict noise (with CFG if enabled)
            - Compute predicted x_0
            - Deterministically compute x_{t-1}
            - Optionally add noise (controlled by eta)

        Args:
            shape: Tuple (batch_size, seq_len, n_embd)
            num_steps: Number of denoising steps (much smaller than training!)
            show_progress: Whether to show progress bar
            prompt: Optional text prompt for conditional generation
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            x_0: Generated embeddings [batch, seq_len, n_embd]
        """
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)

        # Encode prompt if provided
        context = None
        if prompt is not None and guidance_scale != 1.0:
            # Encode prompt for conditional generation
            prompts = [prompt] * shape[0]  # Repeat for batch
            context = self.model.prompt_encoder.encode(prompts)

        # Create timestep schedule (uniform spacing)
        timesteps = torch.linspace(
            self.model.config.timesteps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        iterator = (
            tqdm(enumerate(timesteps), total=len(timesteps), desc="DDIM Sampling")
            if show_progress
            else enumerate(timesteps)
        )

        for i, t in iterator:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise with classifier-free guidance
            if context is not None and guidance_scale != 1.0:
                # Conditional prediction
                noise_cond = self.model(x, t_batch, context=context)
                # Unconditional prediction
                noise_uncond = self.model(x, t_batch, context=None)
                # Apply CFG formula
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # Standard prediction
                predicted_noise = self.model(x, t_batch, context=context)

            # Get alpha values
            alpha_cumprod_t = self.model.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.model.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(
                alpha_cumprod_t
            )

            # Compute sigma (controls stochasticity)
            sigma = (
                self.eta
                * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
                * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # Direction to x_t
            direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * predicted_noise

            # Compute x_{t-1}
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction

            # Add noise (only if eta > 0)
            if self.eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                x = x + sigma * noise

        return x


class ConstrainedDDIMScheduler(DDIMScheduler):
    """
    DDIM scheduler with constraints for infilling tasks.

    Allows fixing certain positions while denoising others.
    Useful for text infilling / editing (Section 10).
    """

    @torch.no_grad()
    def sample_with_constraint(
        self,
        shape: tuple,
        known_embeddings: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples with some positions fixed.

        Args:
            shape: Tuple (batch_size, seq_len, n_embd)
            known_embeddings: Fixed embeddings at known positions [batch, seq_len, n_embd]
            mask: Boolean mask [batch, seq_len] - True for known positions
            num_steps: Number of denoising steps
            show_progress: Whether to show progress bar

        Returns:
            x_0: Generated embeddings [batch, seq_len, n_embd]
        """
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)

        # Expand mask to embedding dimension
        mask_expanded = mask.unsqueeze(-1).expand_as(x)

        # Initialize known positions
        x = torch.where(mask_expanded, known_embeddings, x)

        timesteps = torch.linspace(
            self.model.config.timesteps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        iterator = (
            tqdm(enumerate(timesteps), total=len(timesteps), desc="Constrained DDIM")
            if show_progress
            else enumerate(timesteps)
        )

        for i, t in iterator:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Get alpha values
            alpha_cumprod_t = self.model.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.model.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(
                alpha_cumprod_t
            )

            # Sigma
            sigma = (
                self.eta
                * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
                * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # Direction
            direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * predicted_noise

            # Compute x_{t-1}
            x_next = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction

            # Add noise
            if self.eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x_next)
                x_next = x_next + sigma * noise

            # Re-inject known positions (constraint)
            x = torch.where(mask_expanded, known_embeddings, x_next)

        return x


if __name__ == "__main__":
    print("✅ Schedulers module loaded successfully!")
    print("\nAvailable schedulers:")
    print("  - DDPMScheduler: Ancestral sampling (slow but accurate)")
    print("  - DDIMScheduler: Fast deterministic sampling")
    print("  - ConstrainedDDIMScheduler: Infilling with constraints")
