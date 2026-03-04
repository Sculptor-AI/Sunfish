"""
Discrete Diffusion Sampler
Implements MDLM ancestral posterior sampling for text generation.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Callable
from tqdm import tqdm


def _sample_categorical(categorical_probs):
    """Sample from a categorical distribution via the Gumbel-max trick.

    Works with unnormalized probabilities. Matching the reference at
    https://github.com/kuleshov-group/mdlm (diffusion.py).
    """
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _get_move_chance(t, schedule="linear"):
    """Masking probability at normalized time t in [0, 1].

    t=0 -> clean (no masking), t=1 -> fully masked.
    """
    if schedule == "cosine":
        return 1.0 - torch.cos(t * math.pi / 2)
    return t


class DiscreteDiffusionSampler:
    """
    Sampler for discrete masked diffusion models.

    Uses MDLM ancestral posterior sampling: at each reverse step,
    every masked position independently decides whether to unmask
    based on the diffusion schedule, rather than a confidence-based
    heuristic.
    """

    def __init__(self, model):
        """
        Args:
            model: MaskedDiffusionLM model
        """
        self.model = model
        self.mask_token_id = model.mask_token_id
        self.vocab_size = model.vocab_size
        self.mask_schedule = getattr(model.config, "mask_schedule", "linear")

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        show_progress: bool = True,
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples via MDLM ancestral posterior sampling.

        At each step from t (more masked) to s (less masked), every masked
        position samples from the posterior: stay masked with probability
        proportional to move_chance_s, or unmask to a predicted token with
        probability proportional to p_x0 * (move_chance_t - move_chance_s).

        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of sequences to generate
            num_steps: Number of reverse diffusion steps
            temperature: Sampling temperature (applied to model logits)
            top_k: Top-k filtering (0 = disabled)
            top_p: Top-p (nucleus) filtering (1.0 = disabled)
            show_progress: Show progress bar
            prefix_ids: Optional prefix tokens [batch, prefix_len] to condition on

        Returns:
            tokens: [batch_size, seq_len] generated token IDs
        """
        device = next(self.model.parameters()).device

        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        if prefix_ids is not None:
            if prefix_ids.shape[0] != batch_size:
                raise ValueError("prefix_ids batch size must match batch_size")
            prefix_len = prefix_ids.shape[1]
            if prefix_len > seq_len:
                raise ValueError("prefix_ids length must be <= seq_len")
            tokens[:, :prefix_len] = prefix_ids

        eps = 1e-4
        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating", leave=False)

        for i in iterator:
            t = timesteps[i]
            s = timesteps[i + 1]
            move_chance_t = _get_move_chance(t, self.mask_schedule)
            move_chance_s = _get_move_chance(s, self.mask_schedule)

            logits = self._get_model_logits(tokens, temperature, top_k, top_p)
            p_x0 = F.softmax(logits, dim=-1).to(torch.float64)

            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[..., self.mask_token_id] = move_chance_s

            sampled = _sample_categorical(q_xs)

            is_masked = tokens == self.mask_token_id
            tokens = torch.where(is_masked, sampled, tokens)

        self._final_denoise(tokens)
        return tokens

    @torch.no_grad()
    def sample_with_guidance(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate with optional guidance function using posterior sampling.

        The guidance function takes logits and returns modified logits,
        allowing for controlled generation.

        Args:
            batch_size: Number of sequences
            seq_len: Sequence length
            num_steps: Reverse diffusion steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            guidance_fn: Function(logits, tokens, step) -> modified logits
            guidance_scale: How strongly to apply guidance
            show_progress: Show progress bar

        Returns:
            tokens: Generated token IDs
        """
        device = next(self.model.parameters()).device

        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        eps = 1e-4
        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Guided generation", leave=False)

        for i in iterator:
            t = timesteps[i]
            s = timesteps[i + 1]
            move_chance_t = _get_move_chance(t, self.mask_schedule)
            move_chance_s = _get_move_chance(s, self.mask_schedule)

            logits = self.model.forward(tokens) / temperature
            if self.mask_token_id is not None:
                logits[..., self.mask_token_id] = float("-inf")

            if guidance_fn is not None:
                guided_logits = guidance_fn(logits, tokens, i)
                logits = logits + guidance_scale * (guided_logits - logits)

            if top_k > 0:
                logits = self._top_k_filtering(logits, top_k)

            p_x0 = F.softmax(logits, dim=-1).to(torch.float64)
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[..., self.mask_token_id] = move_chance_s

            sampled = _sample_categorical(q_xs)

            is_masked = tokens == self.mask_token_id
            tokens = torch.where(is_masked, sampled, tokens)

        self._final_denoise(tokens)
        return tokens

    @torch.no_grad()
    def infill(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        infill_len: int = 10,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Infill between prefix and suffix using posterior sampling.

        Prefix and suffix positions are never masked, so the carry-over
        property of the reverse process preserves them automatically.

        Args:
            prefix_ids: [batch, prefix_len] - tokens before infill
            suffix_ids: [batch, suffix_len] - tokens after infill
            infill_len: Number of tokens to generate in middle
            num_steps: Reverse diffusion steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            show_progress: Show progress bar

        Returns:
            tokens: [batch, prefix_len + infill_len + suffix_len]
        """
        device = next(self.model.parameters()).device
        batch_size = prefix_ids.shape[0]
        prefix_len = prefix_ids.shape[1]
        suffix_len = suffix_ids.shape[1]
        total_len = prefix_len + infill_len + suffix_len

        tokens = torch.full(
            (batch_size, total_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens[:, :prefix_len] = prefix_ids
        tokens[:, prefix_len + infill_len:] = suffix_ids

        eps = 1e-4
        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Infilling", leave=False)

        for i in iterator:
            t = timesteps[i]
            s = timesteps[i + 1]
            move_chance_t = _get_move_chance(t, self.mask_schedule)
            move_chance_s = _get_move_chance(s, self.mask_schedule)

            logits = self._get_model_logits(tokens, temperature, top_k, 1.0)
            p_x0 = F.softmax(logits, dim=-1).to(torch.float64)

            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[..., self.mask_token_id] = move_chance_s

            sampled = _sample_categorical(q_xs)

            is_masked = tokens == self.mask_token_id
            tokens = torch.where(is_masked, sampled, tokens)

        self._final_denoise(tokens)
        return tokens

    def _get_model_logits(
        self,
        tokens: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Get filtered model logits with [MASK] logit zeroed."""
        logits = self.model.forward(tokens) / temperature
        if self.mask_token_id is not None:
            logits[..., self.mask_token_id] = float("-inf")
        if top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        if top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)
        return logits

    def _final_denoise(self, tokens: torch.Tensor) -> None:
        """Replace any remaining [MASK] tokens with argmax predictions."""
        still_masked = tokens == self.mask_token_id
        if not still_masked.any():
            return
        logits = self.model.forward(tokens)
        if self.mask_token_id is not None:
            logits[..., self.mask_token_id] = float("-inf")
        final_preds = logits.argmax(dim=-1)
        tokens[still_masked] = final_preds[still_masked]

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")
        return logits


if __name__ == "__main__":
    print("DiscreteDiffusionSampler module loaded successfully!")
    print("\nFeatures:")
    print("  - sample(): MDLM posterior sampling with iterative unmasking")
    print("  - sample_with_guidance(): Controlled generation with guidance")
    print("  - infill(): Fill in between prefix and suffix")
