"""
Discrete Diffusion Sampler
Implements iterative unmasking for text generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
from tqdm import tqdm


class DiscreteDiffusionSampler:
    """
    Sampler for discrete masked diffusion models.

    Implements iterative unmasking where tokens are revealed
    based on model confidence, starting from a fully masked sequence.
    """

    def __init__(self, model):
        """
        Args:
            model: MaskedDiffusionLM model
        """
        self.model = model
        self.mask_token_id = model.mask_token_id
        self.vocab_size = model.vocab_size

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
        Generate samples using iterative unmasking.

        Algorithm:
        1. Start with fully masked sequence (or prefix + masked)
        2. For each step:
           - Get model predictions for all positions
           - Sample tokens from predicted distribution
           - Unmask positions with highest confidence
        3. Return final tokens

        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of sequences to generate
            num_steps: Number of unmasking steps
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (0 = disabled)
            top_p: Top-p (nucleus) filtering (1.0 = disabled)
            show_progress: Show progress bar
            prefix_ids: Optional prefix tokens [batch, prefix_len] to condition on

        Returns:
            tokens: [batch_size, seq_len] - generated token IDs
        """
        device = next(self.model.parameters()).device

        # Initialize with masked tokens
        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Track which positions are still masked
        is_masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Handle prefix conditioning
        if prefix_ids is not None:
            if prefix_ids.shape[0] != batch_size:
                raise ValueError("prefix_ids batch size must match batch_size")
            prefix_len = prefix_ids.shape[1]
            if prefix_len > seq_len:
                raise ValueError("prefix_ids length must be <= seq_len")
            tokens[:, :prefix_len] = prefix_ids
            is_masked[:, :prefix_len] = False

        # Calculate unmasking schedule
        total_to_unmask = is_masked.sum().item()
        unmask_per_step = max(1, total_to_unmask // num_steps)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating", leave=False)

        for step in iterator:
            # Skip if nothing left to unmask
            if not is_masked.any():
                break

            # Get model predictions
            logits = self.model.forward(tokens)  # [batch, seq, vocab]

            # Apply temperature
            logits = logits / temperature

            # Prevent sampling the [MASK] token
            if self.mask_token_id is not None:
                logits[..., self.mask_token_id] = float("-inf")

            # Apply top-k filtering
            if top_k > 0:
                logits = self._top_k_filtering(logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                logits = self._top_p_filtering(logits, top_p)

            # Get probabilities and sample
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1,
            ).view(batch_size, seq_len)

            # Get confidence for each position (max probability)
            confidence = probs.max(dim=-1).values  # [batch, seq]

            # Zero out confidence for already-unmasked positions
            confidence = confidence * is_masked.float()

            # Determine how many to unmask this step
            remaining_steps = num_steps - step
            remaining_masked = is_masked.sum().item()
            num_to_unmask = max(1, remaining_masked // remaining_steps)

            # Find most confident positions to unmask
            flat_confidence = confidence.view(-1)
            flat_is_masked = is_masked.view(-1)

            # Get indices of masked positions
            masked_indices = flat_is_masked.nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                break

            # Select top confident among masked
            masked_confidences = flat_confidence[masked_indices]
            num_to_unmask = min(num_to_unmask, len(masked_indices))

            _, top_relative_indices = masked_confidences.topk(num_to_unmask)
            unmask_indices = masked_indices[top_relative_indices]

            # Update tokens at selected positions
            flat_tokens = tokens.view(-1)
            flat_sampled = sampled.view(-1)
            flat_tokens[unmask_indices] = flat_sampled[unmask_indices]

            # Update mask
            flat_is_masked[unmask_indices] = False

            # Reshape back
            tokens = flat_tokens.view(batch_size, seq_len)
            is_masked = flat_is_masked.view(batch_size, seq_len)

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
        Generate with optional guidance function.

        The guidance function takes logits and returns modified logits,
        allowing for controlled generation.

        Args:
            batch_size: Number of sequences
            seq_len: Sequence length
            num_steps: Unmasking steps
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
        is_masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Guided generation", leave=False)

        for step in iterator:
            if not is_masked.any():
                break

            logits = self.model.forward(tokens) / temperature

            if self.mask_token_id is not None:
                logits[..., self.mask_token_id] = float("-inf")

            # Apply guidance if provided
            if guidance_fn is not None:
                guided_logits = guidance_fn(logits, tokens, step)
                logits = logits + guidance_scale * (guided_logits - logits)

            if top_k > 0:
                logits = self._top_k_filtering(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1,
            ).view(batch_size, seq_len)

            confidence = probs.max(dim=-1).values * is_masked.float()

            remaining_steps = num_steps - step
            remaining_masked = is_masked.sum().item()
            num_to_unmask = max(1, remaining_masked // remaining_steps)

            flat_confidence = confidence.view(-1)
            flat_is_masked = is_masked.view(-1)
            masked_indices = flat_is_masked.nonzero(as_tuple=True)[0]

            if len(masked_indices) == 0:
                break

            masked_confidences = flat_confidence[masked_indices]
            num_to_unmask = min(num_to_unmask, len(masked_indices))

            _, top_relative_indices = masked_confidences.topk(num_to_unmask)
            unmask_indices = masked_indices[top_relative_indices]

            flat_tokens = tokens.view(-1)
            flat_sampled = sampled.view(-1)
            flat_tokens[unmask_indices] = flat_sampled[unmask_indices]
            flat_is_masked[unmask_indices] = False

            tokens = flat_tokens.view(batch_size, seq_len)
            is_masked = flat_is_masked.view(batch_size, seq_len)

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
        Infill between prefix and suffix.

        Args:
            prefix_ids: [batch, prefix_len] - tokens before infill
            suffix_ids: [batch, suffix_len] - tokens after infill
            infill_len: Number of tokens to generate in middle
            num_steps: Unmasking steps
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

        # Initialize sequence
        tokens = torch.full(
            (batch_size, total_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens[:, :prefix_len] = prefix_ids
        tokens[:, prefix_len + infill_len:] = suffix_ids

        # Only infill middle is masked
        is_masked = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        is_masked[:, prefix_len:prefix_len + infill_len] = True

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Infilling", leave=False)

        for step in iterator:
            if not is_masked.any():
                break

            logits = self.model.forward(tokens) / temperature

            if self.mask_token_id is not None:
                logits[..., self.mask_token_id] = float("-inf")

            if top_k > 0:
                logits = self._top_k_filtering(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1,
            ).view(batch_size, total_len)

            confidence = probs.max(dim=-1).values * is_masked.float()

            remaining_steps = num_steps - step
            remaining_masked = is_masked.sum().item()
            num_to_unmask = max(1, remaining_masked // remaining_steps)

            flat_confidence = confidence.view(-1)
            flat_is_masked = is_masked.view(-1)
            masked_indices = flat_is_masked.nonzero(as_tuple=True)[0]

            if len(masked_indices) == 0:
                break

            masked_confidences = flat_confidence[masked_indices]
            num_to_unmask = min(num_to_unmask, len(masked_indices))

            _, top_relative_indices = masked_confidences.topk(num_to_unmask)
            unmask_indices = masked_indices[top_relative_indices]

            flat_tokens = tokens.view(-1)
            flat_sampled = sampled.view(-1)
            flat_tokens[unmask_indices] = flat_sampled[unmask_indices]
            flat_is_masked[unmask_indices] = False

            tokens = flat_tokens.view(batch_size, total_len)
            is_masked = flat_is_masked.view(batch_size, total_len)

        return tokens

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

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")
        return logits


if __name__ == "__main__":
    print("DiscreteDiffusionSampler module loaded successfully!")
    print("\nFeatures:")
    print("  - sample(): Basic iterative unmasking")
    print("  - sample_with_guidance(): Controlled generation")
    print("  - infill(): Fill in between prefix and suffix")
