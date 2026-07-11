"""Small, dependency-free reference pieces of the DiffusionGemma sampler.

This module is intentionally simple enough to use as an oracle when comparing
JAX, PyTorch, and MLX implementations. It follows the public entropy-bound and
adaptive-stopping definitions in Hugging Face's Apache-2.0 implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py
It is not intended to be a fast runtime.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Sequence


def linear_temperature(
    *, remaining_step: int, max_steps: int, minimum: float, maximum: float
) -> float:
    """Return the temperature for a reverse-diffusion step.

    `remaining_step` counts down from `max_steps` to 1, matching the upstream
    generation loop.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 1 <= remaining_step <= max_steps:
        raise ValueError("remaining_step must be between 1 and max_steps")
    if minimum < 0 or maximum <= minimum:
        raise ValueError("temperature bounds must satisfy 0 <= minimum < maximum")
    return minimum + (maximum - minimum) * (remaining_step / max_steps)


def entropy_bound_mask(entropies: Sequence[float], *, bound: float) -> tuple[bool, ...]:
    """Select lowest-entropy positions under the joint-information bound.

    For sorted entropies e_1..e_k, position k is accepted when
    `sum(e_1..e_k) - e_k <= bound`. At least one position is accepted for a
    non-empty canvas because the first position has a zero left-hand side.
    """
    if bound <= 0:
        raise ValueError("bound must be positive")
    if any(not math.isfinite(value) or value < 0 for value in entropies):
        raise ValueError("entropies must be finite and non-negative")

    accepted = [False] * len(entropies)
    cumulative = 0.0
    for index in sorted(range(len(entropies)), key=entropies.__getitem__):
        entropy = entropies[index]
        cumulative += entropy
        if cumulative - entropy <= bound:
            accepted[index] = True
    return tuple(accepted)


def accept_canvas(
    current: Sequence[int],
    denoised: Sequence[int],
    entropies: Sequence[float],
    *,
    bound: float,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Accept selected denoiser tokens and retain old tokens elsewhere."""
    if not (len(current) == len(denoised) == len(entropies)):
        raise ValueError("current, denoised, and entropies must have equal length")
    mask = entropy_bound_mask(entropies, bound=bound)
    accepted = tuple(new if keep else old for old, new, keep in zip(current, denoised, mask))
    return accepted, mask


def renoise_canvas(
    accepted: Sequence[int],
    accepted_mask: Sequence[bool],
    *,
    vocab_size: int,
    rng: random.Random,
) -> tuple[int, ...]:
    """Replace every unaccepted position with uniform categorical noise."""
    if len(accepted) != len(accepted_mask):
        raise ValueError("accepted and accepted_mask must have equal length")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return tuple(
        token if keep else rng.randrange(vocab_size)
        for token, keep in zip(accepted, accepted_mask)
    )


class StableAndConfidentStopper:
    """Stateful reference for DiffusionGemma adaptive stopping."""

    def __init__(self, *, stability_steps: int, mean_entropy_threshold: float):
        if stability_steps < 0:
            raise ValueError("stability_steps must be non-negative")
        if mean_entropy_threshold <= 0:
            raise ValueError("mean_entropy_threshold must be positive")
        self.stability_steps = stability_steps
        self.mean_entropy_threshold = mean_entropy_threshold
        self._history: deque[tuple[int, ...]] = deque(maxlen=stability_steps)

    def reset(self) -> None:
        self._history.clear()

    def __call__(self, argmax_canvas: Sequence[int], entropies: Sequence[float]) -> bool:
        if len(argmax_canvas) != len(entropies):
            raise ValueError("argmax_canvas and entropies must have equal length")
        if not entropies:
            raise ValueError("entropies must not be empty")
        if any(not math.isfinite(value) or value < 0 for value in entropies):
            raise ValueError("entropies must be finite and non-negative")

        canvas = tuple(argmax_canvas)
        stable = self.stability_steps == 0 or (
            len(self._history) == self.stability_steps
            and all(previous == canvas for previous in self._history)
        )
        confident = sum(entropies) / len(entropies) < self.mean_entropy_threshold
        self._history.append(canvas)
        return stable and confident
