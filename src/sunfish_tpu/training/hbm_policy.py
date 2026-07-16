"""Dependency-free HBM policy for each training phase.

The policy is intentionally separate from the Kauldron config so deployment
validation can inspect it without importing JAX, Flax, or Gemma.  Partial
update phases must not make the full frozen gradient tree observable through a
metric: doing so keeps roughly the whole frozen model gradient live in the
compiled step even though the optimizer discards it.
"""

from __future__ import annotations


_TRAINABLE_GRADIENT_GLOBS: dict[str, tuple[str, ...] | None] = {
    "smoke": ("**.lora.**",),
    "router": (
        "**.router_logits.**",
        "**.per_expert_scale",
        "**.router_scale",
    ),
    "lora": ("**.lora.**",),
    # Full training intentionally observes the complete gradient tree.
    "full": None,
}

_REMATERIALIZED_PHASES = frozenset({"lora", "full"})


def trainable_gradient_globs(phase: str) -> tuple[str, ...] | None:
    """Return disjoint relative gradient globs, or ``None`` for a full tree."""
    try:
        return _TRAINABLE_GRADIENT_GLOBS[phase]
    except KeyError as error:
        raise ValueError(f"unsupported training phase {phase!r}") from error


def rematerialize_transformer_blocks(phase: str) -> bool:
    """Whether the phase pays compute to avoid retaining block activations."""
    if phase not in _TRAINABLE_GRADIENT_GLOBS:
        raise ValueError(f"unsupported training phase {phase!r}")
    return phase in _REMATERIALIZED_PHASES
