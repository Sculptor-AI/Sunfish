"""Training metrics whose tree access is explicit enough for XLA DCE."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import flax
import jax
import jax.numpy as jnp
from kauldron import kd


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SelectedTreeNorm(kd.metrics.Metric):
    """L2 norm over only the leaves selected by disjoint Kontext globs.

    Kauldron's default ``TreeReduce(Norm(tensor="grads"))`` consumes every
    gradient leaf.  In a partial-update phase that makes otherwise-dead frozen
    gradients observable and can keep roughly a full model-sized gradient tree
    live inside the compiled train step.  This metric resolves the root once,
    filters it with Kontext before numerical work, and touches only selected
    leaves.

    ``include`` paths are relative to ``tree`` and must be disjoint.  Sunfish's
    phase policy uses one ``lora`` subtree glob, or the three exact router
    parameter families.
    """

    tree: kd.kontext.Key = "grads"
    include: tuple[str, ...] = ()

    @flax.struct.dataclass
    class State(kd.metrics.State):
        sum_squares: jax.Array

        @classmethod
        def empty(cls) -> "SelectedTreeNorm.State":
            return cls(sum_squares=jnp.asarray(0.0, dtype=jnp.float32))

        def merge(self, other: "SelectedTreeNorm.State") -> "SelectedTreeNorm.State":
            return type(self)(sum_squares=self.sum_squares + other.sum_squares)

        def compute(self) -> jax.Array:
            return jnp.sqrt(self.sum_squares)

    def __post_init__(self) -> None:
        if not self.include:
            raise ValueError("SelectedTreeNorm.include must not be empty")
        if len(set(self.include)) != len(self.include):
            raise ValueError("SelectedTreeNorm.include paths must be unique")

    def _resolve_kwargs(self, context: Any) -> dict[str, Any]:
        root = kd.kontext.resolve_from_keypaths(context, {"tree": self.tree})[
            "tree"
        ]
        root = flax.core.unfreeze(root)
        # Filtering happens before get_state.  Consequently the numerical
        # metric has no references to frozen gradient tracers for XLA to keep.
        selected = tuple(
            kd.kontext.filter_by_path(root, path) for path in self.include
        )
        return {"trees": selected}

    def get_state(self, trees: Sequence[Any]) -> "SelectedTreeNorm.State":
        leaves = [leaf for tree in trees for leaf in jax.tree.leaves(tree)]
        if not leaves:
            raise KeyError(
                "SelectedTreeNorm found no leaves for " + ", ".join(self.include)
            )
        sum_squares = jnp.asarray(0.0, dtype=jnp.float32)
        for leaf in leaves:
            value = jnp.asarray(leaf, dtype=jnp.float32)
            sum_squares = sum_squares + jnp.sum(jnp.square(value))
        return self.State(sum_squares=sum_squares)
