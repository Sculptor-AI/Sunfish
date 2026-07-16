"""Pinned Gemma block rematerialization used by recovery/full training."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any


_BLOCK_CALL_PARAMETERS = (
    "self",
    "x",
    "segment_pos",
    "cache",
    "attn_mask",
    "per_layer_input",
    "kv_shared_cache",
    "skip_sliding_mask",
)
_REQUIRED_PARAMETER_COUNT = 5
_REMAT_MARKER = "__sunfish_upstream_block_remat__"


def validate_block_call_signature(call: Callable[..., Any]) -> None:
    """Fail closed unless Gemma still exposes the pinned Block boundary."""
    parameters = tuple(
        inspect.signature(call, follow_wrapped=False).parameters.values()
    )
    names = tuple(parameter.name for parameter in parameters)
    if names != _BLOCK_CALL_PARAMETERS:
        raise RuntimeError(
            "pinned Gemma Block.__call__ signature drifted: " + repr(names)
        )
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    ):
        raise RuntimeError("pinned Gemma Block.__call__ parameter kinds drifted")
    expected_defaults = (
        inspect.Parameter.empty,
    ) * _REQUIRED_PARAMETER_COUNT + (None, None, False)
    defaults = tuple(parameter.default for parameter in parameters)
    if defaults != expected_defaults:
        raise RuntimeError(
            "pinned Gemma Block.__call__ defaults drifted: " + repr(defaults)
        )


def enable_upstream_block_rematerialization() -> None:
    """Install Gemma's pinned SFT remat at the Transformer Block boundary.

    Google's full-SFT config at Gemma revision
    ``09e7b48ae88720f6236b8266c7213eb51bb62b87`` applies
    ``nn.remat`` to ``_modules.Block.__call__`` with the
    ``nothing_saveable`` checkpoint policy and ``static_argnums=7``.  Gemma
    does not expose a per-model block factory, so the upstream mechanism is a
    process-global, idempotent patch.  Sunfish launches each phase in a fresh
    Python process; only recovery (LoRA) and full training enable it.
    """
    import jax  # Imported only after distributed initialization by the launcher.
    from flax import linen as nn
    from gemma.gm.nn.gemma4 import _modules

    current_call = _modules.Block.__call__
    if getattr(current_call, _REMAT_MARKER, False):
        return
    validate_block_call_signature(current_call)
    original_call = current_call

    @functools.partial(
        nn.remat,
        policy=jax.checkpoint_policies.nothing_saveable,
        static_argnums=7,
    )
    def rematted_call_fn(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
        per_layer_input,
        kv_shared_cache,
        skip_sliding_mask,
    ):
        return original_call(
            self,
            x,
            segment_pos,
            cache,
            attn_mask,
            per_layer_input,
            kv_shared_cache,
            skip_sliding_mask,
        )

    def new_call(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
        per_layer_input=None,
        kv_shared_cache=None,
        skip_sliding_mask=False,
    ):
        return rematted_call_fn(
            self,
            x,
            segment_pos,
            cache,
            attn_mask,
            per_layer_input,
            kv_shared_cache,
            skip_sliding_mask,
        )

    setattr(new_call, _REMAT_MARKER, True)
    _modules.Block.__call__ = new_call
