"""Sunfish LoRA including the fused ragged-MoE expert banks."""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Sequence
from typing import Any

import flax
from flax import linen as nn
from gemma import peft
from gemma.diffusion.hackable_diffusion_adapter.hd import lora as upstream_lora
from gemma.gm.nn.gemma4 import _moe
import jax
import jax.numpy as jnp
from kauldron import kontext
import numpy as np

SUNFISH_DENSE_TARGETS = (
    r"layer_\d+/attn(?:/|$)",
    r"layer_\d+/mlp2(?:/|$)",
    r"self_conditioner(?:/|$)",
)
SUNFISH_EXPERT_TARGETS = (
    r"layer_\d+/mlp/(gating_einsum|linear)$",
)


class SunfishLoRA(nn.Module):
    """LoRA wrapper covering attention, shared MLP, SC, and every expert.

    Google's ``all-linear`` adapter does not see ``MoERagged._Weight`` because
    expert banks are raw 3-D/4-D parameter providers rather than Einsum
    modules.  This wrapper delegates supported dense layers to the upstream
    adapter and adds a batched low-rank decomposition per retained expert.
    """

    _: dataclasses.KW_ONLY
    rank: int
    model: nn.Module
    dtype: jnp.dtype = jnp.bfloat16
    dense_target_modules: Sequence[str] = SUNFISH_DENSE_TARGETS
    expert_target_modules: Sequence[str] = SUNFISH_EXPERT_TARGETS
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.scope is not None:
            nn.share_scope(self, self.model)

    def _interceptor(self):
        replace = functools.partial(
            _replace_module,
            rank=self.rank,
            dtype=self.dtype,
            dense_target_modules=self.dense_target_modules,
            expert_target_modules=self.expert_target_modules,
            verbose=self.verbose,
        )
        return peft.ModuleInterceptor(replace)

    @nn.compact
    def __call__(self, *args, **kwargs):
        with self._interceptor():
            return self.model(*args, **kwargs)

    @nn.compact
    def encoder_call(self, *args, **kwargs):
        with self._interceptor():
            return self.model.encoder_call(*args, **kwargs)

    @nn.compact
    def init_cache(self, *args, **kwargs):
        with self._interceptor():
            return self.model.init_cache(*args, **kwargs)

    def __kontext_keys__(self) -> dict[str, str]:
        return kontext.get_keypaths(self.model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class ExpertLoRAWeight(nn.Module):
    """Transparent wrapper adding a per-expert delta to one fused bank."""

    _: dataclasses.KW_ONLY
    rank: int
    dtype: jnp.dtype
    wrapped: _moe._Weight

    def __post_init__(self):
        super().__post_init__()
        if self.scope is not None:
            nn.share_scope(self, self.wrapped)

    @nn.compact
    def __call__(self) -> jax.Array:
        adapter = ExpertLoRAAdapter(
            name="lora",
            rank=self.rank,
            target_shape=self.wrapped.shape,
            dtype=self.dtype,
        )
        return self.wrapped() + adapter()


class ExpertLoRAAdapter(nn.Module):
    """Batched low-rank delta whose leading dimension is the expert id."""

    _: dataclasses.KW_ONLY
    rank: int
    target_shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.bfloat16
    a_init: nn.initializers.Initializer = nn.initializers.kaiming_uniform()
    b_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self) -> jax.Array:
        if len(self.target_shape) == 4:
            experts, gates, hidden, features = self.target_shape
            if gates != 2:
                raise ValueError(f"unsupported expert gate shape {self.target_shape}")
            a = self.param(
                "a", self.a_init, (experts, features, self.rank), self.dtype
            )
            b = self.param(
                "b", self.b_init, (experts, self.rank, gates * hidden), self.dtype
            )
        elif len(self.target_shape) == 3:
            experts, hidden, features = self.target_shape
            a = self.param(
                "a", self.a_init, (experts, hidden, self.rank), self.dtype
            )
            b = self.param(
                "b", self.b_init, (experts, self.rank, features), self.dtype
            )
        else:
            raise ValueError(f"unsupported fused expert shape {self.target_shape}")
        return expert_lora_delta(a, b, self.target_shape)


def expert_lora_delta(
    a: jax.Array, b: jax.Array, target_shape: tuple[int, ...]
) -> jax.Array:
    """Contract rank independently for each expert and restore bank layout."""
    if len(target_shape) == 4:
        experts, gates, hidden, features = target_shape
        expected_a = (experts, features, a.shape[-1])
        expected_b = (experts, a.shape[-1], gates * hidden)
        if a.shape != expected_a or b.shape != expected_b:
            raise ValueError(
                f"expert gate LoRA shapes {a.shape}/{b.shape} do not target {target_shape}"
            )
        flat = jnp.einsum("efr,erh->efh", a, b)
        return flat.reshape(experts, features, gates, hidden).transpose(0, 2, 3, 1)
    if len(target_shape) == 3:
        experts, hidden, features = target_shape
        expected_a = (experts, hidden, a.shape[-1])
        expected_b = (experts, a.shape[-1], features)
        if a.shape != expected_a or b.shape != expected_b:
            raise ValueError(
                f"expert down LoRA shapes {a.shape}/{b.shape} do not target {target_shape}"
            )
        return jnp.einsum("ehr,erf->ehf", a, b)
    raise ValueError(f"unsupported fused expert shape {target_shape}")


def fuse_lora_params(params: dict[str, Any]) -> dict[str, Any]:
    """Fuse both upstream dense and Sunfish batched-expert LoRA leaves."""
    mutable = flax.core.unfreeze(params)
    flat = flax.traverse_util.flatten_dict(mutable)
    base_flat = {path: value for path, value in flat.items() if "lora" not in path}
    pairs: dict[tuple[str, ...], dict[str, jax.Array]] = {}
    for path, value in flat.items():
        if len(path) >= 2 and path[-2] == "lora" and path[-1] in {"a", "b"}:
            pairs.setdefault(path[:-2], {})[path[-1]] = value

    for parent, pair in pairs.items():
        if set(pair) != {"a", "b"}:
            raise ValueError(f"incomplete LoRA pair at {'/'.join(parent)}")
        base_key = upstream_lora._find_base_weight_key(parent, base_flat)  # pylint: disable=protected-access
        if base_key is None:
            raise KeyError(f"no base weight for LoRA adapter at {'/'.join(parent)}")
        base = base_flat[base_key]
        a, b = pair["a"], pair["b"]
        if (
            a.ndim >= 3
            and b.ndim >= 3
            and a.shape[0] == b.shape[0]
            and base.ndim in {3, 4}
            and base.shape[0] == a.shape[0]
        ):
            delta = expert_lora_delta(a, b, base.shape)
        else:
            delta = upstream_lora._compute_lora_delta(  # pylint: disable=protected-access
                a, b, target_shape=base.shape
            )
        base_flat[base_key] = base + delta.astype(base.dtype)
    return flax.traverse_util.unflatten_dict(base_flat)


def _replace_module(
    module: nn.Module,
    *,
    rank: int,
    dtype: np.dtype,
    dense_target_modules: Sequence[str],
    expert_target_modules: Sequence[str],
    verbose: bool,
) -> nn.Module:
    if isinstance(module, _moe._Weight) and _matches_path(
        module, expert_target_modules
    ):
        return ExpertLoRAWeight(rank=rank, dtype=dtype, wrapped=module)
    return upstream_lora._replace_by_lora(  # pylint: disable=protected-access
        module,
        rank=rank,
        dtype=dtype,
        verbose=verbose,
        target_modules=dense_target_modules,
    )


def _matches_path(module: nn.Module, patterns: Sequence[str]) -> bool:
    try:
        path = "/".join(module.path)
    except AttributeError:
        path = module.name or ""
    return any(re.search(pattern, path) for pattern in patterns)
