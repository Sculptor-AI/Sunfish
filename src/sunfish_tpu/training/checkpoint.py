"""Target-sharded exact-tree Orbax initialization for Sunfish."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import flax
import jax
from kauldron import kd
import orbax.checkpoint as ocp


@dataclasses.dataclass(frozen=True)
class ShardedOrbaxInitLoader(kd.ckpts.InitTransform):
    """Restore a bare Gemma parameter tree directly into target shardings.

    The checkpoint at ``path`` must contain the exact non-LoRA parameter tree
    rooted at ``model_param_path``.  LoRA leaves are preserved from model
    initialization.  Base arrays are released before restore, preventing a
    transient second 8B model in HBM.  Unlike Google's current SFT helper, this
    loader never reconstructs the full checkpoint in per-host CPU memory.
    """

    path: str
    model_param_path: str = "gemma_network.gemma_model"

    def transform(self, state: Any) -> Any:
        existing = kd.kontext.get_by_path(state.params, self.model_param_path)
        existing_mutable = flax.core.unfreeze(existing)
        flat = flax.traverse_util.flatten_dict(existing_mutable)
        lora_flat = {
            path: value for path, value in flat.items() if "lora" in path
        }
        base_flat = {
            path: value for path, value in flat.items() if "lora" not in path
        }
        if not base_flat:
            raise ValueError(f"no base parameters found at {self.model_param_path}")
        target_base = flax.traverse_util.unflatten_dict(base_flat)
        abstract_base = jax.tree.map(ocp.utils.to_shape_dtype_struct, target_base)

        # Keep only target metadata and the small trainable adapters live.
        for value in base_flat.values():
            if isinstance(value, jax.Array):
                value.delete()

        checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        try:
            restored_base = checkpointer.restore(
                self.path,
                args=ocp.args.StandardRestore(abstract_base),
            )
            checkpointer.wait_until_finished()
        finally:
            checkpointer.close()

        _validate_exact_tree(abstract_base, restored_base)
        restored_flat = flax.traverse_util.flatten_dict(restored_base)
        overlap = set(restored_flat) & set(lora_flat)
        if overlap:
            raise ValueError(f"checkpoint unexpectedly contains LoRA paths: {sorted(overlap)}")
        merged = flax.traverse_util.unflatten_dict({**restored_flat, **lora_flat})
        params = copy.copy(state.params)
        kd.kontext.set_by_path(params, self.model_param_path, merged)
        return dataclasses.replace(state, params=params)


@dataclasses.dataclass(frozen=True)
class ShardedKauldronParamsInitLoader(kd.ckpts.InitTransform):
    """Promote one pinned Kauldron checkpoint without its run state.

    Kauldron's partial loader is the source-of-truth for its composite
    checkpoint layout. Before invoking it, this adapter replaces the target
    model subtree with shape/dtype/sharding metadata and releases the random
    arrays. That prevents a transient second replicated 8B model while still
    restoring only parameters (never the source optimizer, cursor, or step).
    """

    workdir: str
    step: int
    model_param_path: str = "gemma_network.gemma_model"

    def transform(self, state: Any) -> Any:
        if self.step < 0:
            raise ValueError("Kauldron promotion requires an explicit step")
        existing = kd.kontext.get_by_path(state.params, self.model_param_path)
        abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, existing)
        for value in jax.tree.leaves(existing):
            if isinstance(value, jax.Array):
                value.delete()

        params = _copy_set_path(
            state.params, self.model_param_path.split("."), abstract
        )
        abstract_state = dataclasses.replace(state, params=params)
        full_path = f"params.{self.model_param_path}"
        loader = kd.ckpts.PartialKauldronLoader(
            workdir=self.workdir,
            new_to_old={full_path: full_path},
            step=self.step,
        )
        try:
            restored_state = loader.transform(abstract_state)
        finally:
            loader.close()

        restored = kd.kontext.get_by_path(
            restored_state.params, self.model_param_path
        )
        _validate_exact_tree(abstract, restored)
        return restored_state


def _validate_exact_tree(expected: Any, restored: Any) -> None:
    expected_with_paths, expected_tree = jax.tree.flatten_with_path(expected)
    restored_with_paths, restored_tree = jax.tree.flatten_with_path(restored)
    if expected_tree != restored_tree:
        raise ValueError("initial checkpoint parameter tree does not match the model")
    for (expected_path, expected_leaf), (restored_path, restored_leaf) in zip(
        expected_with_paths, restored_with_paths, strict=True
    ):
        if expected_path != restored_path:
            raise ValueError("initial checkpoint leaf paths differ from the model")
        if expected_leaf.shape != restored_leaf.shape:
            raise ValueError(
                f"checkpoint shape mismatch at {expected_path}: "
                f"{restored_leaf.shape} vs {expected_leaf.shape}"
            )
        if expected_leaf.dtype != restored_leaf.dtype:
            raise ValueError(
                f"checkpoint dtype mismatch at {expected_path}: "
                f"{restored_leaf.dtype} vs {expected_leaf.dtype}"
            )
        expected_sharding = getattr(expected_leaf, "sharding", None)
        restored_sharding = getattr(restored_leaf, "sharding", None)
        if expected_sharding is not None and restored_sharding != expected_sharding:
            raise ValueError(f"checkpoint target sharding mismatch at {expected_path}")


def _copy_set_path(mapping: Any, parts: list[str], value: Any) -> Any:
    """Copy only mappings along one concrete path and replace its leaf."""
    if not parts:
        return value
    if not isinstance(mapping, dict):
        if isinstance(mapping, flax.core.FrozenDict):
            mutable = dict(mapping)
        else:
            raise TypeError(f"checkpoint path traverses non-mapping {type(mapping)}")
    else:
        mutable = dict(mapping)
    head, *tail = parts
    if head not in mutable:
        raise KeyError(f"checkpoint model path component {head!r} is missing")
    mutable[head] = _copy_set_path(mutable[head], tail, value)
    return mutable
