"""Measured-device mesh and sharding strategy for Sunfish training."""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from kauldron import kd
import numpy as np

from sunfish_tpu.training.sharding_policy import (
    partition_axis_for_path,
    resolve_phase_b_mesh,
)
from sunfish_tpu.training.spec import HarnessConfig, Phase


@dataclasses.dataclass(frozen=True, kw_only=True)
class SunfishShardingStrategy(kd.sharding.ShardingStrategy):
    """Kauldron strategy whose scalar state uses the same named mesh."""

    replicated: NamedSharding = dataclasses.field(repr=False)

    @property
    def state(self):
        return kd.train.TrainState(
            step=self.replicated,
            params=self.params,
            collections=self.collections,
            opt_state=self.opt_state,
        )


@dataclasses.dataclass(frozen=True)
class PhaseBTreeSharding:
    """Path-aware callable used for params and matching optimizer leaves."""

    mesh: Mesh
    data_axis_size: int
    min_size_bytes: int = 1 << 20

    def __call__(self, tree: Any) -> Any:
        leaves_with_paths, treedef = jax.tree.flatten_with_path(tree)
        shardings = []
        for key_path, value in leaves_with_paths:
            path = ".".join(_path_component(component) for component in key_path)
            dtype = np.dtype(value.dtype)
            axis = partition_axis_for_path(
                path,
                value.shape,
                data_axis_size=self.data_axis_size,
                itemsize=dtype.itemsize,
                min_size_bytes=self.min_size_bytes,
            )
            spec = [None] * len(value.shape)
            if axis is not None:
                spec[axis] = "data"
            shardings.append(NamedSharding(self.mesh, P(*spec)))
        return jax.tree.unflatten(treedef, shardings)



def _requires_sharded_params(resolved_phase: Phase) -> bool:
    """Params/opt_state must be path/shape-sharded (PhaseBTreeSharding) rather
    than replicated for every phase.

    Phase.FULL always sharded. The readiness/adapter phases (SMOKE/ROUTER/LORA)
    were originally replicated on the assumption that each v4 chip's 32GB HBM
    holds the full 16.2GB bf16 base with room to spare; the first real gate-4
    training smoke on v4-64 disproved that -- the replicated base plus the
    bidirectional diffusion forward's activations exhaust HBM (device buffer
    OOM, ~8MB free). Sharding params along the phase mesh (experts+dense under
    Phase.FULL's mesh, or FSDP along the data axis under the readiness 1-D
    mesh) drops per-chip param bytes ~device-count-fold and fits comfortably.
    This is a memory-layout change only -- identical to the mechanism Phase.FULL
    already relies on -- and does not alter any value a forward/backward pass
    computes. CPU/GPU dev runs (SUNFISH_ALLOW_NON_TPU=1) also require it because
    a single host cannot replicate the base per simulated device.
    """
    return (
        resolved_phase in {Phase.SMOKE, Phase.ROUTER, Phase.LORA}
        or os.environ.get("SUNFISH_ALLOW_NON_TPU") == "1"
    )


def make_training_sharding(config: HarnessConfig) -> SunfishShardingStrategy:
    """Build the approved Phase-A or Phase-B policy after distributed init."""
    devices = sorted(
        jax.devices(), key=lambda device: (int(device.process_index), int(device.id))
    )
    if config.run.phase is Phase.FULL:
        mesh_shape, axis_names, data_axis_size = resolve_phase_b_mesh(
            global_device_count=len(devices), num_experts=config.model.num_experts
        )
    else:
        mesh_shape = (len(devices),)
        axis_names = ("data",)
        data_axis_size = len(devices)
    mesh = Mesh(np.asarray(devices, dtype=object).reshape(mesh_shape), axis_names)
    replicated = NamedSharding(mesh, P())
    batch_axes: str | tuple[str, ...]
    batch_axes = axis_names[0] if len(axis_names) == 1 else tuple(axis_names)
    batch = NamedSharding(mesh, P(batch_axes))

    if config.run.phase is Phase.FULL or _requires_sharded_params(config.run.phase):
        tree_policy = PhaseBTreeSharding(mesh=mesh, data_axis_size=data_axis_size)
        params: Any = tree_policy
        opt_state: Any = tree_policy
    else:
        params = replicated
        opt_state = replicated
    return SunfishShardingStrategy(
        batch=batch,
        params=params,
        collections=replicated,
        opt_state=opt_state,
        aux=replicated,
        replicated=replicated,
    )


def make_training_sharding_for(
    *, phase: str, num_experts: int
) -> SunfishShardingStrategy:
    """Konfig-friendly primitive-argument wrapper around the mesh builder."""
    devices = sorted(
        jax.devices(), key=lambda device: (int(device.process_index), int(device.id))
    )
    resolved_phase = Phase(phase)
    if resolved_phase is Phase.FULL:
        mesh_shape, axis_names, data_axis_size = resolve_phase_b_mesh(
            global_device_count=len(devices), num_experts=num_experts
        )
    else:
        mesh_shape = (len(devices),)
        axis_names = ("data",)
        data_axis_size = len(devices)
    mesh = Mesh(np.asarray(devices, dtype=object).reshape(mesh_shape), axis_names)
    replicated = NamedSharding(mesh, P())
    batch_axes: str | tuple[str, ...]
    batch_axes = axis_names[0] if len(axis_names) == 1 else tuple(axis_names)
    batch = NamedSharding(mesh, P(batch_axes))
    if resolved_phase is Phase.FULL or _requires_sharded_params(resolved_phase):
        tree_policy = PhaseBTreeSharding(mesh=mesh, data_axis_size=data_axis_size)
        params: Any = tree_policy
        opt_state: Any = tree_policy
    else:
        params = replicated
        opt_state = replicated
    return SunfishShardingStrategy(
        batch=batch,
        params=params,
        collections=replicated,
        opt_state=opt_state,
        aux=replicated,
        replicated=replicated,
    )


def _path_component(component: jax.tree_util.KeyEntry) -> str:
    if isinstance(component, jax.tree_util.DictKey):
        return str(component.key)
    if isinstance(component, jax.tree_util.SequenceKey):
        return str(component.idx)
    if isinstance(component, jax.tree_util.GetAttrKey):
        return str(component.name)
    return str(component)
