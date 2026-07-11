"""Pure path/shape policy for Phase-B FSDP partition decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def partition_axis_for_path(
    path: str,
    shape: Sequence[int],
    *,
    data_axis_size: int,
    itemsize: int,
    min_size_bytes: int = 1 << 20,
) -> int | None:
    """Choose the tensor axis sharded over the Phase-B ``data`` mesh.

    Expert banks and their scales are path-forced to axis zero so a 32-wide
    mesh owns exactly one expert per device. Dense tensors prefer row sharding;
    unusual attention layouts fall back to their largest divisible dimension.
    Norms, scalars, and small tensors remain replicated.
    """
    if data_axis_size <= 1 or not shape:
        return None
    padded_path = f".{path}."
    is_expert_leaf = (
        ".mlp." in padded_path
        and any(
            marker in padded_path
            for marker in (".gating_einsum.", ".linear.", ".per_expert_scale.")
        )
    )
    if is_expert_leaf and shape[0] % data_axis_size == 0:
        return 0

    nbytes = math.prod(shape) * itemsize
    if nbytes < min_size_bytes or len(shape) < 2:
        return None
    if shape[0] % data_axis_size == 0:
        return 0
    candidates = [
        axis for axis, size in enumerate(shape) if size % data_axis_size == 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda axis: shape[axis])


def resolve_phase_b_mesh(
    *, global_device_count: int, num_experts: int
) -> tuple[tuple[int, ...], tuple[str, ...], int]:
    """Return mesh shape/names and the expert-sharding axis size.

    If TPU megacore exposes more devices than retained experts, extra devices
    become independent replicas while the expert axis stays 32 wide. If the
    slice is smaller, each device owns an equal number of experts.
    """
    if global_device_count <= 0 or num_experts <= 0:
        raise ValueError("device and expert counts must be positive")
    if global_device_count <= num_experts:
        if num_experts % global_device_count:
            raise ValueError("expert count must divide evenly over Phase-B devices")
        return (global_device_count,), ("data",), global_device_count
    if global_device_count % num_experts:
        raise ValueError("extra Phase-B devices must form complete expert replicas")
    replicas = global_device_count // num_experts
    return (replicas, num_experts), ("replica", "data"), num_experts
