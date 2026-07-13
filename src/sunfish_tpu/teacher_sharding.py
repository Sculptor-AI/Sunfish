"""Forward-only full-128-expert teacher sharding for calibration/traces."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any


def resolve_teacher_mesh(
    *,
    global_device_count: int,
    process_count: int,
    local_device_count: int,
    source_experts: int = 128,
) -> tuple[tuple[int, int], tuple[str, str]]:
    """Place one data replica per host and the expert collective within host."""
    if min(global_device_count, process_count, local_device_count, source_experts) <= 0:
        raise ValueError("teacher topology counts must be positive")
    if process_count * local_device_count != global_device_count:
        raise ValueError("process_count * local_device_count must equal global devices")
    if source_experts % local_device_count:
        raise ValueError("source experts must divide evenly over local expert devices")
    return (process_count, local_device_count), ("data", "expert")


def teacher_partition_axis(path: str, shape: Sequence[int]) -> int | None:
    """Shard only the three large expert-bank/scaling leaves on expert axis."""
    padded = f".{path}."
    if ".mlp." not in padded or not shape:
        return None
    if any(
        marker in padded
        for marker in (".gating_einsum.", ".linear.", ".per_expert_scale.")
    ):
        return 0
    return None


@dataclasses.dataclass(frozen=True)
class TeacherTreeSharding:
    """Path-aware sharding callable for target ShapeDtypeStruct trees."""

    mesh: Any

    def __call__(self, tree: Any) -> Any:
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P

        leaves_with_paths, treedef = jax.tree.flatten_with_path(tree)
        shardings = []
        for key_path, value in leaves_with_paths:
            path = ".".join(_path_component(component) for component in key_path)
            axis = teacher_partition_axis(path, value.shape)
            spec = [None] * len(value.shape)
            if axis is not None:
                spec[axis] = "expert"
            shardings.append(NamedSharding(self.mesh, P(*spec)))
        return jax.tree.unflatten(treedef, shardings)


def make_teacher_mesh_and_shardings(jax: Any, np: Any):
    """Build measured teacher mesh, param policy, batch and replicated sharding."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = sorted(
        jax.devices(), key=lambda device: (int(device.process_index), int(device.id))
    )
    mesh_shape, axis_names = resolve_teacher_mesh(
        global_device_count=len(devices),
        process_count=int(jax.process_count()),
        local_device_count=int(jax.local_device_count()),
    )
    # Sorting by process then device makes every row one physical host. The
    # MoE all-to-all therefore stays within a host; data replicas span hosts.
    mesh = Mesh(np.asarray(devices, dtype=object).reshape(mesh_shape), axis_names)
    return {
        "mesh": mesh,
        "params": TeacherTreeSharding(mesh),
        "batch": NamedSharding(mesh, P("data", None)),
        "batch_vector": NamedSharding(mesh, P("data")),
        "replicated": NamedSharding(mesh, P()),
        "data_axis_size": mesh_shape[0],
        "expert_axis_size": mesh_shape[1],
    }


def _path_component(component: Any) -> str:
    try:
        import jax
    except ModuleNotFoundError:
        return str(component)
    if isinstance(component, jax.tree_util.DictKey):
        return str(component.key)
    if isinstance(component, jax.tree_util.SequenceKey):
        return str(component.idx)
    if isinstance(component, jax.tree_util.GetAttrKey):
        return str(component.name)
    return str(component)
