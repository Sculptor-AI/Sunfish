"""Distributed sharded Orbax save/restore and exact-resume smoke test.

Every process must run this program with the same workdir and run ID.  It
initializes distributed JAX before importing backend-adjacent libraries,
builds the approved Phase-B global data mesh, creates arrays directly in their shardings,
saves model/optimizer/RNG/loader state collectively, restores with explicit
shardings, and compares one resumed optimizer update with an uninterrupted
control using addressable shards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from sunfish_tpu.tpu_preflight import (
    _topology_checks,
    initialize_distributed_jax,
    report,
)
from sunfish_tpu.source_identity import (
    normalize_source_identity,
    require_launcher_run_id,
    source_identity_from_environment,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def verify_checkpoint_evidence(
    hosts: list[dict[str, object]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, object]:
    """Verify every process observed the same exact distributed round trip."""
    errors: list[str] = []
    if len(hosts) != expected_processes:
        errors.append(f"found {len(hosts)} host reports; expected {expected_processes}")
    indices = [host.get("process_index") for host in hosts]
    if sorted(indices) != list(range(expected_processes)):
        errors.append(f"process indices are {sorted(indices)}")
    destinations = {host.get("destination") for host in hosts}
    run_ids = {host.get("run_id") for host in hosts}
    if len(destinations) != 1 or None in destinations:
        errors.append("host checkpoint destinations differ")
    if len(run_ids) != 1 or None in run_ids:
        errors.append("host run IDs differ")
    required_true = (
        "ready",
        "restored_addressable_shards_exact",
        "next_loss_exact",
        "next_gradients_exact",
        "next_update_exact",
    )
    sources = [normalize_source_identity(host.get("sunfish_source")) for host in hosts]
    if any(source is None for source in sources) or len(set(sources)) != 1:
        errors.append("host source identities are missing or differ")
    for host in hosts:
        process = host.get("process_index")
        if host.get("schema_version") != 1:
            errors.append(f"process {process} has an unsupported schema")
        if host.get("process_count") != expected_processes:
            errors.append(f"process {process} reports the wrong process count")
        if host.get("global_device_count") != expected_devices:
            errors.append(f"process {process} reports the wrong global device count")
        if host.get("local_device_count") != expected_local_devices:
            errors.append(f"process {process} reports the wrong local device count")
        topology = host.get("topology")
        if not isinstance(topology, dict) or topology.get("ready") is not True:
            errors.append(f"process {process} topology did not pass")
        for key in required_true:
            if host.get(key) is not True:
                errors.append(f"process {process} {key}={host.get(key)!r}")
    return {
        "schema_version": 1,
        "gates": [5],
        "scope": "synthetic-sharded-state",
        "run_id": next(iter(run_ids), None),
        "destination": next(iter(destinations), None),
        "passed": not errors,
        "errors": errors,
        "hosts": hosts,
        "gate6_note": "real-trainer interrupted comparison is separately required",
        "sunfish_source": hosts[0].get("sunfish_source") if hosts else None,
    }


def _resolve_mesh_shape(
    *, global_device_count: int, process_count: int, data_axis_size: int
) -> tuple[int, int]:
    if global_device_count < 1 or process_count < 1:
        raise ValueError("global device and process counts must be positive")
    data = global_device_count if data_axis_size == 0 else data_axis_size
    if data != global_device_count:
        raise ValueError(
            "the Phase-B FSDP smoke requires the data axis to span every "
            f"global device ({global_device_count}), got {data}"
        )
    return data, 1


def _deterministic_shard(
    index: tuple[slice, ...] | None,
    shape: tuple[int, ...],
    *,
    numpy: Any,
    dtype: Any,
    offset: int,
) -> Any:
    """Create only the requested shard using its global row-major indices."""
    if shape == ():
        return numpy.asarray(offset, dtype=dtype)
    if index is None:
        index = tuple(slice(0, size) for size in shape)
    local_shape = tuple(part.stop - part.start for part in index)
    coordinates = numpy.indices(local_shape, dtype=numpy.int64)
    values = numpy.zeros(local_shape, dtype=numpy.int64)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        values += (coordinates[axis] + index[axis].start) * stride
        stride *= shape[axis]
    return (values + offset).astype(dtype)


def _make_array(
    jax: Any,
    numpy: Any,
    *,
    shape: tuple[int, ...],
    sharding: Any,
    dtype: Any,
    offset: int,
) -> Any:
    return jax.make_array_from_callback(
        shape,
        sharding,
        lambda index: _deterministic_shard(
            index, shape, numpy=numpy, dtype=dtype, offset=offset
        ),
        dtype=dtype,
    )


def _tree_equal_addressable(jax: Any, numpy: Any, left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if left_leaf.shape != right_leaf.shape or left_leaf.dtype != right_leaf.dtype:
            return False
        left_shards = list(left_leaf.addressable_shards)
        right_shards = list(right_leaf.addressable_shards)
        if len(left_shards) != len(right_shards):
            return False
        for left_shard, right_shard in zip(left_shards, right_shards, strict=True):
            if str(left_shard.index) != str(right_shard.index):
                return False
            if not numpy.array_equal(
                numpy.asarray(left_shard.data), numpy.asarray(right_shard.data)
            ):
                return False
    return True


def _build_state(jax: Any, jnp: Any, numpy: Any, mesh: Any) -> tuple[Any, Any]:
    from jax.sharding import NamedSharding, PartitionSpec as P

    data_size = int(mesh.shape["data"])
    features = max(8, data_size * 2)
    global_batch = data_size * 2
    replicated = NamedSharding(mesh, P())
    data_rows = NamedSharding(mesh, P("data", None))
    expert_rows = NamedSharding(mesh, P("data", None, None))

    params = {
        # These mirror docs/sharding_plan.md Phase B: dense matrices and the
        # routed-expert axis are row-sharded over the full data mesh.
        "dense_weight": _make_array(
            jax,
            numpy,
            shape=(features, features),
            sharding=data_rows,
            dtype=numpy.float32,
            offset=1,
        )
        / numpy.float32(1024.0),
        "expert_bank": _make_array(
            jax,
            numpy,
            shape=(data_size, 2, features),
            sharding=expert_rows,
            dtype=numpy.float32,
            offset=29,
        )
        / numpy.float32(4096.0),
        "bias": _make_array(
            jax,
            numpy,
            shape=(features,),
            sharding=replicated,
            dtype=numpy.float32,
            offset=3,
        )
        / numpy.float32(2048.0),
    }
    optimizer = jax.tree.map(jnp.zeros_like, params)
    state = {
        "step": _make_array(
            jax, numpy, shape=(), sharding=replicated, dtype=numpy.int32, offset=0
        ),
        "params": params,
        "optimizer": {"momentum": optimizer},
        "rng": _make_array(
            jax,
            numpy,
            shape=(2,),
            sharding=replicated,
            dtype=numpy.uint32,
            offset=0x12345678,
        ),
        "loader": {
            "manifest_hash": _make_array(
                jax,
                numpy,
                shape=(32,),
                sharding=replicated,
                dtype=numpy.uint8,
                offset=17,
            ),
            "seed": _make_array(
                jax,
                numpy,
                shape=(),
                sharding=replicated,
                dtype=numpy.uint32,
                offset=20260711,
            ),
            "epoch": _make_array(
                jax, numpy, shape=(), sharding=replicated, dtype=numpy.int32, offset=2
            ),
            "shard_sequence": _make_array(
                jax,
                numpy,
                shape=(4,),
                sharding=replicated,
                dtype=numpy.int32,
                offset=41,
            ),
            "current_shard": _make_array(
                jax, numpy, shape=(), sharding=replicated, dtype=numpy.int32, offset=1
            ),
            "record_offset": _make_array(
                jax, numpy, shape=(), sharding=replicated, dtype=numpy.int64, offset=0
            ),
            "packing_buffer": _make_array(
                jax,
                numpy,
                shape=(8,),
                sharding=replicated,
                dtype=numpy.int32,
                offset=73,
            ),
            "packing_size": _make_array(
                jax, numpy, shape=(), sharding=replicated, dtype=numpy.int32, offset=5
            ),
        },
    }
    batch_template = {
        "x": _make_array(
            jax,
            numpy,
            shape=(global_batch, features),
            sharding=data_rows,
            dtype=numpy.float32,
            offset=0,
        ),
        "y": _make_array(
            jax,
            numpy,
            shape=(global_batch, features),
            sharding=data_rows,
            dtype=numpy.float32,
            offset=11,
        ),
    }
    return state, batch_template


def _offset_batch(batch_template: Any, jnp: Any, offset: int) -> Any:
    scale = jnp.asarray(1.0 / 4096.0, dtype=jnp.float32)
    return {
        "x": (batch_template["x"] + jnp.asarray(offset, dtype=jnp.float32)) * scale,
        "y": (batch_template["y"] + jnp.asarray(offset * 3, dtype=jnp.float32)) * scale,
    }


def _update_function(jax: Any, jnp: Any):
    def update(state: Any, batch: Any) -> tuple[Any, Any, Any]:
        def loss_fn(params: Any) -> Any:
            expert_residual = jnp.mean(params["expert_bank"], axis=(0, 1))
            prediction = (
                batch["x"] @ params["dense_weight"]
                + params["bias"]
                + expert_residual
            )
            return jnp.mean(jnp.square(prediction - batch["y"]))

        loss, gradients = jax.value_and_grad(loss_fn)(state["params"])
        momentum = jax.tree.map(
            lambda old, grad: jnp.asarray(0.9, old.dtype) * old + grad,
            state["optimizer"]["momentum"],
            gradients,
        )
        params = jax.tree.map(
            lambda value, velocity: value - jnp.asarray(0.01, value.dtype) * velocity,
            state["params"],
            momentum,
        )
        next_state = {
            **state,
            "step": state["step"] + jnp.asarray(1, dtype=state["step"].dtype),
            "params": params,
            "optimizer": {"momentum": momentum},
            "rng": state["rng"]
            + jnp.asarray([0x9E3779B9, 0x7F4A7C15], dtype=jnp.uint32),
            "loader": {
                **state["loader"],
                "record_offset": state["loader"]["record_offset"]
                + jnp.asarray(batch["x"].shape[0], dtype=jnp.int64),
            },
        }
        return next_state, loss, gradients

    return jax.jit(update)


def run_smoke(
    workdir: str,
    run_id: str,
    *,
    require_tpu: bool = False,
    expected_devices: int = 0,
    expected_processes: int = 0,
    expected_local_devices: int = 0,
    data_axis_size: int = 0,
) -> dict[str, object]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, dot, underscore, or dash")
    require_launcher_run_id(run_id, required=require_tpu)

    # No backend-adjacent import may move above this call.
    jax, initialization = initialize_distributed_jax(require_distributed=require_tpu)
    import jax.numpy as jnp
    import numpy as np

    topology = [
        initialization,
        *_topology_checks(
            jax,
            jnp,
            require_tpu=require_tpu,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        ),
    ]
    topology_report = report(topology)
    if not topology_report["ready"]:
        raise RuntimeError(f"distributed topology failed: {json.dumps(topology_report)}")

    # Orbax and etils may import JAX-adjacent code, so import them only now.
    import orbax.checkpoint as ocp
    from etils import epath
    from jax.sharding import Mesh

    data_size, _ = _resolve_mesh_shape(
        global_device_count=int(jax.device_count()),
        process_count=int(jax.process_count()),
        data_axis_size=data_axis_size,
    )
    ordered_devices = sorted(
        jax.devices(),
        key=lambda device: (int(device.process_index), int(device.id)),
    )
    mesh = Mesh(
        np.asarray(ordered_devices, dtype=object).reshape(data_size),
        ("data",),
    )
    initial_state, batch_template = _build_state(jax, jnp, np, mesh)
    update = _update_function(jax, jnp)
    first_batch = _offset_batch(batch_template, jnp, 0)
    checkpoint_state, _, _ = update(initial_state, first_batch)
    jax.block_until_ready(checkpoint_state)

    destination = epath.Path(workdir) / "sunfish-checkpoint-smoke" / run_id
    abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, checkpoint_state)
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    try:
        checkpointer.save(
            destination,
            args=ocp.args.StandardSave(checkpoint_state),
        )
        checkpointer.wait_until_finished()
        restored = checkpointer.restore(
            destination,
            args=ocp.args.StandardRestore(abstract_state),
        )
    finally:
        checkpointer.close()

    restored_exact = _tree_equal_addressable(jax, np, checkpoint_state, restored)
    if not restored_exact:
        raise RuntimeError("restored addressable shards do not exactly match saved state")

    global_batch = int(batch_template["x"].shape[0])
    second_batch = _offset_batch(batch_template, jnp, global_batch)
    control_state, control_loss, control_gradients = update(checkpoint_state, second_batch)
    resumed_state, resumed_loss, resumed_gradients = update(restored, second_batch)
    jax.block_until_ready((control_state, resumed_state, control_gradients, resumed_gradients))

    loss_exact = _tree_equal_addressable(jax, np, control_loss, resumed_loss)
    gradients_exact = _tree_equal_addressable(
        jax, np, control_gradients, resumed_gradients
    )
    update_exact = _tree_equal_addressable(jax, np, control_state, resumed_state)
    if not (loss_exact and gradients_exact and update_exact):
        raise RuntimeError(
            "resumed loss, gradients, or updated state differs from uninterrupted control"
        )

    return {
        "schema_version": 1,
        "ready": True,
        "run_id": run_id,
        "destination": str(destination),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "global_device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "mesh": {"data": data_size},
        "sharding_policy": "docs/sharding_plan.md Phase B FSDP",
        "restored_addressable_shards_exact": restored_exact,
        "next_loss_exact": loss_exact,
        "next_gradients_exact": gradients_exact,
        "next_update_exact": update_exact,
        "topology": topology_report,
        "sunfish_source": source_identity_from_environment(
            required=require_tpu
        ),
    }


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _write_immutable(path: Any, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable checkpoint evidence changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def write_checkpoint_evidence(
    payload: dict[str, object],
    *,
    evidence_dir: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, object]:
    """Publish per-host evidence and an exact process-0 merged summary."""
    from etils import epath
    from jax.experimental import multihost_utils

    run_id = str(payload["run_id"])
    root = epath.Path(evidence_dir) / run_id
    process_index = int(payload["process_index"])
    _write_immutable(root / f"host-{process_index:05d}.json", payload)
    multihost_utils.sync_global_devices(f"sunfish-checkpoint-evidence-{run_id}")
    summary = None
    if process_index == 0:
        hosts = [
            json.loads((root / f"host-{process:05d}.json").read_text())
            for process in range(expected_processes)
        ]
        summary = verify_checkpoint_evidence(
            hosts,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        )
        _write_immutable(root / "summary.json", summary)
    multihost_utils.sync_global_devices(f"sunfish-checkpoint-summary-{run_id}")
    return summary if summary is not None else payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, help="local path or gs://bucket/prefix")
    parser.add_argument("--run-id", required=True, help="unique identifier; never overwritten")
    parser.add_argument(
        "--evidence-dir",
        help="local or gs:// prefix for immutable all-host evidence",
    )
    parser.add_argument(
        "--allow-non-tpu",
        action="store_true",
        help="run degraded on one local CPU/GPU process",
    )
    parser.add_argument(
        "--expected-devices", type=int, default=_env_int("EXPECTED_TPU_DEVICES", 0)
    )
    parser.add_argument(
        "--expected-processes", type=int, default=_env_int("EXPECTED_TPU_PROCESSES", 0)
    )
    parser.add_argument(
        "--expected-local-devices",
        type=int,
        default=_env_int("EXPECTED_LOCAL_TPU_DEVICES", 0),
    )
    parser.add_argument(
        "--data-axis-size",
        type=int,
        default=0,
        help="Phase-B data-axis size; 0 uses every global device",
    )
    args = parser.parse_args()
    payload = run_smoke(
        args.workdir,
        args.run_id,
        require_tpu=not args.allow_non_tpu,
        expected_devices=args.expected_devices,
        expected_processes=args.expected_processes,
        expected_local_devices=args.expected_local_devices,
        data_axis_size=args.data_axis_size,
    )
    if not args.allow_non_tpu and not args.evidence_dir:
        parser.error("--evidence-dir is required for the TPU readiness gate")
    if args.evidence_dir:
        payload = write_checkpoint_evidence(
            payload,
            evidence_dir=args.evidence_dir,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
