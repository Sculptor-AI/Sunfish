"""Real 8B target-sharded seed restore for Stage-0.5 readiness gate 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import resource
import sys
from collections.abc import Mapping, Sequence
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


def verify_seed_load_evidence(
    hosts: Sequence[Mapping[str, Any]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(hosts) != expected_processes:
        errors.append(f"found {len(hosts)} host reports; expected {expected_processes}")
    indices = [host.get("process_index") for host in hosts]
    if sorted(indices) != list(range(expected_processes)):
        errors.append(f"process indices are {sorted(indices)}")
    run_ids = {host.get("run_id") for host in hosts}
    tree_hashes = {host.get("tree_sha256") for host in hosts}
    sharding_hashes = {host.get("sharding_sha256") for host in hosts}
    seed_hashes = {host.get("seed_manifest_sha256") for host in hosts}
    inventory_hashes = {host.get("seed_gcs_inventory_sha256") for host in hosts}
    seed_paths = {host.get("seed_path") for host in hosts}
    manifest_paths = {host.get("seed_manifest_path") for host in hosts}
    sources = [normalize_source_identity(host.get("sunfish_source")) for host in hosts]
    for label, values in (
        ("run IDs", run_ids),
        ("tree hashes", tree_hashes),
        ("sharding hashes", sharding_hashes),
        ("seed hashes", seed_hashes),
        ("seed GCS inventory hashes", inventory_hashes),
        ("seed paths", seed_paths),
        ("seed manifest paths", manifest_paths),
    ):
        if len(values) != 1 or None in values:
            errors.append(f"host {label} differ")
    if any(source is None for source in sources) or len(set(sources)) != 1:
        errors.append("host source identities are missing or differ")
    for host in hosts:
        process = host.get("process_index")
        if host.get("schema_version") != 1:
            errors.append(f"process {process} has an unsupported schema")
        if host.get("gate") != 3 or host.get("scope") != (
            "real-8b-orbax-seed-target-sharded-restore"
        ):
            errors.append(f"process {process} reports the wrong gate scope")
        if host.get("process_count") != expected_processes:
            errors.append(f"process {process} reports the wrong process count")
        if host.get("global_device_count") != expected_devices:
            errors.append(f"process {process} reports the wrong global device count")
        if host.get("local_device_count") != expected_local_devices:
            errors.append(f"process {process} reports the wrong local device count")
        topology = host.get("topology")
        if not isinstance(topology, Mapping) or topology.get("ready") is not True:
            errors.append(f"process {process} topology did not pass")
        for key in (
            "restored_exact_target_tree",
            "restored_exact_target_shardings",
            "host_does_not_hold_full_model",
        ):
            if host.get(key) is not True:
                errors.append(f"process {process} {key}={host.get(key)!r}")
        global_bytes = int(host.get("global_parameter_bytes", 0))
        host_bytes = int(host.get("host_parameter_bytes", 0))
        if not 0 < host_bytes < global_bytes:
            errors.append(
                f"process {process} resident bytes {host_bytes} not below {global_bytes}"
            )
        device_bytes = host.get("device_parameter_bytes")
        if not isinstance(device_bytes, Mapping) or not device_bytes:
            errors.append(f"process {process} has no per-device byte evidence")
        elif max(int(value) for value in device_bytes.values()) >= global_bytes:
            errors.append(f"process {process} has a device holding the full model")
    return {
        "schema_version": 1,
        "gate": 3,
        "scope": "real-8b-orbax-seed-target-sharded-restore",
        "run_id": next(iter(run_ids), None),
        "seed_manifest_sha256": next(iter(seed_hashes), None),
        "seed_gcs_inventory_sha256": next(iter(inventory_hashes), None),
        "passed": not errors,
        "errors": errors,
        "hosts": list(hosts),
        "sunfish_source": hosts[0].get("sunfish_source") if hosts else None,
    }


def _write_immutable(path: Any, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable seed-load evidence changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def _sharding_signature(tree: Any, *, jax: Any) -> str:
    rows = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        rows.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sharding": str(value.sharding),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resident_parameter_bytes(tree: Any, *, jax: Any) -> tuple[int, dict[str, int]]:
    per_device: dict[str, int] = {}
    for value in jax.tree.leaves(tree):
        if not isinstance(value, jax.Array):
            continue
        for shard in value.addressable_shards:
            device = str(shard.device)
            per_device[device] = per_device.get(device, 0) + math.prod(
                shard.data.shape
            ) * shard.data.dtype.itemsize
    return sum(per_device.values()), per_device


def _memory_stats(jax: Any) -> dict[str, Mapping[str, int | float]]:
    output = {}
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        output[str(device)] = {
            str(key): value
            for key, value in stats.items()
            if isinstance(value, (int, float))
        }
    return output


def run_seed_load_smoke(
    *,
    seed_path: str,
    seed_manifest_path: str,
    seed_manifest_sha256: str,
    evidence_dir: str,
    run_id: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid seed-load run ID")
    require_launcher_run_id(run_id)
    # No backend-adjacent import may move above this call.
    jax, initialization = initialize_distributed_jax(require_distributed=True)
    import flax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh
    import numpy as np
    import orbax.checkpoint as ocp
    from etils import epath

    from sunfish_tpu.orbax_seed import (
        AUDITED_TARGET_TEXT_PARAMETERS_32E,
        OFFICIAL_JAX_TREE_TEXT_DELTA,
        _target_abstract_params,
        _tree_signature,
        require_parameter_count,
    )
    from sunfish_tpu.gcs_inventory import verify_live_gcs_inventory
    from sunfish_tpu.seed_manifest import validate_seed_manifest_bytes
    from sunfish_tpu.training.checkpoint import _validate_exact_tree
    from sunfish_tpu.training.runtime import verify_runtime_contract
    from sunfish_tpu.training.sharding import PhaseBTreeSharding
    from sunfish_tpu.training.sharding_policy import resolve_phase_b_mesh

    topology = report(
        [
            initialization,
            *_topology_checks(
                jax,
                jnp,
                require_tpu=True,
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            ),
        ]
    )
    if not topology["ready"]:
        raise RuntimeError(f"seed-load topology failed: {json.dumps(topology)}")
    versions = verify_runtime_contract(require_tpu=True)
    seed_manifest = epath.Path(seed_manifest_path)
    manifest_payload = validate_seed_manifest_bytes(
        seed_manifest.read_bytes(),
        expected_sha256=seed_manifest_sha256,
        init_path=seed_path,
        phase="smoke",
        expected_num_experts=32,
        expected_top_k_experts=4,
    )
    live_seed_inventory = verify_live_gcs_inventory(
        seed_path, manifest_payload["output_gcs_inventory"]
    )

    devices = sorted(
        jax.devices(), key=lambda device: (int(device.process_index), int(device.id))
    )
    mesh_shape, axis_names, data_axis_size = resolve_phase_b_mesh(
        global_device_count=len(devices), num_experts=32
    )
    mesh = Mesh(np.asarray(devices, dtype=object).reshape(mesh_shape), axis_names)
    abstract = _target_abstract_params(
        num_experts=32,
        top_k_experts=4,
        jax=jax,
        jnp=jnp,
    )
    target_shardings = PhaseBTreeSharding(
        mesh=mesh, data_axis_size=data_axis_size
    )(abstract)
    target = jax.tree.map(
        lambda value, sharding: jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=sharding
        ),
        abstract,
        target_shardings,
    )
    before_memory = _memory_stats(jax)
    before_max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(epath.Path(seed_path), target)
    finally:
        checkpointer.close()
    jax.block_until_ready(restored)
    after_max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    after_memory = _memory_stats(jax)
    _validate_exact_tree(target, restored)
    signature = _tree_signature(restored, flax)
    require_parameter_count(
        signature,
        # The exact-tree seed derives from the official JAX checkpoint, which
        # carries OFFICIAL_JAX_TREE_TEXT_DELTA (-30) versus the safetensors
        # contract — same reconciliation the materializer's own gates apply.
        expected=AUDITED_TARGET_TEXT_PARAMETERS_32E + OFFICIAL_JAX_TREE_TEXT_DELTA,
        label="Stage-0.5 real sharded seed",
    )
    global_bytes = sum(
        math.prod(value.shape) * np.dtype(value.dtype).itemsize
        for value in jax.tree.leaves(restored)
    )
    host_bytes, per_device = _resident_parameter_bytes(restored, jax=jax)
    payload = {
        "schema_version": 1,
        "gate": 3,
        "scope": "real-8b-orbax-seed-target-sharded-restore",
        "run_id": run_id,
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "global_device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "seed_path": seed_path,
        "seed_manifest_path": seed_manifest_path,
        "seed_manifest_sha256": seed_manifest_sha256,
        "selection_metadata": manifest_payload["selection_metadata"],
        "seed_gcs_inventory_sha256": live_seed_inventory["sha256"],
        "restored_exact_target_tree": True,
        "restored_exact_target_shardings": True,
        "host_does_not_hold_full_model": host_bytes < global_bytes,
        "global_parameter_bytes": global_bytes,
        "host_parameter_bytes": host_bytes,
        "device_parameter_bytes": per_device,
        "tree_sha256": signature["sha256"],
        "sharding_sha256": _sharding_signature(restored, jax=jax),
        "mesh": {"shape": list(mesh_shape), "axes": list(axis_names)},
        "host_max_rss_before": before_max_rss,
        "host_max_rss_after": after_max_rss,
        "device_memory_before": before_memory,
        "device_memory_after": after_memory,
        "runtime_versions": versions,
        "topology": topology,
        "sunfish_source": source_identity_from_environment(required=True),
    }
    root = epath.Path(evidence_dir) / run_id
    process_index = int(jax.process_index())
    _write_immutable(root / f"host-{process_index:05d}.json", payload)
    multihost_utils.sync_global_devices(f"sunfish-seed-load-hosts-{run_id}")
    summary = None
    if process_index == 0:
        hosts = [
            json.loads((root / f"host-{process:05d}.json").read_text())
            for process in range(expected_processes)
        ]
        summary = verify_seed_load_evidence(
            hosts,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        )
        _write_immutable(root / "summary.json", summary)
    multihost_utils.sync_global_devices(f"sunfish-seed-load-summary-{run_id}")
    return summary if summary is not None else payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", required=True)
    parser.add_argument("--seed-manifest-path", required=True)
    parser.add_argument("--seed-manifest-sha256", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        payload = run_seed_load_smoke(
            seed_path=args.seed_path,
            seed_manifest_path=args.seed_manifest_path,
            seed_manifest_sha256=args.seed_manifest_sha256,
            evidence_dir=args.evidence_dir,
            run_id=args.run_id,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"sunfish-seed-load-smoke: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("passed", payload.get("host_does_not_hold_full_model")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
