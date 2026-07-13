"""Real-model exact-resume proof for Stage-0.5 gate 6.

This runs the production Sunfish model, optimizer, process-sharded Grain input,
and Kauldron/Orbax checkpoint objects. It checkpoints after one real update,
runs the next update as an uninterrupted control, fully reinitializes the
trainer state as a restarted process would, restores the checkpoint and input
cursor, and reruns that update. Every process compares its addressable shards.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sunfish_tpu.tpu_preflight import (
    _topology_checks,
    initialize_distributed_jax,
    report,
)
from sunfish_tpu.training.spec import HarnessConfig, Phase
from sunfish_tpu.source_identity import (
    normalize_source_identity,
    require_launcher_run_id,
    source_identity_from_environment,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXACT_KEYS = (
    "next_batch_exact",
    "next_loss_exact",
    "next_trainable_gradients_exact",
    "next_trainable_updates_exact",
    "next_trainable_params_exact",
    "next_optimizer_state_exact",
    "next_collections_exact",
    "next_step_exact",
    "control_frozen_params_unchanged",
    "resumed_frozen_params_unchanged",
)


def verify_real_resume_evidence(
    hosts: Sequence[Mapping[str, Any]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    """Merge exact local-shard comparisons into one gate-6 decision."""
    errors: list[str] = []
    if len(hosts) != expected_processes:
        errors.append(f"found {len(hosts)} host reports; expected {expected_processes}")
    indices = [host.get("process_index") for host in hosts]
    if sorted(indices) != list(range(expected_processes)):
        errors.append(f"process indices are {sorted(indices)}")
    run_ids = {host.get("run_id") for host in hosts}
    config_digests = {host.get("config_sha256") for host in hosts}
    config_file_digests = {host.get("config_file_sha256") for host in hosts}
    attempt_ids = {host.get("attempt_id") for host in hosts}
    dataset_hashes = {host.get("dataset_manifest_sha256") for host in hosts}
    seed_hashes = {host.get("seed_manifest_sha256") for host in hosts}
    checkpoint_steps = {host.get("checkpoint_step") for host in hosts}
    sources = [normalize_source_identity(host.get("sunfish_source")) for host in hosts]
    if len(run_ids) != 1 or None in run_ids:
        errors.append("host run IDs differ")
    if len(config_digests) != 1 or None in config_digests:
        errors.append("host config digests differ")
    if (
        len(config_file_digests) != 1
        or not _SHA256.fullmatch(str(next(iter(config_file_digests), "")))
    ):
        errors.append("host raw config-file digests differ or are invalid")
    if len(attempt_ids) != 1 or None in attempt_ids:
        errors.append("host attempt IDs differ")
    if len(dataset_hashes) != 1 or None in dataset_hashes:
        errors.append("host dataset hashes differ")
    if len(seed_hashes) != 1 or None in seed_hashes:
        errors.append("host seed hashes differ")
    if checkpoint_steps != {1}:
        errors.append(f"checkpoint steps differ: {sorted(checkpoint_steps)}")
    if any(source is None for source in sources) or len(set(sources)) != 1:
        errors.append("host source identities are missing or differ")
    for host in hosts:
        process = host.get("process_index")
        if host.get("schema_version") != 1:
            errors.append(f"process {process} has an unsupported schema")
        if host.get("gate") != 6 or host.get("scope") != (
            "production-model-optimizer-grain-orbax"
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
        for key in _EXACT_KEYS:
            if host.get(key) is not True:
                errors.append(f"process {process} {key}={host.get(key)!r}")
        digests = host.get("digests")
        if not isinstance(digests, Mapping):
            errors.append(f"process {process} has no comparison digests")
            continue
        for name in (
            "batch",
            "loss",
            "gradients",
            "updates",
            "params",
            "opt_state",
            "collections",
            "step",
        ):
            pair = digests.get(name)
            if not isinstance(pair, Mapping) or pair.get("control") != pair.get("resumed"):
                errors.append(f"process {process} digest mismatch for {name}")
    return {
        "schema_version": 1,
        "gate": 6,
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": next(iter(run_ids), None),
        "attempt_id": next(iter(attempt_ids), None),
        "config_sha256": next(iter(config_digests), None),
        "config_file_sha256": next(iter(config_file_digests), None),
        "dataset_manifest_sha256": next(iter(dataset_hashes), None),
        "seed_manifest_sha256": next(iter(seed_hashes), None),
        "checkpoint_step": next(iter(checkpoint_steps), None),
        "passed": not errors,
        "errors": errors,
        "hosts": list(hosts),
        "sunfish_source": hosts[0].get("sunfish_source") if hosts else None,
    }


def _path_is_trainable(path: tuple[Any, ...]) -> bool:
    return any("lora" in str(component).lower() for component in path)


def _select_trainable(tree: Any, flax: Any) -> dict[str, Any]:
    flat = flax.traverse_util.flatten_dict(tree)
    selected = {
        "/".join(str(component) for component in path): value
        for path, value in flat.items()
        if _path_is_trainable(path)
    }
    if not selected:
        raise RuntimeError("diagnostic found no trainable LoRA leaves")
    return selected


def _frozen_params_unchanged(before: Any, after: Any, *, flax: Any, jnp: Any):
    before_flat = flax.traverse_util.flatten_dict(before)
    after_flat = flax.traverse_util.flatten_dict(after)
    if set(before_flat) != set(after_flat):
        raise RuntimeError("parameter paths changed during diagnostic step")
    checks = [
        jnp.array_equal(before_flat[path], after_flat[path])
        for path in before_flat
        if not _path_is_trainable(path)
    ]
    if not checks:
        raise RuntimeError("diagnostic found no frozen base leaves")
    return functools.reduce(jnp.logical_and, checks)


def _snapshot(tree: Any, *, jax: Any, np: Any) -> dict[str, Any]:
    leaves, treedef = jax.tree.flatten_with_path(tree)
    entries = []
    for path, leaf in leaves:
        key = jax.tree_util.keystr(path)
        if isinstance(leaf, jax.Array):
            shards = []
            for shard in leaf.addressable_shards:
                shards.append(
                    (
                        str(shard.index),
                        np.asarray(shard.data).copy(),
                    )
                )
            entries.append((key, tuple(int(x) for x in leaf.shape), str(leaf.dtype), shards))
        else:
            entries.append((key, (), type(leaf).__name__, [("host", copy.deepcopy(leaf))]))
    return {"treedef": treedef, "entries": entries}


def _snapshot_equal(left: Mapping[str, Any], right: Mapping[str, Any], *, np: Any) -> bool:
    if left["treedef"] != right["treedef"]:
        return False
    if len(left["entries"]) != len(right["entries"]):
        return False
    for left_entry, right_entry in zip(left["entries"], right["entries"], strict=True):
        if left_entry[:3] != right_entry[:3]:
            return False
        left_shards, right_shards = left_entry[3], right_entry[3]
        if len(left_shards) != len(right_shards):
            return False
        for (left_index, left_data), (right_index, right_data) in zip(
            left_shards, right_shards, strict=True
        ):
            if left_index != right_index:
                return False
            if hasattr(left_data, "shape"):
                if not np.array_equal(left_data, right_data):
                    return False
            elif left_data != right_data:
                return False
    return True


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(snapshot["treedef"]).encode())
    for path, shape, dtype, shards in snapshot["entries"]:
        digest.update(json.dumps([path, shape, dtype], separators=(",", ":")).encode())
        for index, data in shards:
            digest.update(index.encode())
            if hasattr(data, "tobytes"):
                digest.update(data.tobytes(order="C"))
            else:
                digest.update(repr(data).encode())
    return digest.hexdigest()


def _delete_arrays(tree: Any, *, jax: Any) -> None:
    for value in jax.tree.leaves(tree):
        if isinstance(value, jax.Array):
            value.delete()


def _write_immutable(path: Any, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable real-resume evidence changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def run_real_resume_smoke(
    *,
    config_path: Path,
    attempt_id: str,
    evidence_dir: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    config = HarnessConfig.load(config_path)
    if config.run.phase is not Phase.SMOKE:
        raise ValueError("real resume smoke requires phase=smoke")
    if not _RUN_ID.fullmatch(attempt_id):
        raise ValueError("invalid attempt ID")
    require_launcher_run_id(config.run.run_id)
    if config.topology.expected_devices != expected_devices:
        raise ValueError("CLI expected devices differ from the strict config")
    if config.topology.expected_processes != expected_processes:
        raise ValueError("CLI expected processes differ from the strict config")
    if config.topology.expected_local_devices != expected_local_devices:
        raise ValueError("CLI expected local devices differ from the strict config")

    # No backend-adjacent imports may move above distributed initialization.
    jax, initialization = initialize_distributed_jax(require_distributed=True)
    import flax
    import jax.numpy as jnp
    import numpy as np
    from etils import epath
    from jax.experimental import multihost_utils
    from kauldron import konfig
    from kauldron.train import checkpoint_state
    from kauldron.utils.sharding_utils import sharding as sharding_lib

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
        raise RuntimeError(f"distributed topology failed: {json.dumps(topology)}")

    os.environ["SUNFISH_TRAIN_CONFIG"] = str(config_path.resolve())
    os.environ["SUNFISH_ALLOW_NON_TPU"] = "0"
    os.environ["SUNFISH_ATTEMPT_ID"] = attempt_id
    from sunfish_tpu.training.kauldron_config import get_config

    trainer = konfig.resolve(get_config())
    trainer.setup.run(trainer)
    if trainer.checkpointer.latest_step is not None:
        raise FileExistsError(
            "real-resume workdir already contains checkpoints; use a new run ID/workdir"
        )

    elem_spec = trainer.train_ds.element_spec
    state = trainer.trainstep.init(elem_spec=elem_spec, skip_transforms=False)
    ds_iter = iter(trainer.train_ds)
    chrono_template = copy.deepcopy(trainer._chrono)  # pylint: disable=protected-access
    chrono = copy.deepcopy(chrono_template)
    chrono.start_loop()

    first_batch = next(ds_iter)
    first_batch = sharding_lib.device_put(first_batch, trainer.sharding.batch)
    state, _ = trainer.trainstep.step(state, first_batch)
    jax.block_until_ready(state)
    chrono.finish_step()
    checkpoint_step = int(jax.device_get(state.step))
    if checkpoint_step != 1:
        raise RuntimeError(f"warmup produced step {checkpoint_step}, expected 1")
    trainer.checkpointer.save(
        checkpoint_state.CheckpointState(state, chrono, ds_iter),
        step=checkpoint_step,
        force=True,
    )
    trainer.checkpointer.wait_until_finished()

    @functools.partial(jax.jit, donate_argnames=("current_state",))
    def diagnostic_step(current_state, batch):
        with trainer.sharding.set_global_mesh():
            next_state, context = trainer.trainstep._step(  # pylint: disable=protected-access
                current_state, batch
            )
        next_state = sharding_lib.with_sharding_constraint(
            next_state, trainer.sharding.state
        )
        return next_state, {
            "batch": batch,
            "loss": context.loss_total,
            "gradients": _select_trainable(context.grads, flax),
            "updates": _select_trainable(context.updates, flax),
            "params": _select_trainable(next_state.params, flax),
            "opt_state": next_state.opt_state,
            "collections": next_state.collections,
            "step": next_state.step,
            "frozen_unchanged": _frozen_params_unchanged(
                current_state.params, next_state.params, flax=flax, jnp=jnp
            ),
        }

    control_batch = next(ds_iter)
    control_batch = sharding_lib.device_put(control_batch, trainer.sharding.batch)
    control_state, control = diagnostic_step(state, control_batch)
    jax.block_until_ready((control_state, control))
    control_snapshot = {
        key: _snapshot(value, jax=jax, np=np)
        for key, value in control.items()
    }
    _delete_arrays(control_state, jax=jax)
    del control_state, control, state, control_batch, first_batch

    # Mirror Kauldron's real restart path: initialize only target metadata and
    # optimizer structure, create a fresh iterator, then donate both to Orbax.
    restart_state = trainer.trainstep.init(
        elem_spec=elem_spec,
        skip_transforms=True,
    )
    restart_iter = iter(trainer.train_ds)
    restart_chrono = copy.deepcopy(chrono_template)
    restored_state, _, restored_iter = trainer.checkpointer.restore(
        checkpoint_state.CheckpointState(
            restart_state, restart_chrono, restart_iter
        ),
        step=checkpoint_step,
        donate=True,
    )
    resumed_batch = next(restored_iter)
    resumed_batch = sharding_lib.device_put(resumed_batch, trainer.sharding.batch)
    resumed_state, resumed = diagnostic_step(restored_state, resumed_batch)
    jax.block_until_ready((resumed_state, resumed))
    resumed_snapshot = {
        key: _snapshot(value, jax=jax, np=np)
        for key, value in resumed.items()
    }

    exact = {
        key: _snapshot_equal(control_snapshot[key], resumed_snapshot[key], np=np)
        for key in control_snapshot
    }
    digests = {
        key: {
            "control": _snapshot_digest(control_snapshot[key]),
            "resumed": _snapshot_digest(resumed_snapshot[key]),
        }
        for key in control_snapshot
    }
    config_file_sha256 = os.environ.get("SUNFISH_CONFIG_FILE_SHA256", "")
    if not _SHA256.fullmatch(config_file_sha256):
        raise RuntimeError(
            "SUNFISH_CONFIG_FILE_SHA256 must be set by the all-host launcher"
        )
    payload = {
        "schema_version": 1,
        "gate": 6,
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": config.run.run_id,
        "attempt_id": attempt_id,
        "config_sha256": config.digest,
        "config_file_sha256": config_file_sha256,
        "dataset_manifest_sha256": config.data.manifest_sha256,
        "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
        "checkpoint_step": checkpoint_step,
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "global_device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "next_batch_exact": exact["batch"],
        "next_loss_exact": exact["loss"],
        "next_trainable_gradients_exact": exact["gradients"],
        "next_trainable_updates_exact": exact["updates"],
        "next_trainable_params_exact": exact["params"],
        "next_optimizer_state_exact": exact["opt_state"],
        "next_collections_exact": exact["collections"],
        "next_step_exact": exact["step"],
        "control_frozen_params_unchanged": bool(
            np.asarray(control_snapshot["frozen_unchanged"]["entries"][0][3][0][1])
        ),
        "resumed_frozen_params_unchanged": bool(
            np.asarray(resumed_snapshot["frozen_unchanged"]["entries"][0][3][0][1])
        ),
        "digests": {
            key: value for key, value in digests.items() if key != "frozen_unchanged"
        },
        "topology": topology,
        "sunfish_source": source_identity_from_environment(required=True),
    }
    payload["passed"] = all(payload[key] is True for key in _EXACT_KEYS)

    root = epath.Path(evidence_dir) / attempt_id
    host_path = root / f"host-{int(jax.process_index()):05d}.json"
    _write_immutable(host_path, payload)
    multihost_utils.sync_global_devices(f"sunfish-real-resume-hosts-{attempt_id}")
    summary = None
    if int(jax.process_index()) == 0:
        hosts = [
            json.loads((root / f"host-{process:05d}.json").read_text())
            for process in range(expected_processes)
        ]
        summary = verify_real_resume_evidence(
            hosts,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        )
        _write_immutable(root / "summary.json", summary)
    multihost_utils.sync_global_devices(f"sunfish-real-resume-summary-{attempt_id}")

    _delete_arrays(resumed_state, jax=jax)
    trainer.checkpointer.close()
    return summary if summary is not None else payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        payload = run_real_resume_smoke(
            config_path=args.config,
            attempt_id=args.attempt_id,
            evidence_dir=args.evidence_dir,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"sunfish-real-resume-smoke: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
