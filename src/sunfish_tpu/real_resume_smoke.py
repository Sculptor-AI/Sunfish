"""Real-model exact-resume proof for Stage-0.5 gate 6.

This runs the production Sunfish model, optimizer, process-sharded Grain input,
and Kauldron/Orbax checkpoint objects in two sequential Python processes on
every host. The first process checkpoints after one update and records the next
uninterrupted update. It closes its checkpoint manager and exits. A second
process rebuilds the trainer/manager, discovers and restores the checkpoint,
then compares every addressable shard of the next update.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sunfish_tpu.tpu_preflight import (
    _topology_checks,
    initialize_distributed_jax,
    report,
    validate_gcs_uri,
)
from sunfish_tpu.training.spec import HarnessConfig, Phase
from sunfish_tpu.source_identity import (
    normalize_source_identity,
    require_launcher_run_id,
    source_identity_from_environment,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_PHASES = ("prepare", "resume")
_PROCESS_TOKEN_ENV = "SUNFISH_REAL_RESUME_PROCESS_TOKEN"
_ORCHESTRATED_ENV = "SUNFISH_REAL_RESUME_ORCHESTRATED"
_SIGNAL_GRACE_SECONDS = 30.0
_TERMINATE_GRACE_SECONDS = 5.0
_WAIT_POLL_SECONDS = 0.5
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
_DIGEST_KEYS = (
    "batch",
    "loss",
    "gradients",
    "updates",
    "params",
    "opt_state",
    "collections",
    "step",
)


def _encoded_payload(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_encoded_payload(payload)).hexdigest()


def _common_host_evidence_errors(
    hosts: Sequence[Mapping[str, Any]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> list[str]:
    errors: list[str] = []
    if len(hosts) != expected_processes:
        errors.append(f"found {len(hosts)} host reports; expected {expected_processes}")
    indices = [host.get("process_index") for host in hosts]
    if sorted(indices) != list(range(expected_processes)):
        errors.append(f"process indices are {sorted(indices)}")
    sources = [normalize_source_identity(host.get("sunfish_source")) for host in hosts]
    if any(source is None for source in sources) or len(set(sources)) != 1:
        errors.append("host source identities are missing or differ")
    for host in hosts:
        process = host.get("process_index")
        if host.get("process_count") != expected_processes:
            errors.append(f"process {process} reports the wrong process count")
        if host.get("global_device_count") != expected_devices:
            errors.append(f"process {process} reports the wrong global device count")
        if host.get("local_device_count") != expected_local_devices:
            errors.append(f"process {process} reports the wrong local device count")
        topology = host.get("topology")
        if not isinstance(topology, Mapping) or topology.get("ready") is not True:
            errors.append(f"process {process} topology did not pass")
    return errors


def verify_real_resume_prepare_evidence(
    hosts: Sequence[Mapping[str, Any]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    """Merge the first-process checkpoint/control records for gate 6."""
    errors = _common_host_evidence_errors(
        hosts,
        expected_devices=expected_devices,
        expected_processes=expected_processes,
        expected_local_devices=expected_local_devices,
    )
    run_ids = {host.get("run_id") for host in hosts}
    proof_ids = {host.get("proof_id") for host in hosts}
    launcher_attempt_ids = {host.get("launcher_attempt_id") for host in hosts}
    config_digests = {host.get("config_sha256") for host in hosts}
    config_file_digests = {host.get("config_file_sha256") for host in hosts}
    dataset_hashes = {host.get("dataset_manifest_sha256") for host in hosts}
    seed_hashes = {host.get("seed_manifest_sha256") for host in hosts}
    checkpoint_steps = {host.get("checkpoint_step") for host in hosts}
    for label, values, pattern in (
        ("run IDs", run_ids, _RUN_ID),
        ("proof IDs", proof_ids, _RUN_ID),
        ("launcher attempt IDs", launcher_attempt_ids, _RUN_ID),
        ("config digests", config_digests, _SHA256),
        ("raw config-file digests", config_file_digests, _SHA256),
        ("dataset hashes", dataset_hashes, _SHA256),
        ("seed hashes", seed_hashes, _SHA256),
    ):
        value = next(iter(values), None)
        if len(values) != 1 or not isinstance(value, str) or not pattern.fullmatch(value):
            errors.append(f"host {label} differ or are invalid")
    if checkpoint_steps != {1}:
        errors.append(f"checkpoint steps differ: {sorted(checkpoint_steps)}")
    tokens = {host.get("process_token") for host in hosts}
    if len(tokens) != len(hosts):
        errors.append("prepare process tokens are not unique per host")
    for host in hosts:
        process = host.get("process_index")
        if (
            host.get("schema_version") != 2
            or host.get("gate") != 6
            or host.get("phase") != "prepare"
        ):
            errors.append(f"process {process} has the wrong prepare schema/phase")
        if host.get("scope") != "production-model-optimizer-grain-orbax":
            errors.append(f"process {process} reports the wrong gate scope")
        if host.get("passed") is not True:
            errors.append(f"process {process} prepare payload did not pass")
        if host.get("control_frozen_params_unchanged") is not True:
            errors.append(f"process {process} changed frozen control parameters")
        token = host.get("process_token")
        if not isinstance(token, str) or not _SHA256.fullmatch(token):
            errors.append(f"process {process} has no valid prepare process token")
        if not isinstance(host.get("process_pid"), int) or host["process_pid"] <= 1:
            errors.append(f"process {process} has no valid prepare process PID")
        digests = host.get("control_digests")
        if not isinstance(digests, Mapping):
            errors.append(f"process {process} has no control digests")
            continue
        if set(digests) != set(_DIGEST_KEYS):
            errors.append(f"process {process} has the wrong control digest set")
        for name in _DIGEST_KEYS:
            value = digests.get(name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                errors.append(f"process {process} has an invalid control digest for {name}")
    return {
        "schema_version": 2,
        "gate": 6,
        "phase": "prepare",
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": next(iter(run_ids), None),
        "proof_id": next(iter(proof_ids), None),
        "launcher_attempt_id": next(iter(launcher_attempt_ids), None),
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


def verify_real_resume_evidence(
    hosts: Sequence[Mapping[str, Any]],
    *,
    prepare_summary: Mapping[str, Any] | None,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    """Merge and independently revalidate both processes of gate 6."""
    errors = _common_host_evidence_errors(
        hosts,
        expected_devices=expected_devices,
        expected_processes=expected_processes,
        expected_local_devices=expected_local_devices,
    )
    run_ids = {host.get("run_id") for host in hosts}
    config_digests = {host.get("config_sha256") for host in hosts}
    config_file_digests = {host.get("config_file_sha256") for host in hosts}
    attempt_ids = {host.get("attempt_id") for host in hosts}
    launcher_attempt_ids = {host.get("launcher_attempt_id") for host in hosts}
    dataset_hashes = {host.get("dataset_manifest_sha256") for host in hosts}
    seed_hashes = {host.get("seed_manifest_sha256") for host in hosts}
    checkpoint_steps = {host.get("checkpoint_step") for host in hosts}
    prepare_summary_hashes = {host.get("prepare_summary_sha256") for host in hosts}
    prepare_launcher_attempt_ids = {
        host.get("prepare_launcher_attempt_id") for host in hosts
    }
    prepare_tokens = {host.get("prepare_process_token") for host in hosts}
    resume_tokens = {host.get("resume_process_token") for host in hosts}
    for label, values, pattern in (
        ("run IDs", run_ids, _RUN_ID),
        ("config digests", config_digests, _SHA256),
        ("raw config-file digests", config_file_digests, _SHA256),
        ("attempt IDs", attempt_ids, _RUN_ID),
        ("launcher attempt IDs", launcher_attempt_ids, _RUN_ID),
        (
            "prepare launcher attempt IDs",
            prepare_launcher_attempt_ids,
            _RUN_ID,
        ),
        ("dataset hashes", dataset_hashes, _SHA256),
        ("seed hashes", seed_hashes, _SHA256),
    ):
        value = next(iter(values), None)
        if len(values) != 1 or not isinstance(value, str) or not pattern.fullmatch(value):
            errors.append(f"host {label} differ or are invalid")
    if checkpoint_steps != {1}:
        errors.append(f"checkpoint steps differ: {sorted(checkpoint_steps)}")

    embedded_prepare = dict(prepare_summary) if isinstance(prepare_summary, Mapping) else None
    prepare_hosts: list[Mapping[str, Any]] = []
    prepare_by_process: dict[Any, Mapping[str, Any]] = {}
    expected_prepare_sha256 = None
    if embedded_prepare is None:
        errors.append("final evidence has no embedded prepare summary")
    else:
        expected_prepare_sha256 = _payload_sha256(embedded_prepare)
        raw_prepare_hosts = embedded_prepare.get("hosts")
        if not isinstance(raw_prepare_hosts, list) or not all(
            isinstance(host, Mapping) for host in raw_prepare_hosts
        ):
            errors.append("embedded prepare summary has no valid host evidence")
        else:
            prepare_hosts = list(raw_prepare_hosts)
            recomputed_prepare = verify_real_resume_prepare_evidence(
                prepare_hosts,
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            )
            if embedded_prepare != recomputed_prepare:
                errors.append(
                    "embedded prepare summary does not match its host evidence"
                )
            if recomputed_prepare["passed"] is not True:
                errors.append("embedded prepare summary did not pass")
            prepare_by_process = {
                host.get("process_index"): host for host in prepare_hosts
            }

    if (
        len(prepare_summary_hashes) != 1
        or expected_prepare_sha256 is None
        or next(iter(prepare_summary_hashes), None) != expected_prepare_sha256
    ):
        errors.append("host prepare-summary digests differ from embedded evidence")
    if len(prepare_tokens) != len(hosts) or len(resume_tokens) != len(hosts):
        errors.append("prepare/resume process tokens are not unique per host")
    if prepare_tokens.intersection(resume_tokens):
        errors.append("prepare and resume process-token sets overlap")

    if embedded_prepare is not None:
        cross_phase_fields = {
            "run_id": next(iter(run_ids), None),
            "proof_id": next(iter(attempt_ids), None),
            "launcher_attempt_id": next(iter(launcher_attempt_ids), None),
            "config_sha256": next(iter(config_digests), None),
            "config_file_sha256": next(iter(config_file_digests), None),
            "dataset_manifest_sha256": next(iter(dataset_hashes), None),
            "seed_manifest_sha256": next(iter(seed_hashes), None),
            "checkpoint_step": next(iter(checkpoint_steps), None),
        }
        changed = [
            name
            for name, expected in cross_phase_fields.items()
            if embedded_prepare.get(name) != expected
        ]
        if changed:
            errors.append(
                "prepare/resume summary lineage differs: " + ", ".join(changed)
            )
        final_source = hosts[0].get("sunfish_source") if hosts else None
        if normalize_source_identity(embedded_prepare.get("sunfish_source")) != (
            normalize_source_identity(final_source)
        ):
            errors.append("prepare/resume source identities differ")

    for host in hosts:
        process = host.get("process_index")
        prepare_host = prepare_by_process.get(process)
        if host.get("schema_version") != 2:
            errors.append(f"process {process} has an unsupported schema")
        if host.get("gate") != 6 or host.get("scope") != (
            "production-model-optimizer-grain-orbax"
        ):
            errors.append(f"process {process} reports the wrong gate scope")
        if host.get("restart_mode") != "separate-python-processes":
            errors.append(f"process {process} did not use a separate-process restart")
        if host.get("passed") is not True:
            errors.append(f"process {process} resume payload did not pass")
        prepare_token = host.get("prepare_process_token")
        resume_token = host.get("resume_process_token")
        if (
            not isinstance(prepare_token, str)
            or not _SHA256.fullmatch(prepare_token)
            or not isinstance(resume_token, str)
            or not _SHA256.fullmatch(resume_token)
            or prepare_token == resume_token
        ):
            errors.append(f"process {process} has invalid or reused process tokens")
        for name in ("prepare_process_pid", "resume_process_pid"):
            if not isinstance(host.get(name), int) or host[name] <= 1:
                errors.append(f"process {process} has an invalid {name}")
        if prepare_host is None:
            errors.append(f"process {process} has no matching prepare host evidence")
        else:
            for final_name, prepare_name in (
                ("prepare_process_token", "process_token"),
                ("prepare_process_pid", "process_pid"),
                ("prepare_launcher_attempt_id", "launcher_attempt_id"),
            ):
                if host.get(final_name) != prepare_host.get(prepare_name):
                    errors.append(
                        f"process {process} {final_name} differs from prepare evidence"
                    )
            if host.get("launcher_attempt_id") != prepare_host.get(
                "launcher_attempt_id"
            ):
                errors.append(
                    f"process {process} launcher attempt differs between processes"
                )
        for key in _EXACT_KEYS:
            if host.get(key) is not True:
                errors.append(f"process {process} {key}={host.get(key)!r}")
        digests = host.get("digests")
        if not isinstance(digests, Mapping):
            errors.append(f"process {process} has no comparison digests")
            continue
        if set(digests) != set(_DIGEST_KEYS):
            errors.append(f"process {process} has the wrong comparison digest set")
        prepare_digests = (
            prepare_host.get("control_digests", {})
            if isinstance(prepare_host, Mapping)
            else {}
        )
        for name in _DIGEST_KEYS:
            pair = digests.get(name)
            if not isinstance(pair, Mapping) or set(pair) != {"control", "resumed"}:
                errors.append(f"process {process} has an invalid digest pair for {name}")
                continue
            control_digest = pair.get("control")
            resumed_digest = pair.get("resumed")
            if (
                not isinstance(control_digest, str)
                or not _SHA256.fullmatch(control_digest)
                or not isinstance(resumed_digest, str)
                or not _SHA256.fullmatch(resumed_digest)
                or control_digest != resumed_digest
            ):
                errors.append(f"process {process} digest mismatch for {name}")
            if control_digest != prepare_digests.get(name):
                errors.append(
                    f"process {process} control digest for {name} differs from prepare"
                )
    return {
        "schema_version": 2,
        "gate": 6,
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": next(iter(run_ids), None),
        "attempt_id": next(iter(attempt_ids), None),
        "launcher_attempt_id": next(iter(launcher_attempt_ids), None),
        "config_sha256": next(iter(config_digests), None),
        "config_file_sha256": next(iter(config_file_digests), None),
        "dataset_manifest_sha256": next(iter(dataset_hashes), None),
        "seed_manifest_sha256": next(iter(seed_hashes), None),
        "checkpoint_step": next(iter(checkpoint_steps), None),
        "prepare_summary_sha256": next(iter(prepare_summary_hashes), None),
        "prepare_launcher_attempt_id": next(
            iter(prepare_launcher_attempt_ids), None
        ),
        "prepare_summary": embedded_prepare,
        "restart_mode": "separate-python-processes",
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
    encoded = _encoded_payload(payload).decode()
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable real-resume evidence changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def _broadcast_process0_error(
    multihost_utils: Any, np: Any, message: str | None, *, limit: int = 16_384
) -> str | None:
    """Broadcast a bounded process-0 merge/write failure to every host."""
    encoded = (message or "").encode("utf-8")
    if len(encoded) >= limit:
        encoded = encoded[: limit - 1]
    payload = np.zeros((limit,), np.uint8)
    if encoded:
        payload[: len(encoded)] = np.frombuffer(encoded, np.uint8)
    received = np.asarray(multihost_utils.broadcast_one_to_all(payload))
    decoded = bytes(received.tolist()).split(b"\0", 1)[0].decode(
        "utf-8", errors="replace"
    )
    return decoded or None


def _path_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def run_real_resume_phase(
    *,
    config_path: Path,
    attempt_id: str,
    evidence_dir: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
    process_phase: str,
) -> dict[str, Any]:
    """Run one orchestrated process of the two-process exact-resume proof."""
    config = HarnessConfig.load(config_path)
    if config.run.phase is not Phase.SMOKE:
        raise ValueError("real resume smoke requires phase=smoke")
    if not _RUN_ID.fullmatch(attempt_id):
        raise ValueError("invalid attempt ID")
    if process_phase not in _PROCESS_PHASES:
        raise ValueError(f"invalid real-resume process phase: {process_phase}")
    if os.environ.get(_ORCHESTRATED_ENV) != "1":
        raise RuntimeError("real-resume process phases must run through the orchestrator")
    process_token = os.environ.get(_PROCESS_TOKEN_ENV, "")
    if not _SHA256.fullmatch(process_token):
        raise RuntimeError("real-resume process token is missing or invalid")
    launcher_attempt_id = os.environ.get("SUNFISH_ATTEMPT_ID", "")
    if not _RUN_ID.fullmatch(launcher_attempt_id):
        raise RuntimeError("all-host launcher attempt ID is missing or invalid")
    require_launcher_run_id(config.run.run_id)
    if config.topology.expected_devices != expected_devices:
        raise ValueError("CLI expected devices differ from the strict config")
    if config.topology.expected_processes != expected_processes:
        raise ValueError("CLI expected processes differ from the strict config")
    if config.topology.expected_local_devices != expected_local_devices:
        raise ValueError("CLI expected local devices differ from the strict config")
    # Gate evidence is merged across worker-local files. A local path would
    # let every host reach the first barrier, then strand peers when process 0
    # cannot read their reports. Fail before JAX initialization instead.
    validate_gcs_uri(evidence_dir)

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
    from sunfish_tpu.training.kauldron_config import get_config

    trainer = konfig.resolve(get_config())
    trainer.setup.run(trainer)
    elem_spec = trainer.train_ds.element_spec
    chrono_template = copy.deepcopy(trainer._chrono)  # pylint: disable=protected-access

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

    config_file_sha256 = os.environ.get("SUNFISH_CONFIG_FILE_SHA256", "")
    if not _SHA256.fullmatch(config_file_sha256):
        raise RuntimeError(
            "SUNFISH_CONFIG_FILE_SHA256 must be set by the all-host launcher"
        )
    root = epath.Path(evidence_dir) / attempt_id
    process_index = int(jax.process_index())
    current_source = source_identity_from_environment(required=True)
    result: dict[str, Any]
    try:
        if process_phase == "prepare":
            if trainer.checkpointer.latest_step is not None:
                raise FileExistsError(
                    "real-resume workdir already contains checkpoints; use a new run ID/workdir"
                )
            state = trainer.trainstep.init(elem_spec=elem_spec, skip_transforms=False)
            ds_iter = iter(trainer.train_ds)
            chrono = copy.deepcopy(chrono_template)
            chrono.start_loop()
            first_batch = next(ds_iter)
            first_batch = sharding_lib.device_put(first_batch, trainer.sharding.batch)
            state, _ = trainer.trainstep.step(state, first_batch)
            jax.block_until_ready(state)
            chrono.finish_step()
            checkpoint_step = int(jax.device_get(state.step))
            if checkpoint_step != 1:
                raise RuntimeError(
                    f"warmup produced step {checkpoint_step}, expected 1"
                )
            trainer.checkpointer.save(
                checkpoint_state.CheckpointState(state, chrono, ds_iter),
                step=checkpoint_step,
                force=True,
            )
            trainer.checkpointer.wait_until_finished()
            control_batch = next(ds_iter)
            control_batch = sharding_lib.device_put(
                control_batch, trainer.sharding.batch
            )
            control_state, control = diagnostic_step(state, control_batch)
            jax.block_until_ready((control_state, control))
            control_snapshot = {
                key: _snapshot(value, jax=jax, np=np)
                for key, value in control.items()
            }
            control_digests = {
                key: _snapshot_digest(value)
                for key, value in control_snapshot.items()
                if key != "frozen_unchanged"
            }
            payload = {
                "schema_version": 2,
                "gate": 6,
                "phase": "prepare",
                "scope": "production-model-optimizer-grain-orbax",
                "run_id": config.run.run_id,
                "proof_id": attempt_id,
                "launcher_attempt_id": launcher_attempt_id,
                "config_sha256": config.digest,
                "config_file_sha256": config_file_sha256,
                "dataset_manifest_sha256": config.data.manifest_sha256,
                "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
                "checkpoint_step": checkpoint_step,
                "process_index": process_index,
                "process_count": int(jax.process_count()),
                "global_device_count": int(jax.device_count()),
                "local_device_count": int(jax.local_device_count()),
                "process_token": process_token,
                "process_pid": os.getpid(),
                "control_frozen_params_unchanged": bool(
                    np.asarray(
                        control_snapshot["frozen_unchanged"]["entries"][0][3][0][1]
                    )
                ),
                "control_digests": control_digests,
                "topology": topology,
                "sunfish_source": current_source,
            }
            payload["passed"] = payload["control_frozen_params_unchanged"]
            _write_immutable(
                root / f"prepare-host-{process_index:05d}.json", payload
            )
            multihost_utils.sync_global_devices(
                f"sunfish-real-resume-prepare-hosts-{attempt_id}"
            )
            process0_error = None
            if process_index == 0:
                try:
                    hosts = [
                        json.loads(
                            (root / f"prepare-host-{process:05d}.json").read_text()
                        )
                        for process in range(expected_processes)
                    ]
                    summary = verify_real_resume_prepare_evidence(
                        hosts,
                        expected_devices=expected_devices,
                        expected_processes=expected_processes,
                        expected_local_devices=expected_local_devices,
                    )
                    _write_immutable(root / "prepare-summary.json", summary)
                except Exception as error:  # Keep peers out of a stranded barrier.
                    process0_error = f"{type(error).__name__}: {error}"
            process0_error = _broadcast_process0_error(
                multihost_utils, np, process0_error
            )
            _delete_arrays(control_state, jax=jax)
            if process0_error is not None:
                raise RuntimeError(
                    "prepare process-0 finalization failed: " + process0_error
                )
            # Every parent must observe the same merged decision. Otherwise a
            # nonzero process could stop while peers start the resume process
            # and hang its distributed initialization.
            result = json.loads((root / "prepare-summary.json").read_text())
        else:
            if trainer.checkpointer.latest_step != 1:
                raise RuntimeError(
                    "fresh resume process must discover exactly checkpoint step 1"
                )
            prepare_summary_path = root / "prepare-summary.json"
            if not prepare_summary_path.exists():
                raise FileNotFoundError(
                    "fresh resume process cannot find prepare-summary.json"
                )
            prepare_summary = json.loads(prepare_summary_path.read_text())
            prepare_hosts = prepare_summary.get("hosts")
            if not isinstance(prepare_hosts, list):
                raise ValueError("prepare summary has no host evidence")
            verified_prepare = verify_real_resume_prepare_evidence(
                prepare_hosts,
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            )
            if not verified_prepare["passed"] or prepare_summary != verified_prepare:
                raise ValueError("prepare summary failed revalidation")
            expected_prepare_fields = {
                "run_id": config.run.run_id,
                "proof_id": attempt_id,
                "launcher_attempt_id": launcher_attempt_id,
                "config_sha256": config.digest,
                "config_file_sha256": config_file_sha256,
                "dataset_manifest_sha256": config.data.manifest_sha256,
                "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
                "sunfish_source": current_source,
            }
            changed = [
                name
                for name, expected in expected_prepare_fields.items()
                if prepare_summary.get(name) != expected
            ]
            if changed:
                raise ValueError(
                    "fresh resume process differs from preparation: "
                    + ", ".join(changed)
                )
            control = prepare_hosts[process_index]
            if control.get("process_index") != process_index:
                raise ValueError("prepare host evidence is not process-index ordered")
            if control.get("process_token") == process_token:
                raise RuntimeError("prepare and resume reused one Python process token")

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
                step=1,
                donate=True,
            )
            resumed_batch = next(restored_iter)
            resumed_batch = sharding_lib.device_put(
                resumed_batch, trainer.sharding.batch
            )
            resumed_state, resumed = diagnostic_step(restored_state, resumed_batch)
            jax.block_until_ready((resumed_state, resumed))
            resumed_snapshot = {
                key: _snapshot(value, jax=jax, np=np)
                for key, value in resumed.items()
            }
            resumed_digests = {
                key: _snapshot_digest(value)
                for key, value in resumed_snapshot.items()
                if key != "frozen_unchanged"
            }
            control_digests = control["control_digests"]
            exact = {
                key: control_digests.get(key) == resumed_digests.get(key)
                for key in control_digests
            }
            digests = {
                key: {
                    "control": control_digests[key],
                    "resumed": resumed_digests.get(key),
                }
                for key in control_digests
            }
            payload = {
                "schema_version": 2,
                "gate": 6,
                "scope": "production-model-optimizer-grain-orbax",
                "restart_mode": "separate-python-processes",
                "run_id": config.run.run_id,
                "attempt_id": attempt_id,
                "launcher_attempt_id": launcher_attempt_id,
                "config_sha256": config.digest,
                "config_file_sha256": config_file_sha256,
                "dataset_manifest_sha256": config.data.manifest_sha256,
                "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
                "checkpoint_step": 1,
                "prepare_summary_sha256": _path_sha256(prepare_summary_path),
                "prepare_launcher_attempt_id": prepare_summary[
                    "launcher_attempt_id"
                ],
                "prepare_process_token": control["process_token"],
                "resume_process_token": process_token,
                "prepare_process_pid": control["process_pid"],
                "resume_process_pid": os.getpid(),
                "process_index": process_index,
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
                "control_frozen_params_unchanged": control[
                    "control_frozen_params_unchanged"
                ],
                "resumed_frozen_params_unchanged": bool(
                    np.asarray(
                        resumed_snapshot["frozen_unchanged"]["entries"][0][3][0][1]
                    )
                ),
                "digests": digests,
                "topology": topology,
                "sunfish_source": current_source,
            }
            payload["passed"] = all(payload[key] is True for key in _EXACT_KEYS)
            _write_immutable(root / f"host-{process_index:05d}.json", payload)
            multihost_utils.sync_global_devices(
                f"sunfish-real-resume-hosts-{attempt_id}"
            )
            process0_error = None
            if process_index == 0:
                try:
                    hosts = [
                        json.loads((root / f"host-{process:05d}.json").read_text())
                        for process in range(expected_processes)
                    ]
                    summary = verify_real_resume_evidence(
                        hosts,
                        prepare_summary=prepare_summary,
                        expected_devices=expected_devices,
                        expected_processes=expected_processes,
                        expected_local_devices=expected_local_devices,
                    )
                    _write_immutable(root / "summary.json", summary)
                except Exception as error:  # Keep peers out of a stranded barrier.
                    process0_error = f"{type(error).__name__}: {error}"
            process0_error = _broadcast_process0_error(
                multihost_utils, np, process0_error
            )
            _delete_arrays(resumed_state, jax=jax)
            if process0_error is not None:
                raise RuntimeError(
                    "resume process-0 finalization failed: " + process0_error
                )
            result = json.loads((root / "summary.json").read_text())
    finally:
        trainer.checkpointer.close()

    # The prepare process closes every manager before any parent launches the
    # second process. This collective makes the process boundary operationally
    # meaningful rather than merely constructing fresh Python objects.
    multihost_utils.sync_global_devices(
        f"sunfish-real-resume-{process_phase}-closed-{attempt_id}"
    )
    return result


def _phase_command(args: argparse.Namespace, phase: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sunfish_tpu.real_resume_smoke",
        "--config",
        str(args.config),
        "--attempt-id",
        args.attempt_id,
        "--evidence-dir",
        args.evidence_dir,
        "--expected-devices",
        str(args.expected_devices),
        "--expected-processes",
        str(args.expected_processes),
        "--expected-local-devices",
        str(args.expected_local_devices),
        "--process-phase",
        phase,
    ]


def _stop_exact_child(child: Any) -> int:
    """Bound shutdown of the exact process group owned by this orchestrator."""
    process_group = int(child.pid)
    if process_group <= 1:
        raise RuntimeError(f"refusing invalid child process group {process_group}")
    status = child.poll()
    if not _process_group_exists(process_group):
        return int(status) if status is not None else -int(signal.SIGKILL)

    _signal_process_group(process_group, signal.SIGTERM)
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        if child.poll() is None:
            try:
                child.wait(timeout=min(_WAIT_POLL_SECONDS, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            # The direct phase process may exit while a Grain/Kauldron worker
            # remains in its session. Keep the same bounded grace for those
            # descendants before escalating the owned group.
            time.sleep(min(_WAIT_POLL_SECONDS, remaining))

    if _process_group_exists(process_group):
        _signal_process_group(process_group, signal.SIGKILL)
    if child.poll() is None:
        try:
            status = child.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            # SIGKILL is already pending. Do not let a kernel-stuck group hold
            # the all-host SSH launcher forever.
            return -int(signal.SIGKILL)
    else:
        status = child.poll()
    kill_deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while _process_group_exists(process_group) and time.monotonic() < kill_deadline:
        time.sleep(
            min(_WAIT_POLL_SECONDS, max(0.0, kill_deadline - time.monotonic()))
        )
    if _process_group_exists(process_group):
        # A direct child exit is not success while any member of its owned
        # session remains. Returning a signal-style failure prevents the
        # orchestrator from launching the resume half onto an occupied TPU.
        return -int(signal.SIGKILL)
    return int(status) if status is not None else -int(signal.SIGKILL)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # This should not happen for our own session, but fail closed and let
        # the real signal produce the actionable permission error.
        return True
    return True


def _signal_process_group(process_group: int, signum: int) -> bool:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        return False
    return True


def _run_child_process(command: Sequence[str], *, phase: str) -> int:
    env = os.environ.copy()
    env[_ORCHESTRATED_ENV] = "1"
    env[_PROCESS_TOKEN_ENV] = secrets.token_hex(32)
    child: Any = None
    received_signal: int | None = None
    shutdown_deadline: float | None = None
    forwarded_signals: set[int] = set()
    previous_handlers: dict[int, Any] = {}
    cleanup_returncode: int | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal, shutdown_deadline
        if received_signal is None:
            received_signal = signum
            shutdown_deadline = time.monotonic() + _SIGNAL_GRACE_SECONDS
        if child is not None and _signal_process_group(int(child.pid), signum):
            forwarded_signals.add(signum)

    # Install handlers before spawning. A signal delivered while Popen is
    # returning is then latched and forwarded as soon as the child reference is
    # available, so no workload process can escape as an orphan.
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    try:
        if received_signal is not None:
            # A signal can land after handlers are installed but before Popen.
            # Do not create a workload that is already required to shut down.
            returncode = 128 + received_signal
        else:
            # The phase process owns a new session/process group. Signals and
            # the bounded TERM->KILL path therefore cover Grain/Kauldron
            # descendants, never the controller or an unrelated user process.
            child = subprocess.Popen(command, env=env, start_new_session=True)
            if (
                received_signal is not None
                and received_signal not in forwarded_signals
            ):
                if _signal_process_group(int(child.pid), received_signal):
                    forwarded_signals.add(received_signal)
            while True:
                try:
                    returncode = child.wait(timeout=_WAIT_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if (
                        shutdown_deadline is not None
                        and time.monotonic() >= shutdown_deadline
                    ):
                        returncode = _stop_exact_child(child)
                        break
    finally:
        try:
            # Reap the exact child before restoring default handlers. Otherwise
            # a signal in the cleanup window could kill this parent and orphan it.
            if child is not None:
                cleanup_returncode = _stop_exact_child(child)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    if received_signal is not None:
        # A shutdown request is authoritative even if the child handles it and
        # reports success or happened to exit just before forwarding. Never
        # proceed into the resume phase after TERM/HUP/INT.
        returncode = 128 + received_signal
    elif cleanup_returncode is not None and cleanup_returncode != 0:
        # Even a direct-child success cannot advance to resume if cleanup did
        # not prove the phase's exact process group absent.
        returncode = cleanup_returncode
    if returncode < 0:
        returncode = 128 - returncode
    if returncode:
        print(
            f"sunfish-real-resume-smoke: {phase} process exited {returncode}",
            file=sys.stderr,
        )
    return returncode


def _run_orchestrated_restart(args: argparse.Namespace) -> int:
    for phase in _PROCESS_PHASES:
        returncode = _run_child_process(_phase_command(args, phase), phase=phase)
        if returncode:
            return returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    parser.add_argument(
        "--process-phase",
        choices=_PROCESS_PHASES,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.process_phase is None:
        return _run_orchestrated_restart(args)
    try:
        payload = run_real_resume_phase(
            config_path=args.config,
            attempt_id=args.attempt_id,
            evidence_dir=args.evidence_dir,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
            process_phase=args.process_phase,
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"sunfish-real-resume-smoke: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
