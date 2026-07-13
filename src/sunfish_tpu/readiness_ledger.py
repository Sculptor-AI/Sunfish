"""Fail-closed merger for the eight ordered Stage-0.5 readiness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from sunfish_tpu.parity_evidence import validate_stage0_parity_payload
from sunfish_tpu.checkpoint_smoke import verify_checkpoint_evidence
from sunfish_tpu.input_smoke import verify_evidence as verify_input_evidence
from sunfish_tpu.real_resume_smoke import verify_real_resume_evidence
from sunfish_tpu.seed_load_smoke import verify_seed_load_evidence
from sunfish_tpu.source_identity import normalize_source_identity
from sunfish_tpu.topology_smoke import verify_topology_evidence
from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STAGE05_MODEL = {
    "num_experts": 32,
    "top_k_experts": 4,
    "vocab_size": 262144,
    "hidden_size": 2816,
    "num_layers": 30,
    "expert_hidden_size": 704,
    "dtype": "bfloat16",
}
_TPU_RUNTIME_VERSIONS = {**RUNTIME_VERSIONS, **TPU_ONLY_RUNTIME_VERSIONS}
_EVIDENCE_KEYS = {
    "topology",
    "input",
    "seed_load",
    "smoke",
    "checkpoint",
    "real_resume",
    "preemption",
    "run_identity",
    "preemption_run_identity",
    "stage0_parity",
    "config_bundle",
}


def validate_readiness_unlock(
    payload: Any,
    *,
    expected_source: Mapping[str, Any],
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    """Validate a persisted all-pass ledger before any later TPU stage."""
    if not isinstance(payload, Mapping):
        raise ValueError("readiness ledger is not a JSON object")
    if (
        payload.get("schema_version") != 1
        or payload.get("purpose") != "stage-0.5-tpu-readiness-ledger"
    ):
        raise ValueError("readiness ledger schema/purpose differs")
    if payload.get("passed") is not True or payload.get("errors") != []:
        raise ValueError("readiness ledger is not an error-free pass")
    expected_source_normalized = normalize_source_identity(expected_source)
    if (
        expected_source_normalized is None
        or normalize_source_identity(payload.get("sunfish_source"))
        != expected_source_normalized
    ):
        raise ValueError("readiness ledger source identity differs")
    expected_topology = {
        "device_count": expected_devices,
        "process_count": expected_processes,
        "local_device_count": expected_local_devices,
    }
    if payload.get("topology") != expected_topology:
        raise ValueError("readiness ledger topology differs")
    gates = payload.get("ordered_gates")
    if not isinstance(gates, Mapping) or list(gates) != [str(i) for i in range(1, 9)]:
        raise ValueError("readiness ledger does not contain gates 1 through 8 in order")
    for number, gate in gates.items():
        if (
            not isinstance(gate, Mapping)
            or gate.get("passed") is not True
            or not isinstance(gate.get("source"), str)
            or not gate.get("source")
        ):
            raise ValueError(f"readiness ledger gate {number} is not passed")
    for key in (
        "config_sha256",
        "preemption_config_sha256",
        "dataset_manifest_sha256",
        "seed_manifest_sha256",
        "config_bundle_sha256",
    ):
        if not _SHA256.fullmatch(str(payload.get(key, ""))):
            raise ValueError(f"readiness ledger {key} is invalid")
    parity = payload.get("stage0_parity")
    if not isinstance(parity, Mapping):
        raise ValueError("readiness ledger has no Stage-0 parity pin")
    if (
        parity.get("filename") != "stage0-parity-report.json"
        or parity.get("stage") != "stage-0-parity"
        or parity.get("p1_tensors_compared") != 691
        or not _SHA256.fullmatch(str(parity.get("report_sha256", "")))
        or not _SHA256.fullmatch(str(parity.get("checks_sha256", "")))
        or normalize_source_identity(parity.get("sunfish_source"))
        != expected_source_normalized
    ):
        raise ValueError("readiness ledger Stage-0 parity pin is invalid")
    provenance = payload.get("evidence")
    if not isinstance(provenance, Mapping) or set(provenance) != _EVIDENCE_KEYS:
        raise ValueError("readiness ledger evidence inventory differs")
    for name, item in provenance.items():
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not item.get("path")
            or not _SHA256.fullmatch(str(item.get("sha256", "")))
        ):
            raise ValueError(f"readiness ledger evidence pin is invalid for {name}")
    return dict(payload)


def verify_readiness_ledger(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
    evidence_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    required = _EVIDENCE_KEYS
    if set(evidence) != required:
        raise ValueError(
            f"readiness evidence keys differ: missing={sorted(required - set(evidence))}, "
            f"extra={sorted(set(evidence) - required)}"
        )
    if min(expected_devices, expected_processes, expected_local_devices) <= 0:
        raise ValueError("expected topology counts must be positive")
    if expected_processes * expected_local_devices != expected_devices:
        raise ValueError(
            "expected processes * local devices must equal global devices"
        )

    errors: list[str] = []
    gates: dict[str, dict[str, Any]] = {}

    def gate(number: int, passed: bool, source: str, detail: str) -> None:
        if not passed:
            errors.append(f"gate {number} failed: {detail}")
        gates[str(number)] = {
            "passed": bool(passed),
            "source": source,
            "detail": detail,
        }

    identity = evidence["run_identity"]
    preemption_identity = evidence["preemption_run_identity"]
    expected_source = normalize_source_identity(identity.get("sunfish_source"))
    if expected_source is None:
        errors.append("smoke run identity has no valid source identity")

    def validate_run_identity(value: Mapping[str, Any], label: str) -> None:
        if value.get("schema_version") != 1 or value.get("phase") != "smoke":
            errors.append(f"{label} is not a schema-1 smoke identity")
        if not _RUN_ID.fullmatch(str(value.get("run_id", ""))):
            errors.append(f"{label} run ID is invalid")
        for key in (
            "config_sha256",
            "config_file_sha256",
            "dataset_manifest_sha256",
            "init_checkpoint_manifest_sha256",
        ):
            if not _SHA256.fullmatch(str(value.get(key, ""))):
                errors.append(f"{label} {key} is not a SHA-256")
        for key in ("init_checkpoint", "init_checkpoint_manifest"):
            uri = value.get(key)
            if not isinstance(uri, str) or not uri.startswith("gs://"):
                errors.append(f"{label} {key} is not a GCS URI")
        if value.get("init_checkpoint_format") != "orbax-exact-tree":
            errors.append(f"{label} checkpoint format differs")
        if value.get("init_checkpoint_step") != -1:
            errors.append(f"{label} exact-tree seed step differs")
        if value.get("model") != _STAGE05_MODEL:
            errors.append(f"{label} Stage-0.5 model contract differs")
        if value.get("runtime_versions") != _TPU_RUNTIME_VERSIONS:
            errors.append(f"{label} TPU runtime versions differ")
        if value.get("gemma_source_commit") != GEMMA_SOURCE_COMMIT:
            errors.append(f"{label} Gemma source commit differs")
        if normalize_source_identity(value.get("sunfish_source")) != expected_source:
            errors.append(f"{label} source identity differs")

    validate_run_identity(identity, "smoke run identity")
    validate_run_identity(preemption_identity, "preemption run identity")

    def require_recomputed_summary(name: str, recomputed: Mapping[str, Any]) -> None:
        if dict(evidence[name]) != dict(recomputed):
            errors.append(f"{name} summary does not match its embedded host evidence")

    try:
        require_recomputed_summary(
            "topology",
            verify_topology_evidence(
                evidence["topology"].get("hosts", ()),
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            ),
        )
        require_recomputed_summary(
            "input",
            verify_input_evidence(
                list(evidence["input"].get("hosts", ())),
                total_records=int(evidence["input"].get("total_records", -1)),
                expected_processes=expected_processes,
            ),
        )
        require_recomputed_summary(
            "seed_load",
            verify_seed_load_evidence(
                evidence["seed_load"].get("hosts", ()),
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            ),
        )
        require_recomputed_summary(
            "checkpoint",
            verify_checkpoint_evidence(
                list(evidence["checkpoint"].get("hosts", ())),
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            ),
        )
        require_recomputed_summary(
            "real_resume",
            verify_real_resume_evidence(
                evidence["real_resume"].get("hosts", ()),
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            ),
        )
    except (TypeError, ValueError) as error:
        errors.append(f"embedded host evidence cannot be verified: {error}")

    parity_summary = None
    try:
        parity_summary = validate_stage0_parity_payload(
            evidence["stage0_parity"],
            expected_source=identity.get("sunfish_source", {}),
        )
    except ValueError as error:
        errors.append(str(error))

    hashes = evidence_sha256 or {}
    parity_report_sha256 = hashes.get("stage0_parity")
    if not _SHA256.fullmatch(str(parity_report_sha256 or "")):
        errors.append("Stage-0 parity evidence has no exact raw-file hash")
    config_bundle_sha256 = hashes.get("config_bundle")
    if not _SHA256.fullmatch(str(config_bundle_sha256 or "")):
        errors.append("rendered config bundle has no exact raw-file hash")

    bundle = evidence["config_bundle"]
    if (
        bundle.get("schema_version") != 1
        or bundle.get("purpose") != "stage-0.5-rendered-config-bundle"
    ):
        errors.append("rendered config bundle schema/purpose differs")
    if normalize_source_identity(bundle.get("sunfish_source")) != expected_source:
        errors.append("rendered config bundle source differs from the smoke run")
    if bundle.get("topology") != {
        "global_devices": expected_devices,
        "processes": expected_processes,
        "local_devices": expected_local_devices,
    }:
        errors.append("rendered config bundle topology differs")
    parity_pin = bundle.get("stage0_parity")
    expected_parity_pin = (
        {
            "filename": "stage0-parity-report.json",
            **parity_summary,
            "report_sha256": parity_report_sha256,
        }
        if parity_summary is not None
        else None
    )
    if parity_pin != expected_parity_pin:
        errors.append("rendered config bundle Stage-0 parity pin differs")

    topology = evidence["topology"]
    topology_ok = topology.get("passed") is True and topology.get("gate") == 1
    expected = topology.get("expected", {})
    topology_ok = topology_ok and expected == {
        "global_device_count": expected_devices,
        "process_count": expected_processes,
        "local_device_count": expected_local_devices,
    }
    gate(1, topology_ok, "topology", "all-host topology and collective")

    input_summary = evidence["input"]
    input_ok = (
        input_summary.get("ready") is True
        and input_summary.get("expected_processes") == expected_processes
        and input_summary.get("records_observed") == input_summary.get("total_records")
    )
    gate(2, input_ok, "input", "process-disjoint exhaustive GCS epoch")

    seed_load = evidence["seed_load"]
    seed_ok = (
        seed_load.get("passed") is True
        and seed_load.get("gate") == 3
        and seed_load.get("scope") == "real-8b-orbax-seed-target-sharded-restore"
        and len(seed_load.get("hosts", ())) == expected_processes
    )
    gate(3, seed_ok, "seed_load", "real 8B exact-tree target-sharded restore")

    smoke = evidence["smoke"]
    smoke_gate4 = smoke.get("gates", {}).get("4", {})
    smoke_gate4_contract_ok = all(
        (
            smoke_gate4.get("errors") == [],
            isinstance(smoke_gate4.get("metric_steps"), int)
            and smoke_gate4.get("metric_steps") >= 100,
            isinstance(smoke_gate4.get("relative_loss_reduction"), (int, float))
            and math.isfinite(float(smoke_gate4.get("relative_loss_reduction")))
            and float(smoke_gate4.get("relative_loss_reduction")) >= 0.10,
            isinstance(smoke_gate4.get("required_relative_loss_reduction"), (int, float))
            and float(smoke_gate4.get("required_relative_loss_reduction")) >= 0.10,
            isinstance(smoke_gate4.get("max_gradient_norm"), (int, float))
            and math.isfinite(float(smoke_gate4.get("max_gradient_norm")))
            and float(smoke_gate4.get("max_gradient_norm")) > 0.0,
            isinstance(smoke_gate4.get("max_update_norm"), (int, float))
            and math.isfinite(float(smoke_gate4.get("max_update_norm")))
            and float(smoke_gate4.get("max_update_norm")) > 0.0,
        )
    )
    smoke_lineage_ok = all(
        (
            smoke.get("schema_version") == 1,
            smoke.get("passed") is True,
            isinstance(smoke.get("attempt_id"), str) and bool(smoke.get("attempt_id")),
            smoke.get("expected_processes") == expected_processes,
            smoke.get("run_id") == identity.get("run_id"),
            smoke.get("config_sha256") == identity.get("config_sha256"),
            smoke.get("dataset_manifest_sha256")
            == identity.get("dataset_manifest_sha256"),
            smoke.get("seed_manifest_sha256")
            == identity.get("init_checkpoint_manifest_sha256"),
            normalize_source_identity(smoke.get("sunfish_source"))
            == expected_source,
        )
    )
    gate(
        4,
        smoke_lineage_ok
        and smoke_gate4.get("passed") is True
        and smoke_gate4_contract_ok,
        "smoke",
        "real 100-500-step tiny-dataset overfit with nonzero updates",
    )

    checkpoint = evidence["checkpoint"]
    checkpoint_ok = (
        checkpoint.get("passed") is True
        and 5 in checkpoint.get("gates", ())
        and checkpoint.get("scope") == "synthetic-sharded-state"
        and len(checkpoint.get("hosts", ())) == expected_processes
    )
    gate(5, checkpoint_ok, "checkpoint", "distributed Orbax state round trip")

    real_resume = evidence["real_resume"]
    resume_ok = (
        real_resume.get("passed") is True
        and real_resume.get("gate") == 6
        and real_resume.get("scope") == "production-model-optimizer-grain-orbax"
        and len(real_resume.get("hosts", ())) == expected_processes
    )
    gate(6, resume_ok, "real_resume", "production next-step exact restart")

    preemption = evidence["preemption"]
    preemption_ok = all(
        (
            preemption.get("passed") is True,
            preemption.get("schema_version") == 1,
            preemption.get("gate") == 7,
            preemption.get("finalized_checkpoint_survived") is True,
            preemption.get("automatic_same_workdir_restore") is True,
            preemption.get("resume_continued_from_checkpoint") is True,
            preemption.get("fresh_start_metric_absent") is True,
            preemption.get("manual_gcs_cleanup_performed") is False,
            preemption.get("train_complete_found") is True,
            preemption.get("final_checkpoint_found") is True,
            int(preemption.get("preempted_launch_returncode", 0)) != 0,
            int(preemption.get("resumed_launch_returncode", -1)) == 0,
            _SHA256.fullmatch(str(preemption.get("resumed_output_sha256", "")))
            is not None,
            _SHA256.fullmatch(str(preemption.get("preempted_output_sha256", "")))
            is not None,
        )
    )
    gate(
        7,
        preemption_ok,
        "recovery",
        "exact user-process interruption and automatic same-workdir recovery; TPU VM untouched",
    )

    smoke_gate8 = smoke.get("gates", {}).get("8", {})
    p95_wait_ratio = smoke_gate8.get("p95_input_wait_ratio")
    max_p95_wait_ratio = smoke_gate8.get("max_p95_input_wait_ratio")
    smoke_gate8_contract_ok = all(
        (
            smoke_gate8.get("errors") == [],
            isinstance(smoke_gate8.get("steady_state_steps"), list)
            and bool(smoke_gate8.get("steady_state_steps")),
            isinstance(p95_wait_ratio, (int, float))
            and math.isfinite(float(p95_wait_ratio))
            and float(p95_wait_ratio) <= 0.10,
            isinstance(max_p95_wait_ratio, (int, float))
            and float(max_p95_wait_ratio) <= 0.10,
            smoke_gate8.get("local_cache_policy")
            == "none-direct-gcs-range-reads",
        )
    )
    gate(
        8,
        smoke_lineage_ok
        and smoke_gate8.get("passed") is True
        and smoke_gate8_contract_ok,
        "smoke",
        "steady-state p95 input-wait ratio and zero local cache",
    )

    if normalize_source_identity(
        preemption_identity.get("sunfish_source")
    ) != expected_source:
        errors.append("gate-7 source tree differs from the smoke run")
    for name in (
        "topology",
        "input",
        "seed_load",
        "smoke",
        "checkpoint",
        "real_resume",
    ):
        if normalize_source_identity(evidence[name].get("sunfish_source")) != expected_source:
            errors.append(f"{name} evidence source tree differs from the smoke run")
    if identity.get("topology") != {
        "device_count": expected_devices,
        "process_count": expected_processes,
        "local_device_count": expected_local_devices,
    }:
        errors.append("run identity topology differs from the granted slice")

    identity_dataset = identity.get("dataset_manifest_sha256")
    if bundle.get("dataset_manifest_sha256") != identity_dataset:
        errors.append("rendered config bundle dataset differs from the smoke run")
    input_manifests = {
        host.get("manifest_sha256") for host in input_summary.get("hosts", ())
    }
    if input_manifests != {identity_dataset}:
        errors.append("gate-2 dataset differs from the smoke run identity")
    identity_seed = identity.get("init_checkpoint_manifest_sha256")
    if bundle.get("seed_manifest_sha256") != identity_seed:
        errors.append("rendered config bundle seed differs from the smoke run")
    seed_hashes = {
        host.get("seed_manifest_sha256") for host in seed_load.get("hosts", ())
    }
    if seed_hashes != {identity_seed}:
        errors.append("gate-3 seed differs from the smoke run identity")
    if {host.get("seed_path") for host in seed_load.get("hosts", ())} != {
        identity.get("init_checkpoint")
    }:
        errors.append("gate-3 seed path differs from the smoke run identity")
    if {
        host.get("seed_manifest_path") for host in seed_load.get("hosts", ())
    } != {identity.get("init_checkpoint_manifest")}:
        errors.append("gate-3 seed manifest path differs from the smoke run identity")
    if not _SHA256.fullmatch(
        str(seed_load.get("seed_gcs_inventory_sha256", ""))
    ):
        errors.append("gate-3 seed GCS inventory hash is invalid")
    preemption_plan = preemption.get("plan", {})
    if preemption_plan.get("config_sha256") != preemption_identity.get(
        "config_sha256"
    ):
        errors.append("gate-7 config differs from its run identity")
    resume_proof = preemption.get("resume_proof", {})
    for proof_key, identity_key in (
        ("run_id", "run_id"),
        ("config_sha256", "config_sha256"),
        ("dataset_manifest_sha256", "dataset_manifest_sha256"),
        ("seed_manifest_sha256", "init_checkpoint_manifest_sha256"),
    ):
        if resume_proof.get(proof_key) != preemption_identity.get(identity_key):
            errors.append(f"gate-7 resume proof differs for {proof_key}")
    if resume_proof.get("attempt_id") != preemption_plan.get("resume_attempt"):
        errors.append("gate-7 resume proof attempt differs from the plan")
    if resume_proof.get("step") != preemption_plan.get("preempt_after_step"):
        errors.append("gate-7 resume proof did not continue at the checkpoint step")
    if normalize_source_identity(resume_proof.get("sunfish_source")) != expected_source:
        errors.append("gate-7 resume proof source differs from the smoke run")
    if not _SHA256.fullmatch(str(resume_proof.get("metric_sha256", ""))):
        errors.append("gate-7 resume proof metric hash is invalid")
    for key in (
        "dataset_manifest_sha256",
        "init_checkpoint",
        "init_checkpoint_format",
        "init_checkpoint_step",
        "init_checkpoint_manifest",
        "init_checkpoint_manifest_sha256",
        "model",
        "runtime_versions",
        "gemma_source_commit",
        "topology",
    ):
        if preemption_identity.get(key) != identity.get(key):
            errors.append(f"gate-7 identity differs from the smoke run for {key}")
    if preemption_identity.get("run_id") == identity.get("run_id"):
        errors.append("gate-7 must use a fresh run ID and workdir")

    real_hosts = real_resume.get("hosts", ())
    if {host.get("dataset_manifest_sha256") for host in real_hosts} != {
        identity_dataset
    }:
        errors.append("gate-6 dataset differs from the smoke run identity")
    if {host.get("seed_manifest_sha256") for host in real_hosts} != {
        identity_seed
    }:
        errors.append("gate-6 seed differs from the smoke run identity")

    configs = bundle.get("configs")
    if not isinstance(configs, Mapping) or set(configs) != {
        "sunfish-smoke.toml",
        "sunfish-resume-smoke.toml",
        "sunfish-preemption-smoke.toml",
    }:
        errors.append("rendered config bundle entries differ")
        configs = {}

    def require_config_pin(
        filename: str, *, run_id: Any, config_sha256: Any, config_file_sha256: Any
    ) -> None:
        pin = configs.get(filename)
        if not isinstance(pin, Mapping):
            errors.append(f"rendered config bundle has no {filename} pin")
            return
        if pin.get("run_id") != run_id:
            errors.append(f"rendered {filename} run ID differs")
        if pin.get("config_sha256") != config_sha256:
            errors.append(f"rendered {filename} canonical digest differs")
        if pin.get("config_file_sha256") != config_file_sha256:
            errors.append(f"rendered {filename} raw-file digest differs")

    require_config_pin(
        "sunfish-smoke.toml",
        run_id=identity.get("run_id"),
        config_sha256=identity.get("config_sha256"),
        config_file_sha256=identity.get("config_file_sha256"),
    )
    require_config_pin(
        "sunfish-resume-smoke.toml",
        run_id=real_resume.get("run_id"),
        config_sha256=real_resume.get("config_sha256"),
        config_file_sha256=real_resume.get("config_file_sha256"),
    )
    require_config_pin(
        "sunfish-preemption-smoke.toml",
        run_id=preemption_identity.get("run_id"),
        config_sha256=preemption_identity.get("config_sha256"),
        config_file_sha256=preemption_identity.get("config_file_sha256"),
    )
    preemption_config_pin = configs.get("sunfish-preemption-smoke.toml", {})
    if isinstance(preemption_config_pin, Mapping):
        if preemption_plan.get("run_id") != preemption_config_pin.get("run_id"):
            errors.append("gate-7 plan run ID differs from the rendered config")
        if preemption_plan.get("workdir") != preemption_config_pin.get("workdir"):
            errors.append("gate-7 plan workdir differs from the rendered config")
        workdir = str(preemption_config_pin.get("workdir", "")).rstrip("/")
        step = preemption_plan.get("preempt_after_step")
        resume_attempt = preemption_plan.get("resume_attempt")
        expected_plan_paths = {
            "preempt_marker": f"{workdir}/checkpoints/ckpt_{step}/commit_success.txt",
            "final_marker": f"{workdir}/checkpoints/ckpt_100/commit_success.txt",
            "train_complete": f"{workdir}/train_complete.txt",
            "resume_first_metric": (
                f"{workdir}/readiness/{resume_attempt}/metrics/step-{int(step):09d}.json"
                if isinstance(step, int)
                else "invalid"
            ),
            "fresh_start_metric": (
                f"{workdir}/readiness/{resume_attempt}/metrics/step-000000000.json"
            ),
        }
        for key, expected in expected_plan_paths.items():
            if preemption_plan.get(key) != expected:
                errors.append(f"gate-7 plan {key} differs from the rendered run")
    if preemption_plan.get("preempt_attempt") == preemption_plan.get("resume_attempt"):
        errors.append("gate-7 preempt and resume attempts are not distinct")
    run_ids = {
        pin.get("run_id")
        for pin in configs.values()
        if isinstance(pin, Mapping)
    }
    workdirs = {
        pin.get("workdir")
        for pin in configs.values()
        if isinstance(pin, Mapping)
    }
    if len(run_ids) != 3 or len(workdirs) != 3 or None in run_ids or None in workdirs:
        errors.append("rendered readiness run IDs/workdirs are not isolated")
    checkpoint_destination = checkpoint.get("destination")
    readiness_prefix = f"{str(bundle.get('storage_root', '')).rstrip('/')}/readiness/"
    if not isinstance(checkpoint_destination, str) or not checkpoint_destination.startswith(
        readiness_prefix
    ):
        errors.append("gate-5 checkpoint destination is outside the readiness prefix")

    return {
        "schema_version": 1,
        "purpose": "stage-0.5-tpu-readiness-ledger",
        "passed": not errors,
        "errors": errors,
        "ordered_gates": gates,
        "run_id": identity.get("run_id"),
        "config_sha256": identity.get("config_sha256"),
        "preemption_config_sha256": preemption_identity.get("config_sha256"),
        "dataset_manifest_sha256": identity_dataset,
        "seed_manifest_sha256": identity_seed,
        "topology": identity.get("topology"),
        "sunfish_source": identity.get("sunfish_source"),
        "stage0_parity": expected_parity_pin,
        "config_bundle_sha256": config_bundle_sha256,
    }


def _read_with_hash(path: Any) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"readiness evidence must be a JSON object: {path}")
    return parsed, hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "topology",
        "input",
        "seed-load",
        "smoke",
        "checkpoint",
        "real-resume",
        "preemption",
        "run-identity",
        "preemption-run-identity",
        "stage0-parity",
        "config-bundle",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    parser.add_argument("--output", required=True, help="immutable local or gs:// JSON object")
    args = parser.parse_args(argv)

    from etils import epath

    paths = {
        "topology": args.topology,
        "input": args.input,
        "seed_load": args.seed_load,
        "smoke": args.smoke,
        "checkpoint": args.checkpoint,
        "real_resume": args.real_resume,
        "preemption": args.preemption,
        "run_identity": args.run_identity,
        "preemption_run_identity": args.preemption_run_identity,
        "stage0_parity": args.stage0_parity,
        "config_bundle": args.config_bundle,
    }
    try:
        evidence = {}
        provenance = {}
        for name, raw_path in paths.items():
            evidence[name], digest = _read_with_hash(epath.Path(raw_path))
            provenance[name] = {"path": raw_path, "sha256": digest}
        payload = verify_readiness_ledger(
            evidence,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
            evidence_sha256={
                name: item["sha256"] for name, item in provenance.items()
            },
        )
        payload["evidence"] = provenance
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        output = epath.Path(args.output)
        if output.exists() and output.read_text() != encoded:
            raise FileExistsError(f"immutable readiness ledger changed at {output}")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"sunfish-readiness-ledger: {error}", file=sys.stderr)
        return 2
    print(encoded, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
