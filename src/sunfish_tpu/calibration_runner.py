"""Distributed Stage-1 full-teacher router calibration runner.

The 25B text teacher is sharded over a measured ``(data, expert)`` mesh with
one data replica per host and the expert collective confined to local devices.
Each collective step consumes exactly one fixed-shape record per process. Mass
is accumulated globally on device and process 0 alone writes immutable flushes;
restart resumes from the last completed flush boundary without double-counting.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import hashlib
import json
import math
import operator
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
from sunfish_tpu.source_identity import (
    normalize_source_identity,
    require_launcher_run_id,
    source_identity_from_environment,
)
from sunfish_tpu.stage1_contract import (
    MASS_COVERAGE_32,
    MASS_COVERAGE_48,
    MIN_CALIBRATION_INPUT_TOKENS,
    RECONSTRUCTION_SAMPLE_TOKENS,
    validate_calibration_source_contract,
)
from sunfish_tpu.calibration_data_inventory import (
    verify_live_calibration_data_inventory,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GCS_REVISION = re.compile(r"^gcs-inventory-sha256:[0-9a-f]{64}$")
_FLUSH = re.compile(r"router_stats\.host0\.shard(\d+)-step(\d+)\.json$")
PHASES = ("prefill", "denoise_high", "denoise_mid", "denoise_low")
WORKLOADS = (
    "code_completion",
    "repo_edit",
    "tool_calls",
    "agent_trajectory",
    "general_control",
    "reasoning_control",
)
NOISE_RATES = (0.85, 0.50, 0.15)
RECONSTRUCTION_FIELDS = (
    "shared_pre_router_residual",
    "topk_indices",
    "final_scaled_topk_weights",
)


def calibration_bucket_names() -> list[str]:
    """Canonical 60-bucket taxonomy required by PLAN Stage 1."""
    names = [f"prefill/{workload}" for workload in WORKLOADS]
    names.extend(
        f"{phase}/{workload}/pos{position}"
        for phase in PHASES[1:]
        for workload in WORKLOADS
        for position in range(3)
    )
    return names


def pack_calibration_tokens(
    token_ids: Sequence[int],
    *,
    prompt_length: int,
    canvas_size: int,
    pad_token: int,
    vocab_size: int,
) -> dict[str, list[int] | list[bool] | int]:
    """Split one document into fixed prompt/canvas shapes without truncation ambiguity."""
    tokens = [int(token) for token in token_ids]
    if len(tokens) < 2:
        raise ValueError("calibration records need at least two tokens")
    if any(token < 0 or token >= vocab_size for token in tokens):
        raise ValueError("calibration token is outside the vocabulary")
    if len(tokens) <= prompt_length + canvas_size:
        split = min(
            prompt_length,
            max(1, len(tokens) // 2, len(tokens) - canvas_size),
        )
    else:
        split = prompt_length
    prompt_tokens = tokens[:split]
    canvas_tokens = tokens[split : split + canvas_size]
    if not canvas_tokens:
        raise ValueError("calibration record produced an empty canvas")
    prompt = prompt_tokens + [pad_token] * (prompt_length - len(prompt_tokens))
    canvas = canvas_tokens + [pad_token] * (canvas_size - len(canvas_tokens))
    return {
        "prompt": prompt,
        "prompt_mask": [True] * len(prompt_tokens)
        + [False] * (prompt_length - len(prompt_tokens)),
        "canvas": canvas,
        "canvas_mask": [True] * len(canvas_tokens)
        + [False] * (canvas_size - len(canvas_tokens)),
        "prompt_tokens": len(prompt_tokens),
        "canvas_tokens": len(canvas_tokens),
    }


def usable_record_count(
    total_records: int, *, process_count: int, max_records: int = 0
) -> int:
    if total_records <= 0 or process_count <= 0 or max_records < 0:
        raise ValueError("record/process counts are invalid")
    requested = min(total_records, max_records) if max_records else total_records
    usable = requested - (requested % process_count)
    if usable <= 0:
        raise ValueError("dataset has fewer usable records than processes")
    return usable


def calibration_completion_status(
    *,
    total_records: int,
    process_count: int,
    processed_records: int,
    max_records: int,
    observed_input_tokens: int,
    minimum_input_tokens: int,
) -> dict[str, Any]:
    """Classify a completed collector as gating or debug-only.

    ``max_records`` is deliberately a pilot/debug control.  Even when its cap
    happens to include the complete dataset, setting it makes the run
    non-promotable so an operator cannot mistake a bounded invocation for the
    precommitted full-corpus Stage-1 gate.
    """
    if max_records < 0 or observed_input_tokens < 0 or minimum_input_tokens < 0:
        raise ValueError("calibration completion counts are invalid")
    full_usable_records = usable_record_count(
        total_records, process_count=process_count
    )
    expected_processed_records = usable_record_count(
        total_records,
        process_count=process_count,
        max_records=max_records,
    )
    if processed_records != expected_processed_records:
        raise ValueError("processed calibration records are outside the usable corpus")
    debug_run = max_records > 0
    full_corpus_consumed = (
        not debug_run and processed_records == full_usable_records
    )
    minimum_tokens_observed = observed_input_tokens >= minimum_input_tokens
    return {
        "run_mode": "debug-capped" if debug_run else "full",
        "debug_run": debug_run,
        "max_records": max_records,
        "source_records": total_records,
        "full_usable_records": full_usable_records,
        "processed_records": processed_records,
        "full_corpus_consumed": full_corpus_consumed,
        "observed_input_tokens": observed_input_tokens,
        "minimum_observed_input_tokens": minimum_input_tokens,
        "minimum_tokens_observed": minimum_tokens_observed,
        "promotion_eligible": full_corpus_consumed and minimum_tokens_observed,
    }


def mass_candidate_payload(
    results: Sequence[Any],
    *,
    min_coverage: float,
    source_revision: str,
    dataset_manifest_sha256: str,
    corpus_gcs_inventory_sha256: str,
    calibration_run_sha256: str,
    completion: Mapping[str, Any],
    retained_experts: int = 32,
    top_k_experts: int = 8,
    sunfish_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(results) != 30:
        raise ValueError("mass candidate requires exactly 30 layers")
    if not 0 < top_k_experts <= retained_experts <= 128:
        raise ValueError("mass candidate expert counts are invalid")
    if any(len(result.selected) != retained_experts for result in results):
        raise ValueError("mass candidate selection width differs from retained experts")
    if normalize_source_identity(sunfish_source) is None:
        raise ValueError("mass candidate requires a valid Sunfish source identity")
    if not _SHA256.fullmatch(calibration_run_sha256) or not _SHA256.fullmatch(
        corpus_gcs_inventory_sha256
    ):
        raise ValueError(
            "mass candidate requires calibration-run and corpus-inventory digests"
        )
    required_completion = {
        "run_mode",
        "debug_run",
        "full_usable_records",
        "processed_records",
        "full_corpus_consumed",
        "observed_input_tokens",
        "minimum_observed_input_tokens",
        "minimum_tokens_observed",
        "promotion_eligible",
    }
    if not required_completion <= completion.keys():
        raise ValueError("mass candidate lacks calibration completion evidence")
    debug_run = completion.get("debug_run") is True
    promotion_eligible = completion.get("promotion_eligible") is True
    expected_eligibility = (
        not debug_run
        and completion.get("run_mode") == "full"
        and completion.get("full_corpus_consumed") is True
        and completion.get("minimum_tokens_observed") is True
    )
    if promotion_eligible != expected_eligibility:
        raise ValueError("mass candidate has inconsistent promotion eligibility")
    if debug_run:
        purpose_suffix = "debug-non-promotable"
    elif not promotion_eligible:
        purpose_suffix = "incomplete-non-promotable"
    else:
        purpose_suffix = "pending-reconstruction"
    return {
        "schema_version": 1,
        "purpose": f"stage-1-{retained_experts}e-router-mass-candidate-{purpose_suffix}",
        "promotion_allowed": False,
        "promotion_eligible": promotion_eligible,
        "selection_method": "coverage-constrained-router-mass",
        "source_experts": 128,
        "retained_experts": retained_experts,
        "top_k_experts": top_k_experts,
        "min_coverage": min_coverage,
        "mass_gate_satisfied": all(result.satisfied for result in results),
        "reconstruction_gate_satisfied": False,
        "source_revision": source_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "corpus_gcs_inventory_sha256": corpus_gcs_inventory_sha256,
        "calibration_run_sha256": calibration_run_sha256,
        "calibration_mode": completion["run_mode"],
        "debug_run": debug_run,
        "full_usable_records": int(completion["full_usable_records"]),
        "processed_records": int(completion["processed_records"]),
        "full_corpus_consumed": completion["full_corpus_consumed"] is True,
        "observed_input_tokens": int(completion["observed_input_tokens"]),
        "minimum_observed_input_tokens": int(
            completion["minimum_observed_input_tokens"]
        ),
        "minimum_tokens_observed": completion["minimum_tokens_observed"] is True,
        "sunfish_source": dict(sunfish_source),
        "layers": {
            str(layer): list(result.selected)
            for layer, result in enumerate(results)
        },
        "layer_metrics": {
            str(layer): {
                "coverage": result.coverage,
                "weighted_retained": result.weighted_retained,
                "satisfied": result.satisfied,
            }
            for layer, result in enumerate(results)
        },
    }


class CalibrationSource:
    """Bucket-aware wrapper around the production GCS range-read source."""

    def __init__(
        self,
        directory: str,
        expected_manifest_sha256: str,
        *,
        validated_source_contract: Mapping[str, object],
    ):
        from etils import epath
        from sunfish_tpu.training.data import EPathShardedRecordSource

        root = epath.Path(directory)
        manifest_bytes = (root / "manifest.json").read_bytes()
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != expected_manifest_sha256:
            raise ValueError(
                f"calibration manifest mismatch: expected {expected_manifest_sha256}, got {actual}"
            )
        manifest = json.loads(manifest_bytes)
        if validated_source_contract.get("promotion_allowed") is not True:
            raise ValueError("calibration source contract is not promotable")
        if manifest.get("source_profile") != validated_source_contract.get(
            "source_profile"
        ) or manifest.get("source_receipt_sha256") != validated_source_contract.get(
            "source_receipt_sha256"
        ):
            raise ValueError("calibration manifest differs from validated source contract")
        if manifest.get("corpus_gcs_inventory_sha256") != (
            validated_source_contract.get("corpus_gcs_inventory_sha256")
        ):
            raise ValueError(
                "calibration manifest corpus inventory differs from validated contract"
            )
        failures = manifest.get("failures", ())
        if failures:
            raise ValueError("calibration manifest records failed source buckets")
        self.total_tokens = int(manifest.get("total_tokens", -1))
        self.record_tokens = int(manifest.get("record_tokens", -1))
        if self.total_tokens <= 0 or self.record_tokens < 2:
            raise ValueError("calibration manifest lacks fixed-window token metadata")
        self._source = EPathShardedRecordSource(
            directory,
            expected_manifest_sha256=expected_manifest_sha256,
            # Metadata is checked before JAX; this exact byte check ensures the
            # source opened for use still matches every manifest SHA-256.
            verify_shard_hashes=True,
        )
        self._ends: list[int] = []
        self._workloads: list[str] = []
        total = 0
        for shard in manifest.get("shards", ()):
            workload = shard.get("bucket")
            if workload not in WORKLOADS:
                raise ValueError(f"unknown calibration workload {workload!r}")
            total += int(shard["records"])
            self._ends.append(total)
            self._workloads.append(workload)
        missing = set(WORKLOADS) - set(self._workloads)
        if missing:
            raise ValueError(f"calibration manifest is missing buckets {sorted(missing)}")
        if total != len(self._source):
            raise ValueError("calibration bucket record counts differ from manifest")

    def __len__(self):
        return len(self._source)

    def __getitem__(self, index: int):
        if isinstance(index, bool):
            raise TypeError("calibration source index must be an integer")
        try:
            index = operator.index(index)
        except TypeError:
            raise TypeError("calibration source index must be an integer") from None
        if not 0 <= index < len(self._source):
            raise IndexError("calibration source index out of range")
        shard = bisect.bisect_right(self._ends, index)
        return self._source[index], WORKLOADS.index(self._workloads[shard])


def _existing_flush_boundary(output_dir: Any) -> tuple[int, int]:
    """Return (next sequence, next collective step) from immutable flush names."""
    found = []
    for path in output_dir.glob("router_stats.host0.shard*-step*.json"):
        match = _FLUSH.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), int(match.group(2))))
    if not found:
        return 0, 0
    found.sort()
    for expected_sequence, (sequence, _) in enumerate(found):
        if sequence != expected_sequence:
            raise ValueError("calibration flush sequence has a gap")
    sequence, step = found[-1]
    return sequence + 1, step


def _existing_reconstruction_counts(
    raw_output_dir: Any,
    *,
    process_index: int,
    run_id: str,
    calibration_run_sha256: str,
) -> tuple[dict[str, int], int]:
    """Recover immutable per-bucket quotas for this host after preemption."""
    counts = {
        f"{phase}/{workload}": 0
        for phase in PHASES
        for workload in WORKLOADS
    }
    total = 0
    host = raw_output_dir / f"host-{process_index:05d}"
    for path in host.glob("step-*.json"):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 1:
            raise ValueError(f"unknown reconstruction manifest schema at {path}")
        if int(payload.get("process_index", -1)) != process_index:
            raise ValueError(f"reconstruction manifest host mismatch at {path}")
        if (
            payload.get("run_id") != run_id
            or payload.get("calibration_run_sha256") != calibration_run_sha256
            or payload.get("artifact_id") != path.stem
        ):
            raise ValueError(f"reconstruction manifest lineage mismatch at {path}")
        bucket = payload.get("bucket")
        if bucket not in counts:
            raise ValueError(f"unknown reconstruction bucket at {path}: {bucket!r}")
        tokens = int(payload.get("tokens", -1))
        if tokens <= 0:
            raise ValueError(f"invalid reconstruction token count at {path}")
        counts[bucket] += tokens
        total += tokens
    return counts, total


def _write_immutable(path: Any, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable calibration identity changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def _broadcast_process0_error(
    multihost_utils: Any, np: Any, message: str | None, *, limit: int = 16_384
) -> str | None:
    """Broadcast a caught process-0 finalization failure to every host."""
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


def _first_local_replica(array: Any):
    shards = list(array.addressable_shards)
    if not shards:
        raise RuntimeError("calibration artifact has no addressable shard")
    first_index = str(shards[0].index)
    for shard in shards[1:]:
        if str(shard.index) != first_index:
            raise RuntimeError("artifact is not replicated across the local expert axis")
    return shards[0].data


def run_calibration(
    *,
    source_checkpoint: str,
    source_revision: str,
    source_anonymous: bool,
    readiness_ledger_path: str,
    readiness_ledger_sha256: str,
    data_directory: str,
    data_manifest_sha256: str,
    output_dir: str,
    raw_output_dir: str,
    run_id: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
    prompt_length: int,
    canvas_size: int,
    flush_every_records: int,
    max_records: int,
    reconstruction_tokens: int,
    seed: int,
    min_coverage: float,
    fallback_min_coverage: float,
    min_source_tokens: int,
    data_source_receipt_path: str = "",
    data_source_receipt_sha256: str = "",
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid calibration run ID")
    if not _GCS_REVISION.fullmatch(source_revision):
        raise ValueError("source revision must be a GCS inventory SHA-256")
    if not _SHA256.fullmatch(readiness_ledger_sha256):
        raise ValueError("readiness ledger SHA-256 is invalid")
    require_launcher_run_id(run_id)
    if min(prompt_length, canvas_size, flush_every_records) <= 0:
        raise ValueError("shape/flush values must be positive")
    if (
        reconstruction_tokens < 0
        or min_source_tokens < 0
        or not 0.0 <= min_coverage <= 1.0
        or not 0.0 <= fallback_min_coverage <= 1.0
    ):
        raise ValueError("reconstruction/coverage values are invalid")
    canonical = {
        "minimum calibration input tokens": (
            min_source_tokens,
            MIN_CALIBRATION_INPUT_TOKENS,
        ),
        "32E router-mass coverage": (min_coverage, MASS_COVERAGE_32),
        "48E router-mass coverage": (
            fallback_min_coverage,
            MASS_COVERAGE_48,
        ),
        "reconstruction sample tokens": (
            reconstruction_tokens,
            RECONSTRUCTION_SAMPLE_TOKENS,
        ),
    }
    changed = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in canonical.items()
        if actual != expected
    ]
    if changed:
        raise ValueError(
            "Stage-1 promotion thresholds are canonical and non-overridable: "
            + "; ".join(changed)
        )

    # The Stage-0.5 receipt is storage/control-plane state, not a JAX backend
    # dependency. Refuse an unauthorized Stage-1 launch before touching TPU.
    from etils import epath
    from sunfish_tpu.readiness_ledger import validate_readiness_unlock

    if not data_source_receipt_path:
        raise ValueError("Stage-1 calibration requires a reviewed source receipt")
    data_manifest_bytes = (epath.Path(data_directory) / "manifest.json").read_bytes()
    actual_data_manifest_sha256 = hashlib.sha256(data_manifest_bytes).hexdigest()
    if actual_data_manifest_sha256 != data_manifest_sha256:
        raise ValueError(
            "calibration manifest bytes differ before TPU initialization: "
            f"{actual_data_manifest_sha256} != {data_manifest_sha256}"
        )
    try:
        data_manifest = json.loads(data_manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("calibration manifest is invalid JSON") from error
    if not isinstance(data_manifest, Mapping):
        raise ValueError("calibration manifest must be a JSON object")
    data_source_receipt_bytes = epath.Path(data_source_receipt_path).read_bytes()
    validated_source_contract = validate_calibration_source_contract(
        data_manifest,
        data_source_receipt_bytes,
        expected_receipt_sha256=data_source_receipt_sha256,
    )
    live_corpus_gcs_inventory = verify_live_calibration_data_inventory(
        data_directory,
        data_manifest,
        validated_source_contract["corpus_gcs_inventory"],
    )

    current_source_identity = source_identity_from_environment(required=True)
    readiness_path = epath.Path(readiness_ledger_path)
    readiness_bytes = readiness_path.read_bytes()
    actual_readiness_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    if actual_readiness_sha256 != readiness_ledger_sha256:
        raise ValueError(
            "readiness ledger bytes differ: "
            f"{actual_readiness_sha256} != {readiness_ledger_sha256}"
        )
    validate_readiness_unlock(
        json.loads(readiness_bytes),
        expected_source=current_source_identity,
        expected_devices=expected_devices,
        expected_processes=expected_processes,
        expected_local_devices=expected_local_devices,
    )

    # No backend-adjacent import may move above this call.
    jax, initialization = initialize_distributed_jax(require_distributed=True)
    import flax
    from gemma import gm
    import jax.numpy as jnp
    import numpy as np
    from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
    from jax.experimental import multihost_utils

    from sunfish.expert_selection import select_per_layer
    from sunfish_tpu.calibration import (
        CalibrationState,
        accumulate,
        call_with_router_artifacts,
        call_with_router_probabilities,
        flush_host,
        init_state,
        merge_flushes,
        selection_inputs,
    )
    from sunfish_tpu.orbax_seed import (
        AUDITED_SOURCE_TEXT_PARAMETERS,
        _target_abstract_params,
        _tree_signature,
        require_parameter_count,
    )
    from sunfish_tpu.reconstruction_drain import ReconstructionDrain
    from sunfish_tpu.reconstruction_inventory import build_artifact_inventory
    from sunfish_tpu.gcs_inventory import (
        build_gcs_inventory,
        verify_live_gcs_inventory,
    )
    from sunfish_tpu.teacher_sharding import make_teacher_mesh_and_shardings
    from sunfish_tpu.training.checkpoint import _validate_exact_tree
    from sunfish_tpu.training.model import make_gemma_network
    from sunfish_tpu.training.runtime import verify_runtime_contract

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
        raise RuntimeError(f"calibration topology failed: {json.dumps(topology)}")
    versions = verify_runtime_contract(require_tpu=True)
    source_gcs_inventory = build_gcs_inventory(
        source_checkpoint, anonymous=source_anonymous
    )
    expected_source_revision = (
        f"gcs-inventory-sha256:{source_gcs_inventory['sha256']}"
    )
    if source_revision != expected_source_revision:
        raise ValueError(
            "source revision differs from the live teacher GCS inventory: "
            f"expected {expected_source_revision}"
        )
    shardings = make_teacher_mesh_and_shardings(jax, np)
    if shardings["data_axis_size"] != int(jax.process_count()):
        raise RuntimeError("teacher calibration requires exactly one data replica per process")

    root = epath.Path(output_dir) / run_id
    raw_root = epath.Path(raw_output_dir) / run_id
    source = CalibrationSource(
        data_directory,
        data_manifest_sha256,
        validated_source_contract=validated_source_contract,
    )
    if source.record_tokens > prompt_length + canvas_size:
        raise ValueError(
            "calibration records exceed prompt+canvas capacity; rebuild fixed windows"
        )
    if source.total_tokens < min_source_tokens:
        raise ValueError(
            f"calibration manifest has {source.total_tokens:,} tokens; "
            f"at least {min_source_tokens:,} are required"
        )
    usable = usable_record_count(
        len(source), process_count=expected_processes, max_records=max_records
    )
    full_usable = usable_record_count(
        len(source), process_count=expected_processes
    )
    steps = usable // expected_processes
    debug_run = max_records > 0
    maximum_usable_input_tokens = min(
        source.total_tokens, full_usable * source.record_tokens
    )
    if not debug_run and maximum_usable_input_tokens < min_source_tokens:
        raise ValueError(
            "full usable calibration prefix can contain at most "
            f"{maximum_usable_input_tokens:,} input tokens; "
            f"at least {min_source_tokens:,} are required"
        )
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "source_checkpoint": source_checkpoint,
        "source_revision": source_revision,
        "source_gcs_inventory": source_gcs_inventory,
        "readiness_ledger": readiness_ledger_path,
        "readiness_ledger_sha256": readiness_ledger_sha256,
        "dataset_directory": data_directory,
        "dataset_manifest_sha256": data_manifest_sha256,
        "dataset_source_receipt": data_source_receipt_path,
        "dataset_source_receipt_sha256": data_source_receipt_sha256,
        "dataset_source_profile": validated_source_contract["source_profile"],
        "corpus_gcs_inventory": live_corpus_gcs_inventory,
        "corpus_gcs_inventory_sha256": live_corpus_gcs_inventory["sha256"],
        "run_mode": "debug-capped" if debug_run else "full",
        "debug_run": debug_run,
        "max_records": max_records,
        "source_records": len(source),
        "full_usable_records": full_usable,
        "usable_records": usable,
        "source_tokens": source.total_tokens,
        "maximum_usable_input_tokens": maximum_usable_input_tokens,
        "minimum_source_tokens": min_source_tokens,
        "reconstruction_sample_tokens": reconstruction_tokens,
        "record_tokens": source.record_tokens,
        "collective_steps": steps,
        "discarded_tail_records": len(source) - usable,
        "process_divisibility_tail_records": len(source) - full_usable,
        "capped_records": full_usable - usable,
        "prompt_length": prompt_length,
        "canvas_size": canvas_size,
        "flush_every_records": flush_every_records,
        "noise_rates": dict(zip(PHASES[1:], NOISE_RATES, strict=True)),
        "position_buckets": True,
        "coverage_floors": {
            "32_experts": min_coverage,
            "48_experts": fallback_min_coverage,
        },
        "topology": {
            "devices": expected_devices,
            "processes": expected_processes,
            "local_devices": expected_local_devices,
            "teacher_mesh": [
                shardings["data_axis_size"],
                shardings["expert_axis_size"],
            ],
        },
        "runtime_versions": versions,
        "sunfish_source": current_source_identity,
    }
    if int(jax.process_index()) == 0:
        _write_immutable(root / "calibration-run.json", identity)
    multihost_utils.sync_global_devices(f"sunfish-calibration-identity-{run_id}")
    if json.loads((root / "calibration-run.json").read_text()) != identity:
        raise RuntimeError("calibration identity readback differs")
    calibration_run_sha256 = hashlib.sha256(
        (root / "calibration-run.json").read_bytes()
    ).hexdigest()

    network = make_gemma_network(
        num_experts=128,
        top_k_experts=8,
        dtype="bfloat16",
        use_lora=False,
        lora_rank=1,
    )
    target_abstract = _target_abstract_params(
        num_experts=128,
        top_k_experts=8,
        jax=jax,
        jnp=jnp,
    )
    target_shardings = shardings["params"](target_abstract)
    sharded_target = jax.tree.map(
        lambda value, sharding: jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=sharding
        ),
        target_abstract,
        target_shardings,
    )
    params = gm.ckpts.load_params(
        epath.Path(source_checkpoint),
        params=sharded_target,
        donate=False,
        text_only=True,
    )
    # Span the actual multi-host checkpoint read with generation/CRC checks.
    # A same-size replacement during load changes either generation or CRC and
    # invalidates this attempt before any calibration evidence is emitted.
    verify_live_gcs_inventory(
        source_checkpoint, source_gcs_inventory, anonymous=source_anonymous
    )
    _validate_exact_tree(sharded_target, params)
    source_signature = _tree_signature(params, flax)
    require_parameter_count(
        source_signature,
        expected=AUDITED_SOURCE_TEXT_PARAMETERS,
        label="sharded calibration teacher",
    )

    cache_length = prompt_length + canvas_size
    @functools.partial(jax.jit, static_argnames=("capture_artifacts",))
    def prefill(current_params, prompt, prompt_mask, *, capture_artifacts):
        cache = network.apply(
            {"params": current_params},
            batch_size=shardings["data_axis_size"],
            cache_length=cache_length,
            method=network.init_cache,
        )
        positions = mask_helpers.build_positions_from_mask(prompt_mask)
        attention_mask = mask_helpers.make_causal_prefill_mask(
            prompt_mask, cache_length
        )

        def encoder_forward():
            output = network.apply(
                {"params": current_params},
                x=prompt,
                conditioning_embeddings={
                    "kv_cache": cache,
                    "positions": positions,
                    "attention_mask": attention_mask,
                },
                method=network.encoder_call,
            )
            return output.cache

        if capture_artifacts:
            output, artifacts = call_with_router_artifacts(
                encoder_forward, expected_layers=30
            )
            return output, artifacts
        output, probabilities = call_with_router_probabilities(
            encoder_forward, expected_layers=30
        )
        return output, {"probabilities": probabilities}

    def decoder_attention(prompt_mask, canvas_mask):
        return mask_helpers.create_decoder_attention_mask(
            prompt_mask=prompt_mask,
            canvas_mask=canvas_mask,
            selected_canvas_idx=jnp.zeros((canvas_mask.shape[0],), jnp.int32),
            prompt_len=prompt_length,
            total_canvas_len=canvas_size,
            canvas_size=canvas_size,
            num_queries=canvas_size,
        )

    def decoder_positions(prompt_mask):
        start = jnp.sum(prompt_mask, axis=-1, dtype=jnp.int32)
        return start[:, None] + jnp.arange(canvas_size, dtype=jnp.int32)[None, :]

    def corrupt(canvas, canvas_mask, key, rate):
        replace_key, token_key = jax.random.split(key)
        replace = jax.random.uniform(replace_key, canvas.shape) < rate
        replace &= canvas_mask
        random_tokens = jax.random.randint(
            token_key, canvas.shape, 0, 262_144, dtype=jnp.int32
        )
        return jnp.where(replace, random_tokens, canvas)[..., None]

    def decode_apply(current_params, xt, time_value, cache, prompt_mask, canvas_mask, sc_logits=None):
        conditioning = {
            "kv_cache": cache,
            "positions": decoder_positions(prompt_mask),
            "attention_mask": decoder_attention(prompt_mask, canvas_mask),
        }
        if sc_logits is not None:
            conditioning["sc_logits"] = sc_logits
        return network.apply(
            {"params": current_params},
            xt=xt,
            time=time_value,
            conditioning=conditioning,
            is_training=False,
        )

    @functools.partial(jax.jit, static_argnames=("capture_artifacts",))
    def denoise(
        current_params,
        cache,
        prompt_mask,
        canvas,
        canvas_mask,
        key,
        rate,
        *,
        capture_artifacts,
    ):
        xt = corrupt(canvas, canvas_mask, key, rate)
        time_value = jnp.full(
            (canvas.shape[0], 1, 1), rate, dtype=jnp.float32
        )
        first = decode_apply(
            current_params, xt, time_value, cache, prompt_mask, canvas_mask
        )
        sc_logits = jax.lax.stop_gradient(first["logits"])
        forward = lambda: decode_apply(
            current_params,
            xt,
            time_value,
            cache,
            prompt_mask,
            canvas_mask,
            sc_logits,
        )
        if capture_artifacts:
            _, artifacts = call_with_router_artifacts(
                forward, expected_layers=30
            )
            return artifacts
        _, probabilities = call_with_router_probabilities(
            forward, expected_layers=30
        )
        return {"probabilities": probabilities}

    bucket_names = calibration_bucket_names()
    state = init_state(len(bucket_names), 30, 128)
    state = CalibrationState(
        mass=jax.device_put(state.mass, shardings["replicated"]),
        tokens=jax.device_put(state.tokens, shardings["replicated"]),
    )

    @jax.jit
    def fold(current_state, probabilities, workloads, token_mask, phase_index):
        length = token_mask.shape[1]
        workload_per_token = jnp.repeat(workloads, length)
        prefill_ids = workload_per_token
        position = jnp.minimum(
            jnp.arange(length, dtype=jnp.int32) * 3 // length,
            2,
        )
        position = jnp.tile(position, workloads.shape[0])
        denoise_ids = (
            len(WORKLOADS)
            + (phase_index - 1) * len(WORKLOADS) * 3
            + workload_per_token * 3
            + position
        )
        bucket_ids = jnp.where(phase_index == 0, prefill_ids, denoise_ids)
        bucket_ids = jnp.where(token_mask.reshape(-1), bucket_ids, -1)
        updated = accumulate(current_state, probabilities, bucket_ids)
        return CalibrationState(
            mass=jax.lax.with_sharding_constraint(
                updated.mass, shardings["replicated"]
            ),
            tokens=jax.lax.with_sharding_constraint(
                updated.tokens, shardings["replicated"]
            ),
        )

    sequence, start_step = _existing_flush_boundary(root)
    if start_step > steps:
        raise ValueError("existing calibration flush is beyond this dataset")
    per_bucket_reconstruction_cap = math.ceil(
        reconstruction_tokens / (24 * expected_processes)
    ) if reconstruction_tokens else 0
    per_host_reconstruction_cap = math.ceil(
        reconstruction_tokens / expected_processes
    ) if reconstruction_tokens else 0
    reconstruction_counts, existing_reconstruction_tokens = (
        _existing_reconstruction_counts(
            raw_root,
            process_index=int(jax.process_index()),
            run_id=run_id,
            calibration_run_sha256=calibration_run_sha256,
        )
    )
    if any(
        count > per_bucket_reconstruction_cap
        for count in reconstruction_counts.values()
    ):
        raise ValueError("existing reconstruction artifacts exceed a bucket quota")
    if existing_reconstruction_tokens > per_host_reconstruction_cap:
        raise ValueError("existing reconstruction artifacts exceed the host quota")
    drain = ReconstructionDrain(
        output_dir=str(raw_root),
        process_index=int(jax.process_index()),
        run_id=run_id,
        calibration_run_sha256=calibration_run_sha256,
        max_tokens=per_host_reconstruction_cap,
        initial_tokens=existing_reconstruction_tokens,
    )
    since_flush = 0
    try:
        with shardings["mesh"]:
            for step in range(start_step, steps):
                record_index = int(jax.process_index()) + step * expected_processes
                words, workload = source[record_index]
                packed = pack_calibration_tokens(
                    words,
                    prompt_length=prompt_length,
                    canvas_size=canvas_size,
                    pad_token=0,
                    vocab_size=262_144,
                )

                def global_array(value, dtype, sharding):
                    local = np.asarray([value], dtype=dtype)
                    global_shape = (shardings["data_axis_size"],) + local.shape[1:]
                    return jax.make_array_from_process_local_data(
                        sharding, local, global_shape=global_shape
                    )

                prompt = global_array(packed["prompt"], np.int32, shardings["batch"])
                prompt_mask = global_array(
                    packed["prompt_mask"], np.bool_, shardings["batch"]
                )
                canvas = global_array(packed["canvas"], np.int32, shardings["batch"])
                canvas_mask = global_array(
                    packed["canvas_mask"], np.bool_, shardings["batch"]
                )
                workloads = global_array(
                    workload, np.int32, shardings["batch_vector"]
                )

                prefill_bucket = f"prefill/{WORKLOADS[workload]}"
                prefill_artifact_id = f"step-{step:09d}-prefill"
                prefill_artifact_manifest = (
                    raw_root
                    / f"host-{int(jax.process_index()):05d}"
                    / f"{prefill_artifact_id}.json"
                )
                sample_prefill_artifact = (
                    reconstruction_counts[prefill_bucket]
                    < per_bucket_reconstruction_cap
                    and drain.tokens_submitted < drain.max_tokens
                    and not prefill_artifact_manifest.exists()
                )
                cache, prefill_artifacts = prefill(
                    params,
                    prompt,
                    prompt_mask,
                    capture_artifacts=sample_prefill_artifact,
                )
                state = fold(
                    state,
                    prefill_artifacts["probabilities"],
                    workloads,
                    prompt_mask,
                    jnp.asarray(0, jnp.int32),
                )
                if sample_prefill_artifact:
                    valid = min(
                        int(packed["prompt_tokens"]),
                        per_bucket_reconstruction_cap
                        - reconstruction_counts[prefill_bucket],
                    )
                    local_artifacts = {
                        name: _first_local_replica(prefill_artifacts[name])
                        for name in (
                            "shared_pre_router_residual",
                            "topk_indices",
                            "final_scaled_topk_weights",
                        )
                    }
                    before_tokens = drain.tokens_submitted
                    if drain.submit(
                        local_artifacts,
                        bucket=prefill_bucket,
                        artifact_id=prefill_artifact_id,
                        valid_tokens=valid,
                    ):
                        reconstruction_counts[prefill_bucket] += (
                            drain.tokens_submitted - before_tokens
                        )
                for phase_offset, (phase, rate) in enumerate(
                    zip(PHASES[1:], NOISE_RATES, strict=True), start=1
                ):
                    bucket = f"{phase}/{WORKLOADS[workload]}"
                    artifact_id = f"step-{step:09d}-{phase}"
                    artifact_manifest = (
                        raw_root
                        / f"host-{int(jax.process_index()):05d}"
                        / f"{artifact_id}.json"
                    )
                    sample_artifact = (
                        reconstruction_counts[bucket] < per_bucket_reconstruction_cap
                        and drain.tokens_submitted < drain.max_tokens
                        and not artifact_manifest.exists()
                    )
                    key = jax.random.fold_in(
                        jax.random.key(seed + phase_offset), step
                    )
                    artifacts = denoise(
                        params,
                        cache,
                        prompt_mask,
                        canvas,
                        canvas_mask,
                        key,
                        jnp.asarray(rate, jnp.float32),
                        capture_artifacts=sample_artifact,
                    )
                    state = fold(
                        state,
                        artifacts["probabilities"],
                        workloads,
                        canvas_mask,
                        jnp.asarray(phase_offset, jnp.int32),
                    )
                    if sample_artifact:
                        valid = min(
                            int(packed["canvas_tokens"]),
                            per_bucket_reconstruction_cap
                            - reconstruction_counts[bucket],
                        )
                        local_artifacts = {
                            name: _first_local_replica(artifacts[name])
                            for name in (
                                "shared_pre_router_residual",
                                "topk_indices",
                                "final_scaled_topk_weights",
                            )
                        }
                        before_tokens = drain.tokens_submitted
                        submitted = drain.submit(
                            local_artifacts,
                            bucket=bucket,
                            artifact_id=artifact_id,
                            valid_tokens=valid,
                        )
                        if submitted:
                            reconstruction_counts[bucket] += (
                                drain.tokens_submitted - before_tokens
                            )
                since_flush += 1
                boundary = step + 1
                if since_flush == flush_every_records or boundary == steps:
                    jax.block_until_ready(state)
                    # Raw artifacts from this interval must be durable before
                    # its stats boundary becomes resumably complete.
                    drain.flush()
                    multihost_utils.sync_global_devices(
                        f"sunfish-calibration-artifacts-boundary-{run_id}-{sequence}"
                    )
                    if int(jax.process_index()) == 0:
                        flush_host(
                            state,
                            bucket_names=bucket_names,
                            process_index=0,
                            shard_id=f"{sequence:05d}-step{boundary:09d}",
                            output_dir=root,
                            check=True,
                        )
                    multihost_utils.sync_global_devices(
                        f"sunfish-calibration-flush-{run_id}-{sequence}"
                    )
                    sequence += 1
                    since_flush = 0
                    fresh = init_state(len(bucket_names), 30, 128)
                    state = CalibrationState(
                        mass=jax.device_put(fresh.mass, shardings["replicated"]),
                        tokens=jax.device_put(fresh.tokens, shardings["replicated"]),
                    )
    finally:
        artifact_manifests = drain.close()

    multihost_utils.sync_global_devices(f"sunfish-calibration-artifacts-{run_id}")
    corpus_inventory_error = None
    if int(jax.process_index()) == 0:
        try:
            verify_live_calibration_data_inventory(
                data_directory,
                data_manifest,
                live_corpus_gcs_inventory,
            )
        except Exception as error:
            corpus_inventory_error = f"{type(error).__name__}: {error}"
    corpus_inventory_error = _broadcast_process0_error(
        multihost_utils, np, corpus_inventory_error
    )
    if corpus_inventory_error is not None:
        raise RuntimeError(
            "calibration corpus changed while in use: " + corpus_inventory_error
        )
    reconstruction_buckets = [
        f"{phase}/{workload}" for phase in PHASES for workload in WORKLOADS
    ]
    gathered_reconstruction_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                [reconstruction_counts[bucket] for bucket in reconstruction_buckets],
                np.int32,
            )
        )
    ).reshape((expected_processes, len(reconstruction_buckets)))
    global_reconstruction_counts = gathered_reconstruction_counts.sum(axis=0)
    per_bucket_artifact_floor = (
        reconstruction_tokens // len(reconstruction_buckets)
        if reconstruction_tokens
        else 0
    )
    artifact_sample_satisfied = (
        int(global_reconstruction_counts.sum()) >= reconstruction_tokens
        and all(
            int(value) >= per_bucket_artifact_floor
            for value in global_reconstruction_counts
        )
    )

    def finalize_calibration():
        merged = merge_flushes(root, require_phase_coverage=True)
        # Prefill and denoise-high each cover one copy of the source record's
        # disjoint prompt/canvas split. Their sum is therefore the number of
        # input tokens actually observed, rather than the manifest's claim.
        observed_input_tokens = sum(
            merged.tokens(bucket)
            for bucket in merged.buckets
            if bucket.startswith("prefill/")
            or bucket.startswith("denoise_high/")
        )
        completion = calibration_completion_status(
            total_records=len(source),
            process_count=expected_processes,
            processed_records=usable,
            max_records=max_records,
            observed_input_tokens=observed_input_tokens,
            minimum_input_tokens=min_source_tokens,
        )
        bucket_weights = {
            bucket: (0.25 if bucket.split("/")[1].endswith("control") else 1.0)
            for bucket in merged.buckets
        }
        results_32 = select_per_layer(
            [selection_inputs(merged, layer) for layer in range(30)],
            k=32,
            bucket_weights=bucket_weights,
            min_coverage=min_coverage,
        )
        results_48 = select_per_layer(
            [selection_inputs(merged, layer) for layer in range(30)],
            k=48,
            bucket_weights=bucket_weights,
            min_coverage=fallback_min_coverage,
        )
        candidate = mass_candidate_payload(
            results_32,
            min_coverage=min_coverage,
            source_revision=source_revision,
            dataset_manifest_sha256=data_manifest_sha256,
            corpus_gcs_inventory_sha256=live_corpus_gcs_inventory["sha256"],
            calibration_run_sha256=calibration_run_sha256,
            completion=completion,
            sunfish_source=current_source_identity,
        )
        fallback_candidate = mass_candidate_payload(
            results_48,
            min_coverage=fallback_min_coverage,
            source_revision=source_revision,
            dataset_manifest_sha256=data_manifest_sha256,
            corpus_gcs_inventory_sha256=live_corpus_gcs_inventory["sha256"],
            calibration_run_sha256=calibration_run_sha256,
            completion=completion,
            retained_experts=48,
            sunfish_source=current_source_identity,
        )
        if completion["promotion_eligible"]:
            candidate_path = root / "selection-mass-candidate.json"
            fallback_candidate_path = root / "selection-mass-candidate-48e.json"
        else:
            candidate_path = root / "selection-mass-debug-candidate.json"
            fallback_candidate_path = (
                root / "selection-mass-debug-candidate-48e.json"
            )
        _write_immutable(candidate_path, candidate)
        _write_immutable(fallback_candidate_path, fallback_candidate)
        candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        fallback_candidate_sha256 = hashlib.sha256(
            fallback_candidate_path.read_bytes()
        ).hexdigest()
        candidates_promotion_eligible = bool(
            candidate["promotion_eligible"]
            and fallback_candidate["promotion_eligible"]
        )
        if completion["promotion_eligible"] != candidates_promotion_eligible:
            raise RuntimeError(
                "mass candidate promotion status differs from calibration completion"
            )
        artifact_inventory = build_artifact_inventory(
            raw_root,
            run_id=run_id,
            calibration_run_sha256=calibration_run_sha256,
            expected_processes=expected_processes,
            allowed_buckets=reconstruction_buckets,
            field_names=RECONSTRUCTION_FIELDS,
        )
        inventory_counts = np.asarray(
            [
                artifact_inventory["tokens_by_bucket"][bucket]
                for bucket in reconstruction_buckets
            ],
            np.int64,
        )
        if not np.array_equal(inventory_counts, global_reconstruction_counts):
            raise RuntimeError(
                "reconstruction artifact inventory differs from gathered host counts"
            )
        artifact_inventory_path = root / "reconstruction-artifact-inventory.json"
        _write_immutable(artifact_inventory_path, artifact_inventory)
        artifact_inventory_sha256 = hashlib.sha256(
            artifact_inventory_path.read_bytes()
        ).hexdigest()
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "execution_completed": True,
            "run_succeeded": artifact_sample_satisfied,
            "passed": False,
            "mass_gate_satisfied": candidate["mass_gate_satisfied"],
            "fallback_48_mass_gate_satisfied": fallback_candidate[
                "mass_gate_satisfied"
            ],
            "reconstruction_gate_satisfied": False,
            "promotion_allowed": False,
            "promotion_eligible": completion["promotion_eligible"],
            "run_mode": completion["run_mode"],
            "debug_run": completion["debug_run"],
            "max_records": completion["max_records"],
            "source_records": completion["source_records"],
            "full_usable_records": completion["full_usable_records"],
            "processed_records": completion["processed_records"],
            "full_corpus_consumed": completion["full_corpus_consumed"],
            "minimum_tokens_observed": completion["minimum_tokens_observed"],
            "maximum_usable_input_tokens": maximum_usable_input_tokens,
            "collective_steps": steps,
            "observed_input_tokens": observed_input_tokens,
            "minimum_observed_input_tokens": min_source_tokens,
            "dataset_manifest_sha256": data_manifest_sha256,
            "corpus_gcs_inventory_sha256": live_corpus_gcs_inventory["sha256"],
            "calibration_run_sha256": calibration_run_sha256,
            "discarded_tail_records": len(source) - usable,
            "process_divisibility_tail_records": len(source) - full_usable,
            "capped_records": full_usable - usable,
            "flushes": sequence,
            "source_tree": source_signature,
            "candidate": str(candidate_path),
            "candidate_sha256": candidate_sha256,
            "fallback_candidate": str(fallback_candidate_path),
            "fallback_candidate_sha256": fallback_candidate_sha256,
            "raw_artifact_prefix": str(raw_root),
            "artifact_inventory": str(artifact_inventory_path),
            "artifact_inventory_sha256": artifact_inventory_sha256,
            "artifact_sample_satisfied": artifact_sample_satisfied,
            "artifact_tokens": int(global_reconstruction_counts.sum()),
            "artifact_tokens_by_bucket": {
                bucket: int(global_reconstruction_counts[index])
                for index, bucket in enumerate(reconstruction_buckets)
            },
            "artifact_tokens_per_bucket_floor": per_bucket_artifact_floor,
            "hardware_topology": topology,
        }
        _write_immutable(root / "summary.json", summary)
        return summary

    summary = None
    finalization_error = None
    if int(jax.process_index()) == 0:
        try:
            summary = finalize_calibration()
        except Exception as error:  # propagate instead of stranding peer collectives
            finalization_error = f"{type(error).__name__}: {error}"
    finalization_error = _broadcast_process0_error(
        multihost_utils, np, finalization_error
    )
    if finalization_error is not None:
        raise RuntimeError(
            "calibration process-0 finalization failed: " + finalization_error
        )
    multihost_utils.sync_global_devices(f"sunfish-calibration-summary-{run_id}")
    return summary if summary is not None else {
        "schema_version": 1,
        "run_id": run_id,
        "process_index": int(jax.process_index()),
        "artifact_manifests": artifact_manifests,
        "execution_completed": True,
        "run_succeeded": artifact_sample_satisfied,
        "artifact_sample_satisfied": artifact_sample_satisfied,
        "passed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-anonymous", action="store_true")
    parser.add_argument("--readiness-ledger", required=True)
    parser.add_argument("--readiness-ledger-sha256", required=True)
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--data-source-receipt", required=True)
    parser.add_argument("--data-source-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--flush-every-records", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--reconstruction-tokens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--min-coverage", type=float, default=0.225)
    parser.add_argument("--fallback-min-coverage", type=float, default=0.3375)
    parser.add_argument("--min-source-tokens", type=int, default=75_000_000)
    args = parser.parse_args(argv)
    try:
        payload = run_calibration(
            source_checkpoint=args.source_checkpoint,
            source_revision=args.source_revision,
            source_anonymous=args.source_anonymous,
            readiness_ledger_path=args.readiness_ledger,
            readiness_ledger_sha256=args.readiness_ledger_sha256,
            data_directory=args.data_directory,
            data_manifest_sha256=args.data_manifest_sha256,
            data_source_receipt_path=args.data_source_receipt,
            data_source_receipt_sha256=args.data_source_receipt_sha256,
            output_dir=args.output_dir,
            raw_output_dir=args.raw_output_dir,
            run_id=args.run_id,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
            prompt_length=args.prompt_length,
            canvas_size=args.canvas_size,
            flush_every_records=args.flush_every_records,
            max_records=args.max_records,
            reconstruction_tokens=args.reconstruction_tokens,
            seed=args.seed,
            min_coverage=args.min_coverage,
            fallback_min_coverage=args.fallback_min_coverage,
            min_source_tokens=args.min_source_tokens,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"sunfish-calibrate: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["run_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
