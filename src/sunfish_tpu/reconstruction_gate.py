"""Distributed Stage-1 layer-output reconstruction gate.

This job reloads the sharded 128-expert teacher, derives the mass-selected
32-expert candidate in HBM, and replays the bounded calibration residuals
through the exact pinned Gemma 4 MoE FFW. It validates the recorded teacher
routes before comparing the complete shared-dense-plus-routed FFW output.
Only a predeclared threshold pair can produce a promotable selection manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from sunfish_tpu.calibration import PHASES, WORKLOADS
from sunfish_tpu.calibration_data_inventory import (
    validate_calibration_data_inventory_payload,
)
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
from sunfish_tpu.reconstruction_inventory import (
    paths_for_process,
    validate_artifact_inventory,
    verify_live_artifact_inventory,
)
from sunfish_tpu.stage1_contract import (
    APPROVED_SELECTION_METHOD,
    APPROVED_SELECTION_PURPOSE,
    MASS_COVERAGE_BY_EXPERTS,
    MIN_CALIBRATION_INPUT_TOKENS,
    RECONSTRUCTION_MAX_RELATIVE_RMSE,
    RECONSTRUCTION_MIN_COSINE_SIMILARITY,
    RECONSTRUCTION_MIN_TOKENS_PER_BUCKET,
    RECONSTRUCTION_SAMPLE_TOKENS,
    reconstruction_thresholds,
)

NUM_LAYERS = 30
SOURCE_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2816
RECONSTRUCTION_BUCKETS = tuple(
    f"{phase}/{workload}" for phase in PHASES for workload in WORKLOADS
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = (
    "shared_pre_router_residual",
    "topk_indices",
    "final_scaled_topk_weights",
)


def summarize_reconstruction(
    metric_sums: Sequence[Sequence[Sequence[float]]],
    tokens: Sequence[int],
    *,
    route_mismatches: int,
    max_relative_rmse: float,
    min_cosine_similarity: float,
    min_total_tokens: int,
    min_tokens_per_bucket: int,
) -> dict[str, Any]:
    """Turn additive sufficient statistics into a fail-closed gate report."""
    if not 0.0 <= max_relative_rmse or not -1.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("invalid reconstruction thresholds")
    if min_total_tokens < 0 or min_tokens_per_bucket < 0 or route_mismatches < 0:
        raise ValueError("invalid reconstruction counts")
    if len(metric_sums) != len(RECONSTRUCTION_BUCKETS) or len(tokens) != len(
        RECONSTRUCTION_BUCKETS
    ):
        raise ValueError("reconstruction statistics have the wrong bucket count")

    errors: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    total_error = total_teacher = total_dot = total_candidate = 0.0
    worst_relative = (-1.0, "")
    worst_cosine = (2.0, "")
    for bucket_index, bucket in enumerate(RECONSTRUCTION_BUCKETS):
        count = int(tokens[bucket_index])
        if count < min_tokens_per_bucket:
            errors.append(
                f"{bucket}: {count} tokens below minimum {min_tokens_per_bucket}"
            )
        layers = metric_sums[bucket_index]
        if len(layers) != NUM_LAYERS:
            raise ValueError(f"{bucket} has {len(layers)} layers, expected {NUM_LAYERS}")
        bucket_rows: dict[str, Any] = {}
        for layer, values in enumerate(layers):
            if len(values) != 4:
                raise ValueError("metric rows must contain four additive sums")
            error_sq, teacher_sq, dot, candidate_sq = map(float, values)
            if not all(
                math.isfinite(value)
                for value in (error_sq, teacher_sq, dot, candidate_sq)
            ):
                errors.append(f"{bucket}/layer{layer}: non-finite metric")
                relative = math.inf
                cosine = -1.0
            elif teacher_sq <= 0.0 or candidate_sq <= 0.0:
                errors.append(f"{bucket}/layer{layer}: zero output norm")
                relative = math.inf
                cosine = -1.0
            else:
                relative = math.sqrt(max(0.0, error_sq) / teacher_sq)
                cosine = dot / math.sqrt(teacher_sq * candidate_sq)
                cosine = max(-1.0, min(1.0, cosine))
                if relative > max_relative_rmse:
                    errors.append(
                        f"{bucket}/layer{layer}: relative RMSE {relative:.6g} "
                        f"exceeds {max_relative_rmse:.6g}"
                    )
                if cosine < min_cosine_similarity:
                    errors.append(
                        f"{bucket}/layer{layer}: cosine {cosine:.6g} below "
                        f"{min_cosine_similarity:.6g}"
                    )
            label = f"{bucket}/layer{layer}"
            if relative > worst_relative[0]:
                worst_relative = (relative, label)
            if cosine < worst_cosine[0]:
                worst_cosine = (cosine, label)
            bucket_rows[str(layer)] = {
                "relative_rmse": relative,
                "cosine_similarity": cosine,
                "error_squared_sum": error_sq,
                "teacher_squared_sum": teacher_sq,
                "candidate_squared_sum": candidate_sq,
            }
            total_error += error_sq
            total_teacher += teacher_sq
            total_dot += dot
            total_candidate += candidate_sq
        rows[bucket] = {"tokens": count, "layers": bucket_rows}

    total_tokens = sum(int(value) for value in tokens)
    if total_tokens < min_total_tokens:
        errors.append(
            f"total reconstruction tokens {total_tokens} below {min_total_tokens}"
        )
    if route_mismatches:
        errors.append(f"recorded teacher route values differ at {route_mismatches} entries")
    overall_relative = (
        math.sqrt(max(0.0, total_error) / total_teacher)
        if total_teacher > 0.0
        else math.inf
    )
    overall_cosine = (
        total_dot / math.sqrt(total_teacher * total_candidate)
        if total_teacher > 0.0 and total_candidate > 0.0
        else -1.0
    )
    return {
        "passed": not errors,
        "errors": errors,
        "thresholds": {
            "max_relative_rmse": max_relative_rmse,
            "min_cosine_similarity": min_cosine_similarity,
            "min_total_tokens": min_total_tokens,
            "min_tokens_per_bucket": min_tokens_per_bucket,
        },
        "total_tokens": total_tokens,
        "route_mismatches": route_mismatches,
        "overall_relative_rmse": overall_relative,
        "overall_cosine_similarity": overall_cosine,
        "worst_relative_rmse": {
            "value": worst_relative[0],
            "bucket_layer": worst_relative[1],
        },
        "worst_cosine_similarity": {
            "value": worst_cosine[0],
            "bucket_layer": worst_cosine[1],
        },
        "buckets": rows,
    }


def validate_calibration_for_reconstruction(
    calibration_identity: Mapping[str, Any],
    calibration_summary: Mapping[str, Any],
    mass_candidate: Mapping[str, Any],
    *,
    calibration_run_sha256: str,
    calibration_summary_sha256: str,
    mass_candidate_sha256: str,
    mass_candidate_path: str,
) -> dict[str, Any]:
    """Validate the full-corpus receipt required before reconstruction."""
    for digest in (
        calibration_run_sha256,
        calibration_summary_sha256,
        mass_candidate_sha256,
    ):
        if not _SHA256.fullmatch(digest):
            raise ValueError("calibration promotion provenance requires SHA-256 digests")
    if calibration_identity.get("schema_version") != 1:
        raise ValueError("calibration identity has an unsupported schema")
    if calibration_summary.get("schema_version") != 1:
        raise ValueError("calibration summary has an unsupported schema")
    run_id = calibration_identity.get("run_id")
    if not isinstance(run_id, str) or calibration_summary.get("run_id") != run_id:
        raise ValueError("calibration identity and summary run IDs differ")
    if (
        calibration_identity.get("run_mode") != "full"
        or calibration_identity.get("debug_run") is not False
        or calibration_identity.get("max_records") != 0
    ):
        raise ValueError("debug/capped calibration cannot authorize reconstruction")
    corpus_inventory = calibration_identity.get("corpus_gcs_inventory")
    dataset_directory = calibration_identity.get("dataset_directory")
    if not isinstance(corpus_inventory, Mapping) or not isinstance(
        dataset_directory, str
    ):
        raise ValueError("calibration identity has no corpus GCS inventory")
    canonical_corpus_inventory = validate_calibration_data_inventory_payload(
        corpus_inventory, expected_directory=dataset_directory
    )
    corpus_inventory_sha256 = calibration_identity.get(
        "corpus_gcs_inventory_sha256"
    )
    if (
        not isinstance(corpus_inventory_sha256, str)
        or not _SHA256.fullmatch(corpus_inventory_sha256)
        or corpus_inventory_sha256 != canonical_corpus_inventory["sha256"]
    ):
        raise ValueError("calibration identity corpus GCS inventory digest differs")

    try:
        source_records = int(calibration_identity["source_records"])
        full_usable_records = int(calibration_identity["full_usable_records"])
        usable_records = int(calibration_identity["usable_records"])
        collective_steps = int(calibration_identity["collective_steps"])
        source_tokens = int(calibration_identity["source_tokens"])
        record_tokens = int(calibration_identity["record_tokens"])
        maximum_usable_input_tokens = int(
            calibration_identity["maximum_usable_input_tokens"]
        )
        minimum_input_tokens = int(calibration_identity["minimum_source_tokens"])
        observed_input_tokens = int(calibration_summary["observed_input_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("calibration completion counts are missing or invalid") from error
    if not 0 < full_usable_records == usable_records <= source_records:
        raise ValueError("calibration identity does not cover its full usable corpus")
    topology = calibration_identity.get("topology")
    try:
        process_count = int(topology["processes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("calibration identity has invalid process topology") from error
    if (
        collective_steps <= 0
        or process_count <= 0
        or collective_steps * process_count != full_usable_records
        or not 0 <= source_records - full_usable_records < process_count
        or source_tokens <= 0
        or record_tokens < 2
        or maximum_usable_input_tokens
        != min(source_tokens, full_usable_records * record_tokens)
        or minimum_input_tokens != MIN_CALIBRATION_INPUT_TOKENS
        or calibration_identity.get("reconstruction_sample_tokens")
        != RECONSTRUCTION_SAMPLE_TOKENS
        or calibration_identity.get("coverage_floors")
        != {
            "32_experts": MASS_COVERAGE_BY_EXPERTS[32],
            "48_experts": MASS_COVERAGE_BY_EXPERTS[48],
        }
    ):
        raise ValueError("calibration identity has invalid step/token thresholds")

    required_true = (
        "execution_completed",
        "run_succeeded",
        "artifact_sample_satisfied",
        "promotion_eligible",
        "full_corpus_consumed",
        "minimum_tokens_observed",
    )
    if any(calibration_summary.get(key) is not True for key in required_true):
        raise ValueError("calibration summary is incomplete or non-promotable")
    if (
        calibration_summary.get("run_mode") != "full"
        or calibration_summary.get("debug_run") is not False
        or calibration_summary.get("max_records") != 0
    ):
        raise ValueError("calibration summary describes a debug/capped run")
    if calibration_summary.get("calibration_run_sha256") != calibration_run_sha256:
        raise ValueError("calibration summary is not bound to the calibration identity")
    if calibration_summary.get("dataset_manifest_sha256") != (
        calibration_identity.get("dataset_manifest_sha256")
    ):
        raise ValueError("calibration summary dataset differs from its identity")
    if (
        calibration_summary.get("corpus_gcs_inventory_sha256")
        != corpus_inventory_sha256
    ):
        raise ValueError("calibration summary corpus inventory differs from identity")
    if (
        calibration_summary.get("source_records") != source_records
        or calibration_summary.get("full_usable_records") != full_usable_records
        or calibration_summary.get("processed_records") != full_usable_records
        or calibration_summary.get("collective_steps") != collective_steps
        or calibration_summary.get("maximum_usable_input_tokens")
        != maximum_usable_input_tokens
    ):
        raise ValueError("calibration summary record/step coverage is incomplete")
    if calibration_summary.get("minimum_observed_input_tokens") != minimum_input_tokens:
        raise ValueError("calibration summary changed the input-token threshold")
    if observed_input_tokens < minimum_input_tokens:
        raise ValueError("calibration observed fewer input tokens than required")
    if observed_input_tokens > maximum_usable_input_tokens:
        raise ValueError("calibration observed more input tokens than its usable corpus")

    if mass_candidate.get("schema_version") != 1:
        raise ValueError("mass candidate has an unsupported schema")
    if mass_candidate.get("mass_gate_satisfied") is not True:
        raise ValueError("mass candidate did not pass")
    if mass_candidate.get("source_revision") != calibration_identity.get(
        "source_revision"
    ):
        raise ValueError("mass candidate source revision differs from calibration")
    if mass_candidate.get("promotion_allowed") is not False:
        raise ValueError("mass candidate has an invalid promotion state")
    required_candidate_true = (
        "promotion_eligible",
        "full_corpus_consumed",
        "minimum_tokens_observed",
    )
    if any(mass_candidate.get(key) is not True for key in required_candidate_true):
        raise ValueError("mass candidate is not backed by full calibration")
    if (
        mass_candidate.get("calibration_mode") != "full"
        or mass_candidate.get("debug_run") is not False
        or mass_candidate.get("calibration_run_sha256") != calibration_run_sha256
    ):
        raise ValueError("mass candidate is bound to a debug or different calibration")
    if (
        mass_candidate.get("dataset_manifest_sha256")
        != calibration_identity.get("dataset_manifest_sha256")
        or mass_candidate.get("corpus_gcs_inventory_sha256")
        != corpus_inventory_sha256
        or mass_candidate.get("observed_input_tokens") != observed_input_tokens
        or mass_candidate.get("minimum_observed_input_tokens")
        != minimum_input_tokens
        or mass_candidate.get("full_usable_records") != full_usable_records
        or mass_candidate.get("processed_records") != full_usable_records
    ):
        raise ValueError("mass candidate calibration counts or dataset differ")

    retained_experts = mass_candidate.get("retained_experts")
    if retained_experts == 32:
        summary_path_key = "candidate"
        summary_digest_key = "candidate_sha256"
        summary_mass_key = "mass_gate_satisfied"
    elif retained_experts == 48:
        summary_path_key = "fallback_candidate"
        summary_digest_key = "fallback_candidate_sha256"
        summary_mass_key = "fallback_48_mass_gate_satisfied"
    else:
        raise ValueError("mass candidate retained-expert rung is unsupported")
    if (
        mass_candidate.get("min_coverage")
        != MASS_COVERAGE_BY_EXPERTS[retained_experts]
    ):
        raise ValueError("mass candidate changed the canonical coverage floor")
    if (
        calibration_summary.get(summary_path_key) != mass_candidate_path
        or calibration_summary.get(summary_digest_key) != mass_candidate_sha256
        or calibration_summary.get(summary_mass_key) is not True
    ):
        raise ValueError("mass candidate is not the one recorded by calibration")
    if normalize_source_identity(mass_candidate.get("sunfish_source")) != (
        normalize_source_identity(calibration_identity.get("sunfish_source"))
    ):
        raise ValueError("mass candidate source tree differs from calibration")
    return {
        "calibration_run_sha256": calibration_run_sha256,
        "calibration_summary_sha256": calibration_summary_sha256,
        "observed_input_tokens": observed_input_tokens,
        "minimum_observed_input_tokens": minimum_input_tokens,
        "maximum_usable_input_tokens": maximum_usable_input_tokens,
        "source_records": source_records,
        "full_usable_records": full_usable_records,
        "processed_records": full_usable_records,
        "dataset_manifest_sha256": calibration_identity[
            "dataset_manifest_sha256"
        ],
        "corpus_gcs_inventory_sha256": corpus_inventory_sha256,
        "mass_min_coverage": MASS_COVERAGE_BY_EXPERTS[retained_experts],
    }


def approved_selection_payload(
    mass_candidate: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    *,
    mass_candidate_sha256: str,
    reconstruction_run_sha256: str,
    reconstruction_summary_sha256: str,
    calibration_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only selection sidecar accepted by non-smoke seed creation."""
    if not mass_candidate.get("mass_gate_satisfied"):
        raise ValueError("router-mass candidate did not pass")
    if mass_candidate.get("promotion_allowed"):
        raise ValueError("mass-only candidate must not already be promotable")
    if mass_candidate.get("promotion_eligible") is not True:
        raise ValueError("mass-only candidate is not backed by full calibration")
    if not reconstruction.get("passed"):
        raise ValueError("reconstruction gate did not pass")
    if normalize_source_identity(mass_candidate.get("sunfish_source")) is None:
        raise ValueError("mass candidate has no valid Sunfish source identity")
    for digest in (
        mass_candidate_sha256,
        reconstruction_run_sha256,
        reconstruction_summary_sha256,
        calibration_provenance.get("calibration_run_sha256"),
        calibration_provenance.get("calibration_summary_sha256"),
        calibration_provenance.get("artifact_inventory_sha256"),
        calibration_provenance.get("corpus_gcs_inventory_sha256"),
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("promotion provenance requires SHA-256 digests")
    try:
        observed_input_tokens = int(calibration_provenance["observed_input_tokens"])
        minimum_input_tokens = int(
            calibration_provenance["minimum_observed_input_tokens"]
        )
        full_usable_records = int(calibration_provenance["full_usable_records"])
        processed_records = int(calibration_provenance["processed_records"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("promotion calibration counts are invalid") from error
    if (
        minimum_input_tokens != MIN_CALIBRATION_INPUT_TOKENS
        or observed_input_tokens < minimum_input_tokens
        or not 0 < processed_records == full_usable_records
        or mass_candidate.get("observed_input_tokens") != observed_input_tokens
        or mass_candidate.get("full_usable_records") != full_usable_records
        or mass_candidate.get("processed_records") != processed_records
        or mass_candidate.get("calibration_run_sha256")
        != calibration_provenance.get("calibration_run_sha256")
        or mass_candidate.get("dataset_manifest_sha256")
        != calibration_provenance.get("dataset_manifest_sha256")
        or mass_candidate.get("corpus_gcs_inventory_sha256")
        != calibration_provenance.get("corpus_gcs_inventory_sha256")
    ):
        raise ValueError("promotion calibration coverage is incomplete")
    retained_experts = mass_candidate.get("retained_experts")
    if (
        retained_experts not in MASS_COVERAGE_BY_EXPERTS
        or mass_candidate.get("min_coverage")
        != MASS_COVERAGE_BY_EXPERTS[retained_experts]
        or calibration_provenance.get("mass_min_coverage")
        != MASS_COVERAGE_BY_EXPERTS[retained_experts]
    ):
        raise ValueError("promotion router-mass threshold differs from canonical")
    expected_thresholds = reconstruction_thresholds()
    if reconstruction.get("thresholds") != expected_thresholds:
        raise ValueError("promotion reconstruction thresholds differ from canonical")
    if reconstruction.get("calibration_provenance") != dict(
        calibration_provenance
    ):
        raise ValueError("reconstruction summary changed calibration provenance")
    return {
        "schema_version": 1,
        "purpose": APPROVED_SELECTION_PURPOSE,
        "promotion_allowed": True,
        "selection_method": APPROVED_SELECTION_METHOD,
        "source_experts": SOURCE_EXPERTS,
        "retained_experts": mass_candidate["retained_experts"],
        "top_k_experts": mass_candidate["top_k_experts"],
        "source_revision": mass_candidate["source_revision"],
        "dataset_manifest_sha256": mass_candidate["dataset_manifest_sha256"],
        "sunfish_source": mass_candidate["sunfish_source"],
        "mass_candidate_sha256": mass_candidate_sha256,
        "calibration_run_sha256": calibration_provenance[
            "calibration_run_sha256"
        ],
        "calibration_summary_sha256": calibration_provenance[
            "calibration_summary_sha256"
        ],
        "calibration_artifact_inventory_sha256": calibration_provenance[
            "artifact_inventory_sha256"
        ],
        "calibration_corpus_gcs_inventory_sha256": calibration_provenance[
            "corpus_gcs_inventory_sha256"
        ],
        "mass_min_coverage": MASS_COVERAGE_BY_EXPERTS[retained_experts],
        "calibration_observed_input_tokens": observed_input_tokens,
        "calibration_minimum_observed_input_tokens": minimum_input_tokens,
        "calibration_full_usable_records": full_usable_records,
        "calibration_processed_records": processed_records,
        "reconstruction_run_sha256": reconstruction_run_sha256,
        "reconstruction_summary_sha256": reconstruction_summary_sha256,
        "mass_gate_satisfied": True,
        "reconstruction_gate_satisfied": True,
        "thresholds": reconstruction["thresholds"],
        "worst_relative_rmse": reconstruction["worst_relative_rmse"],
        "worst_cosine_similarity": reconstruction["worst_cosine_similarity"],
        "layers": mass_candidate["layers"],
    }


def _path_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable(path: Any, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable reconstruction evidence changed at {path}")
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


def _validate_selection(
    payload: Mapping[str, Any],
) -> tuple[dict[int, tuple[int, ...]], int]:
    if payload.get("source_experts") != SOURCE_EXPERTS:
        raise ValueError("mass candidate source expert count differs")
    retained_experts = int(payload.get("retained_experts", 0))
    if retained_experts not in {32, 48}:
        raise ValueError("mass candidate must retain either 32 or 48 experts")
    if payload.get("top_k_experts") != TOP_K:
        raise ValueError("reconstruction must evaluate the 32/8 storage-pruning rung")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != {
        str(layer) for layer in range(NUM_LAYERS)
    }:
        raise ValueError("mass candidate must contain exactly layers 0..29")
    selection = {}
    for layer in range(NUM_LAYERS):
        experts = tuple(int(value) for value in layers[str(layer)])
        if len(experts) != retained_experts:
            raise ValueError(
                f"layer {layer} does not retain {retained_experts} experts"
            )
        if experts != tuple(sorted(set(experts))):
            raise ValueError(f"layer {layer} expert IDs are not unique and sorted")
        if any(value < 0 or value >= SOURCE_EXPERTS for value in experts):
            raise ValueError(f"layer {layer} expert ID is out of range")
        selection[layer] = experts
    return selection, retained_experts


def _find_layer_tree(tree: Any, layer: int) -> Any:
    target = f"layer_{layer}"
    found = []

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        if target in value and isinstance(value[target], Mapping):
            found.append(value[target])
        for child in value.values():
            visit(child)

    visit(tree)
    if len(found) != 1:
        raise ValueError(f"parameter tree contains {len(found)} matches for {target}")
    return found[0]


def _ffw_parameter_tree(layer_tree: Mapping[str, Any], *, use_post_norm: bool):
    names = {"pre_ffw2_norm", "mlp2", "pre_ffw_norm", "mlp"}
    if use_post_norm:
        names.update({"post_ffw2_norm", "post_ffw1_norm", "post_ffw_norm"})
    missing = names - set(layer_tree)
    if missing:
        raise ValueError(f"layer FFW parameter tree is missing {sorted(missing)}")
    return {name: layer_tree[name] for name in sorted(names)}


def _load_artifact(
    path: Any,
    *,
    np: Any,
    run_id: str,
    calibration_run_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(path.read_text())
    if metadata.get("schema_version") != 1 or set(metadata.get("fields", {})) != set(
        _FIELDS
    ):
        raise ValueError(f"invalid reconstruction manifest {path}")
    if (
        metadata.get("run_id") != run_id
        or metadata.get("calibration_run_sha256") != calibration_run_sha256
        or metadata.get("artifact_id") != path.stem
    ):
        raise ValueError(f"reconstruction manifest lineage differs at {path}")
    bucket = metadata.get("bucket")
    if bucket not in RECONSTRUCTION_BUCKETS:
        raise ValueError(f"unknown reconstruction bucket {bucket!r} at {path}")
    tokens = int(metadata.get("tokens", -1))
    if tokens <= 0:
        raise ValueError(f"invalid reconstruction token count at {path}")
    arrays = {}
    for name in _FIELDS:
        field = metadata["fields"][name]
        field_path = path.parent / field["path"]
        if field_path.name != field["path"]:
            raise ValueError(f"artifact field path escapes its host directory at {path}")
        payload = field_path.read_bytes()
        if len(payload) != int(field["bytes"]):
            raise ValueError(f"artifact byte count differs for {field_path}")
        if hashlib.sha256(payload).hexdigest() != field["sha256"]:
            raise ValueError(f"artifact hash differs for {field_path}")
        dtype = np.dtype(field["dtype"])
        array = np.frombuffer(payload, dtype=dtype).reshape(tuple(field["shape"]))
        arrays[name] = array
    residual = arrays["shared_pre_router_residual"]
    choices = arrays["topk_indices"]
    weights = arrays["final_scaled_topk_weights"]
    if residual.shape != (tokens, NUM_LAYERS, HIDDEN_SIZE) or residual.dtype.itemsize != 2:
        raise ValueError(f"residual artifact shape/dtype differs at {path}")
    if choices.shape != (tokens, NUM_LAYERS, TOP_K) or choices.dtype != np.uint8:
        raise ValueError(f"top-k artifact shape/dtype differs at {path}")
    if weights.shape != (tokens, NUM_LAYERS, TOP_K) or weights.dtype != np.float16:
        raise ValueError(f"weight artifact shape/dtype differs at {path}")
    return metadata, arrays


def _make_reconstruction_modules(
    *, nn: Any, config: Any, retained_experts: int
):
    from gemma.gm.nn.gemma4 import _layers, _modules, _moe

    class ReconstructionFFW(nn.Module):
        num_experts: int

        @nn.compact
        def __call__(self, residual):
            dense = _layers.RMSNorm(name="pre_ffw2_norm")(residual)
            dense = _modules.FeedForward(
                features=config.embed_dim,
                hidden_dim=config.moe_dense_hidden_dim,
                name="mlp2",
            )(dense)
            if config.use_post_ffw_norm:
                dense = _layers.RMSNorm(name="post_ffw2_norm")(dense)

            routed_input = _layers.RMSNorm(name="pre_ffw_norm")(residual)
            routed = _moe.MoERagged(
                features=config.embed_dim,
                hidden_dim=config.expert_dim,
                num_experts=self.num_experts,
                num_experts_per_datapoint=TOP_K,
                name="mlp",
            )(routed_input, unnormalized_x=residual)
            if config.use_post_ffw_norm:
                routed = _layers.RMSNorm(name="post_ffw1_norm")(routed)

            output = dense + routed
            if config.use_post_ffw_norm:
                output = _layers.RMSNorm(name="post_ffw_norm")(output)
            return output

    return (
        ReconstructionFFW(num_experts=SOURCE_EXPERTS),
        ReconstructionFFW(num_experts=retained_experts),
    )


def run_reconstruction_gate(
    *,
    source_checkpoint: str,
    source_anonymous: bool,
    calibration_dir: str,
    raw_dir: str,
    candidate_path: str,
    output_dir: str,
    run_id: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
    max_relative_rmse: float,
    min_cosine_similarity: float,
    min_total_tokens: int,
    min_tokens_per_bucket: int,
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid reconstruction run ID")
    canonical = {
        "maximum relative RMSE": (
            max_relative_rmse,
            RECONSTRUCTION_MAX_RELATIVE_RMSE,
        ),
        "minimum cosine similarity": (
            min_cosine_similarity,
            RECONSTRUCTION_MIN_COSINE_SIMILARITY,
        ),
        "minimum total reconstruction tokens": (
            min_total_tokens,
            RECONSTRUCTION_SAMPLE_TOKENS,
        ),
        "minimum reconstruction tokens per bucket": (
            min_tokens_per_bucket,
            RECONSTRUCTION_MIN_TOKENS_PER_BUCKET,
        ),
    }
    changed = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in canonical.items()
        if actual != expected
    ]
    if changed:
        raise ValueError(
            "Stage-1 reconstruction thresholds are canonical and non-overridable: "
            + "; ".join(changed)
        )
    require_launcher_run_id(run_id)
    # Refuse Stage-1 continuation from a missing/edited readiness receipt before
    # distributed JAX can initialize a backend.
    from etils import epath
    from sunfish_tpu.readiness_ledger import validate_readiness_unlock

    current_source_identity = source_identity_from_environment(required=True)
    calibration_root = epath.Path(calibration_dir)
    raw_root = epath.Path(raw_dir)
    candidate_file = epath.Path(candidate_path)
    calibration_identity_file = calibration_root / "calibration-run.json"
    calibration_summary_file = calibration_root / "summary.json"
    calibration_identity_bytes = calibration_identity_file.read_bytes()
    calibration_summary_bytes = calibration_summary_file.read_bytes()
    mass_candidate_bytes = candidate_file.read_bytes()
    calibration_identity = json.loads(calibration_identity_bytes)
    calibration_summary = json.loads(calibration_summary_bytes)
    mass_candidate = json.loads(mass_candidate_bytes)
    if calibration_identity.get("source_checkpoint") != source_checkpoint:
        raise ValueError("source checkpoint differs from calibration identity")
    if normalize_source_identity(calibration_identity.get("sunfish_source")) != (
        normalize_source_identity(current_source_identity)
    ):
        raise ValueError("reconstruction source tree differs from calibration")
    if calibration_summary.get("raw_artifact_prefix") != str(raw_root):
        raise ValueError("raw reconstruction artifacts differ from calibration summary")
    candidate_sha256 = hashlib.sha256(mass_candidate_bytes).hexdigest()
    calibration_identity_sha256 = hashlib.sha256(
        calibration_identity_bytes
    ).hexdigest()
    calibration_summary_sha256 = hashlib.sha256(
        calibration_summary_bytes
    ).hexdigest()
    calibration_provenance = validate_calibration_for_reconstruction(
        calibration_identity,
        calibration_summary,
        mass_candidate,
        calibration_run_sha256=calibration_identity_sha256,
        calibration_summary_sha256=calibration_summary_sha256,
        mass_candidate_sha256=candidate_sha256,
        mass_candidate_path=str(candidate_file),
    )
    artifact_inventory_file = (
        calibration_root / "reconstruction-artifact-inventory.json"
    )
    if calibration_summary.get("artifact_inventory") != str(
        artifact_inventory_file
    ):
        raise ValueError("calibration summary artifact inventory path differs")
    expected_artifact_inventory_sha256 = calibration_summary.get(
        "artifact_inventory_sha256"
    )
    if not isinstance(expected_artifact_inventory_sha256, str) or not _SHA256.fullmatch(
        expected_artifact_inventory_sha256
    ):
        raise ValueError("calibration summary has no artifact inventory digest")
    artifact_inventory_bytes = artifact_inventory_file.read_bytes()
    if (
        hashlib.sha256(artifact_inventory_bytes).hexdigest()
        != expected_artifact_inventory_sha256
    ):
        raise ValueError("calibration artifact inventory bytes changed")
    artifact_inventory = validate_artifact_inventory(
        json.loads(artifact_inventory_bytes),
        root=raw_root,
        run_id=calibration_identity["run_id"],
        calibration_run_sha256=calibration_identity_sha256,
        expected_processes=int(calibration_identity["topology"]["processes"]),
        allowed_buckets=RECONSTRUCTION_BUCKETS,
        field_names=_FIELDS,
    )
    if (
        artifact_inventory.get("total_tokens")
        != calibration_summary.get("artifact_tokens")
        or artifact_inventory.get("tokens_by_bucket")
        != calibration_summary.get("artifact_tokens_by_bucket")
    ):
        raise ValueError("calibration artifact inventory counts differ from summary")
    verify_live_artifact_inventory(raw_root, artifact_inventory)
    calibration_provenance["artifact_inventory_sha256"] = (
        expected_artifact_inventory_sha256
    )
    readiness_ledger_path = calibration_identity.get("readiness_ledger")
    readiness_ledger_sha256 = calibration_identity.get(
        "readiness_ledger_sha256"
    )
    if not isinstance(readiness_ledger_path, str) or not _SHA256.fullmatch(
        str(readiness_ledger_sha256 or "")
    ):
        raise ValueError("calibration identity has no pinned readiness ledger")
    readiness_bytes = epath.Path(readiness_ledger_path).read_bytes()
    if hashlib.sha256(readiness_bytes).hexdigest() != readiness_ledger_sha256:
        raise ValueError("calibration readiness ledger bytes changed")
    validate_readiness_unlock(
        json.loads(readiness_bytes),
        expected_source=current_source_identity,
        expected_devices=expected_devices,
        expected_processes=expected_processes,
        expected_local_devices=expected_local_devices,
    )
    # No backend-adjacent import may move above distributed initialization.
    jax, initialization = initialize_distributed_jax(require_distributed=True)
    import flax
    import flax.linen as nn
    from gemma import gm
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    import numpy as np

    from sunfish_tpu.calibration import call_with_router_artifacts
    from sunfish_tpu.gcs_inventory import verify_live_gcs_inventory
    from sunfish_tpu.orbax_seed import (
        AUDITED_SOURCE_TEXT_PARAMETERS,
        _prune_nested_params,
        _target_abstract_params,
        _tree_signature,
        audited_target_text_parameters,
        require_parameter_count,
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
        raise RuntimeError(f"reconstruction topology failed: {json.dumps(topology)}")
    versions = verify_runtime_contract(require_tpu=True)
    shardings = make_teacher_mesh_and_shardings(jax, np)
    mesh = shardings["mesh"]
    root = epath.Path(output_dir) / run_id
    selection, retained_experts = _validate_selection(mass_candidate)
    if not mass_candidate.get("mass_gate_satisfied"):
        raise ValueError("router-mass candidate failed; reconstruction is not authorized")
    if mass_candidate.get("promotion_allowed"):
        raise ValueError("input mass candidate is already promotable")
    source_inventory = calibration_identity.get("source_gcs_inventory")
    if not isinstance(source_inventory, Mapping):
        raise ValueError("calibration identity has no teacher GCS inventory")
    verify_live_gcs_inventory(
        source_checkpoint, source_inventory, anonymous=source_anonymous
    )
    if mass_candidate.get("source_revision") != calibration_identity.get(
        "source_revision"
    ):
        raise ValueError("candidate source revision differs from calibration identity")
    if mass_candidate.get("dataset_manifest_sha256") != calibration_identity.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("candidate dataset differs from calibration identity")
    if normalize_source_identity(mass_candidate.get("sunfish_source")) != (
        normalize_source_identity(calibration_identity.get("sunfish_source"))
    ):
        raise ValueError("candidate source tree differs from calibration identity")

    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "source_checkpoint": source_checkpoint,
        "source_revision": calibration_identity["source_revision"],
        "calibration_run_id": calibration_identity["run_id"],
        "calibration_identity_sha256": calibration_identity_sha256,
        "calibration_summary": str(calibration_summary_file),
        "calibration_summary_sha256": calibration_summary_sha256,
        "calibration_artifact_inventory": str(artifact_inventory_file),
        "calibration_artifact_inventory_sha256": (
            expected_artifact_inventory_sha256
        ),
        "calibration_corpus_gcs_inventory_sha256": calibration_provenance[
            "corpus_gcs_inventory_sha256"
        ],
        "calibration_observed_input_tokens": calibration_provenance[
            "observed_input_tokens"
        ],
        "calibration_minimum_observed_input_tokens": calibration_provenance[
            "minimum_observed_input_tokens"
        ],
        "calibration_full_usable_records": calibration_provenance[
            "full_usable_records"
        ],
        "readiness_ledger": readiness_ledger_path,
        "readiness_ledger_sha256": readiness_ledger_sha256,
        "mass_candidate": candidate_path,
        "mass_candidate_sha256": candidate_sha256,
        "raw_artifacts": raw_dir,
        "thresholds": {
            "max_relative_rmse": max_relative_rmse,
            "min_cosine_similarity": min_cosine_similarity,
            "min_total_tokens": min_total_tokens,
            "min_tokens_per_bucket": min_tokens_per_bucket,
        },
        "runtime_versions": versions,
        "sunfish_source": current_source_identity,
        "topology": {
            "devices": expected_devices,
            "processes": expected_processes,
            "local_devices": expected_local_devices,
        },
    }
    if int(jax.process_index()) == 0:
        _write_immutable(root / "reconstruction-run.json", identity)
    multihost_utils.sync_global_devices(f"sunfish-reconstruction-identity-{run_id}")
    if json.loads((root / "reconstruction-run.json").read_text()) != identity:
        raise RuntimeError("reconstruction identity readback differs")

    source_abstract = _target_abstract_params(
        num_experts=SOURCE_EXPERTS,
        top_k_experts=TOP_K,
        jax=jax,
        jnp=jnp,
    )
    source_shardings = shardings["params"](source_abstract)
    sharded_source = jax.tree.map(
        lambda value, sharding: jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=sharding
        ),
        source_abstract,
        source_shardings,
    )
    source_params = gm.ckpts.load_params(
        epath.Path(source_checkpoint),
        params=sharded_source,
        donate=False,
        text_only=True,
    )
    _validate_exact_tree(sharded_source, source_params)
    source_signature = _tree_signature(source_params, flax)
    require_parameter_count(
        source_signature,
        expected=AUDITED_SOURCE_TEXT_PARAMETERS,
        label="reconstruction teacher",
    )

    with mesh:
        candidate_params, pruning = _prune_nested_params(
            source_params,
            selection=selection,
            retained_experts=retained_experts,
            flax=flax,
            jax=jax,
            jnp=jnp,
            delete_source_arrays=False,
        )
        candidate_abstract = _target_abstract_params(
            num_experts=retained_experts,
            top_k_experts=TOP_K,
            jax=jax,
            jnp=jnp,
        )
        candidate_shardings = shardings["params"](candidate_abstract)
        sharded_candidate = jax.tree.map(
            lambda value, sharding: jax.ShapeDtypeStruct(
                value.shape, value.dtype, sharding=sharding
            ),
            candidate_abstract,
            candidate_shardings,
        )
        candidate_params = jax.tree.map(
            lambda value, sharding: jax.device_put(value, sharding),
            candidate_params,
            candidate_shardings,
        )
        _validate_exact_tree(sharded_candidate, candidate_params)
        candidate_signature = _tree_signature(candidate_params, flax)
        require_parameter_count(
            candidate_signature,
            expected=audited_target_text_parameters(retained_experts),
            label="reconstruction candidate",
        )

    network = make_gemma_network(
        num_experts=SOURCE_EXPERTS,
        top_k_experts=TOP_K,
        dtype="bfloat16",
        use_lora=False,
        lora_rank=1,
    )
    teacher_ffw, candidate_ffw = _make_reconstruction_modules(
        nn=nn,
        config=network.gemma_model.config,
        retained_experts=retained_experts,
    )
    use_post_norm = bool(network.gemma_model.config.use_post_ffw_norm)
    source_layers = [
        _ffw_parameter_tree(
            _find_layer_tree(source_params, layer), use_post_norm=use_post_norm
        )
        for layer in range(NUM_LAYERS)
    ]
    candidate_layers = [
        _ffw_parameter_tree(
            _find_layer_tree(candidate_params, layer), use_post_norm=use_post_norm
        )
        for layer in range(NUM_LAYERS)
    ]
    metric_sums = jax.device_put(
        jnp.zeros((len(RECONSTRUCTION_BUCKETS), NUM_LAYERS, 4), jnp.float32),
        shardings["replicated"],
    )
    token_counts = jax.device_put(
        jnp.zeros((len(RECONSTRUCTION_BUCKETS),), jnp.int32),
        shardings["replicated"],
    )
    route_mismatches = jax.device_put(
        jnp.zeros((), jnp.int32), shardings["replicated"]
    )

    @jax.jit
    def add_token_counts(current, bucket_ids, valid):
        onehot = (
            (bucket_ids[:, None] == jnp.arange(len(RECONSTRUCTION_BUCKETS))[None, :])
            & valid[:, None]
        )
        updated = current + jnp.sum(onehot, axis=0, dtype=jnp.int32)
        return jax.lax.with_sharding_constraint(updated, shardings["replicated"])

    @jax.jit
    def evaluate_layer(
        source_layer,
        candidate_layer,
        residual,
        recorded_choices,
        recorded_weights,
        bucket_ids,
        valid,
        layer_index,
        current_sums,
        current_mismatches,
    ):
        residual = residual[:, None, :]

        def teacher_forward():
            return teacher_ffw.apply({"params": source_layer}, residual)

        teacher_output, observed = call_with_router_artifacts(
            teacher_forward, expected_layers=1
        )
        candidate_output = candidate_ffw.apply(
            {"params": candidate_layer}, residual
        )
        teacher_output = teacher_output[:, 0, :].astype(jnp.float32)
        candidate_output = candidate_output[:, 0, :].astype(jnp.float32)
        observed_choices = observed["topk_indices"][:, 0, :]
        observed_weights = observed["final_scaled_topk_weights"][:, 0, :]
        mismatches = jnp.sum(
            ((observed_choices != recorded_choices) & valid[:, None]).astype(jnp.int32)
        ) + jnp.sum(
            (
                (observed_weights.astype(jnp.float16) != recorded_weights)
                & valid[:, None]
            ).astype(jnp.int32)
        )

        difference = candidate_output - teacher_output
        values = jnp.stack(
            (
                jnp.sum(jnp.square(difference), axis=-1),
                jnp.sum(jnp.square(teacher_output), axis=-1),
                jnp.sum(teacher_output * candidate_output, axis=-1),
                jnp.sum(jnp.square(candidate_output), axis=-1),
            ),
            axis=-1,
        )
        onehot = jnp.asarray(
            (
                bucket_ids[:, None]
                == jnp.arange(len(RECONSTRUCTION_BUCKETS))[None, :]
            )
            & valid[:, None],
            jnp.float32,
        )
        contribution = jnp.einsum("tb,tm->bm", onehot, values)
        updated_sums = current_sums.at[:, layer_index, :].add(contribution)
        return (
            jax.lax.with_sharding_constraint(
                updated_sums, shardings["replicated"]
            ),
            jax.lax.with_sharding_constraint(
                current_mismatches + mismatches,
                shardings["replicated"],
            ),
        )

    process_index = int(jax.process_index())
    local_paths = paths_for_process(raw_root, artifact_inventory, process_index)
    manifest_counts = np.asarray(
        multihost_utils.process_allgather(np.asarray(len(local_paths), np.int32))
    ).reshape(-1)
    max_manifests = int(np.max(manifest_counts))
    if max_manifests <= 0:
        raise FileNotFoundError("no reconstruction artifacts found")
    local_tokens_seen = 0
    data_tensor = NamedSharding(mesh, P("data", None, None))
    data_vector = NamedSharding(mesh, P("data"))

    with mesh:
        for artifact_index in range(max_manifests):
            if artifact_index < len(local_paths):
                metadata, arrays = _load_artifact(
                    local_paths[artifact_index],
                    np=np,
                    run_id=calibration_identity["run_id"],
                    calibration_run_sha256=calibration_identity_sha256,
                )
                local_count = int(metadata["tokens"])
                local_bucket = RECONSTRUCTION_BUCKETS.index(metadata["bucket"])
                local_tokens_seen += local_count
            else:
                local_count = 0
                local_bucket = -1
                arrays = {
                    "shared_pre_router_residual": np.empty(
                        (0, NUM_LAYERS, HIDDEN_SIZE), dtype=jnp.bfloat16
                    ),
                    "topk_indices": np.empty((0, NUM_LAYERS, TOP_K), np.uint8),
                    "final_scaled_topk_weights": np.empty(
                        (0, NUM_LAYERS, TOP_K), np.float16
                    ),
                }
            step_counts = np.asarray(
                multihost_utils.process_allgather(
                    np.asarray(local_count, np.int32)
                )
            ).reshape(-1)
            batch_per_process = int(np.max(step_counts))
            if batch_per_process <= 0:
                raise RuntimeError("artifact alignment produced an empty collective")

            def padded(array, fill=0):
                pad = batch_per_process - array.shape[0]
                if pad < 0:
                    raise RuntimeError("artifact token count exceeds collective batch")
                return np.pad(array, ((0, pad),) + ((0, 0),) * (array.ndim - 1), constant_values=fill)

            local_residual = padded(arrays["shared_pre_router_residual"])
            local_choices = padded(arrays["topk_indices"])
            local_weights = padded(arrays["final_scaled_topk_weights"])
            local_bucket_ids = np.full((batch_per_process,), -1, np.int32)
            local_bucket_ids[:local_count] = local_bucket
            local_valid = np.zeros((batch_per_process,), np.bool_)
            local_valid[:local_count] = True

            def global_array(local, sharding):
                global_shape = (expected_processes * batch_per_process,) + local.shape[1:]
                return jax.make_array_from_process_local_data(
                    sharding, local, global_shape=global_shape
                )

            residual = global_array(local_residual, data_tensor)
            choices = global_array(local_choices, data_tensor)
            weights = global_array(local_weights, data_tensor)
            bucket_ids = global_array(local_bucket_ids, data_vector)
            valid = global_array(local_valid, data_vector)
            token_counts = add_token_counts(token_counts, bucket_ids, valid)
            for layer in range(NUM_LAYERS):
                metric_sums, route_mismatches = evaluate_layer(
                    source_layers[layer],
                    candidate_layers[layer],
                    residual[:, layer, :],
                    choices[:, layer, :],
                    weights[:, layer, :],
                    bucket_ids,
                    valid,
                    jnp.asarray(layer, jnp.int32),
                    metric_sums,
                    route_mismatches,
                )
            if (artifact_index + 1) % 8 == 0:
                jax.block_until_ready(metric_sums)

    jax.block_until_ready((metric_sums, token_counts, route_mismatches))
    host_evidence = {
        "schema_version": 1,
        "run_id": run_id,
        "process_index": process_index,
        "artifact_manifests": len(local_paths),
        "artifact_tokens": local_tokens_seen,
        "completed": True,
    }
    _write_immutable(root / f"host-{process_index:05d}.json", host_evidence)
    multihost_utils.sync_global_devices(f"sunfish-reconstruction-hosts-{run_id}")

    def finalize_reconstruction():
        reconstruction = summarize_reconstruction(
            np.asarray(metric_sums).tolist(),
            np.asarray(token_counts).tolist(),
            route_mismatches=int(np.asarray(route_mismatches)),
            max_relative_rmse=max_relative_rmse,
            min_cosine_similarity=min_cosine_similarity,
            min_total_tokens=min_total_tokens,
            min_tokens_per_bucket=min_tokens_per_bucket,
        )
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            **reconstruction,
            "calibration_provenance": dict(calibration_provenance),
            "source_tree": source_signature,
            "candidate_tree": candidate_signature,
            "pruning": pruning,
            "mass_candidate_sha256": candidate_sha256,
            "hardware_topology": topology,
            "host_artifacts": [
                json.loads((root / f"host-{process:05d}.json").read_text())
                for process in range(expected_processes)
            ],
        }
        _write_immutable(root / "summary.json", summary)
        if summary["passed"]:
            run_sha256 = _path_sha256(root / "reconstruction-run.json")
            summary_sha256 = _path_sha256(root / "summary.json")
            approved = approved_selection_payload(
                mass_candidate,
                summary,
                mass_candidate_sha256=candidate_sha256,
                reconstruction_run_sha256=run_sha256,
                reconstruction_summary_sha256=summary_sha256,
                calibration_provenance=calibration_provenance,
            )
            _write_immutable(calibration_root / "selection-approved.json", approved)
        return summary

    summary = None
    finalization_error = None
    if process_index == 0:
        try:
            summary = finalize_reconstruction()
        except Exception as error:  # propagate instead of stranding peer collectives
            finalization_error = f"{type(error).__name__}: {error}"
    finalization_error = _broadcast_process0_error(
        multihost_utils, np, finalization_error
    )
    if finalization_error is not None:
        raise RuntimeError(
            "reconstruction process-0 finalization failed: " + finalization_error
        )
    multihost_utils.sync_global_devices(f"sunfish-reconstruction-summary-{run_id}")
    return summary if summary is not None else host_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-anonymous", action="store_true")
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    parser.add_argument("--max-relative-rmse", type=float, required=True)
    parser.add_argument("--min-cosine-similarity", type=float, required=True)
    parser.add_argument("--min-total-tokens", type=int, default=100_000)
    parser.add_argument("--min-tokens-per-bucket", type=int, default=4_000)
    args = parser.parse_args(argv)
    try:
        payload = run_reconstruction_gate(
            source_checkpoint=args.source_checkpoint,
            source_anonymous=args.source_anonymous,
            calibration_dir=args.calibration_dir,
            raw_dir=args.raw_dir,
            candidate_path=args.candidate,
            output_dir=args.output_dir,
            run_id=args.run_id,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
            max_relative_rmse=args.max_relative_rmse,
            min_cosine_similarity=args.min_cosine_similarity,
            min_total_tokens=args.min_total_tokens,
            min_tokens_per_bucket=args.min_tokens_per_bucket,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"sunfish-reconstruction-gate: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
