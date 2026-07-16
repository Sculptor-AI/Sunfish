"""Dependency-free provenance checks for exact-tree Orbax seed manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sunfish_tpu.source_identity import normalize_source_identity
from sunfish_tpu.gcs_inventory import validate_gcs_inventory
from sunfish_tpu.stage1_contract import (
    APPROVED_SELECTION_METHOD,
    APPROVED_SELECTION_PURPOSE,
    MASS_COVERAGE_BY_EXPERTS,
    MIN_CALIBRATION_INPUT_TOKENS,
    RECONSTRUCTION_MAX_RELATIVE_RMSE,
    RECONSTRUCTION_MIN_COSINE_SIMILARITY,
    reconstruction_thresholds,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GCS_REVISION = re.compile(r"^gcs-inventory-sha256:[0-9a-f]{64}$")
_SELECTION_LAYERS = 30
STAGE05_PURPOSE = "stage-0.5-infrastructure-readiness-only"
STAGE1_APPROVED_PURPOSE = APPROVED_SELECTION_PURPOSE
STAGE1_APPROVED_METHOD = APPROVED_SELECTION_METHOD
_APPROVAL_DIGESTS = (
    "mass_candidate_sha256",
    "calibration_run_sha256",
    "calibration_summary_sha256",
    "calibration_artifact_inventory_sha256",
    "calibration_corpus_gcs_inventory_sha256",
    "reconstruction_run_sha256",
    "reconstruction_summary_sha256",
)


def canonical_layer_selection_sha256(layers: Mapping[str, Any]) -> str:
    """Hash one already-normalized ``{"0": [...], ..., "29": [...]}`` map."""
    canonical = json.dumps(
        layers, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _validated_layer_selection(
    payload: Mapping[str, Any],
    *,
    source_experts: int,
    retained_experts: int,
) -> tuple[dict[str, list[int]], str]:
    raw_layers = payload.get("layers")
    expected_layers = {str(layer) for layer in range(_SELECTION_LAYERS)}
    if not isinstance(raw_layers, Mapping) or set(raw_layers) != expected_layers:
        raise ValueError("selection must contain exactly layers 0..29")
    layers: dict[str, list[int]] = {}
    for layer in range(_SELECTION_LAYERS):
        values = raw_layers[str(layer)]
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"selection layer {layer} is invalid")
        experts = tuple(values)
        if (
            len(experts) != retained_experts
            or experts != tuple(sorted(set(experts)))
            or any(value < 0 or value >= source_experts for value in experts)
        ):
            raise ValueError(f"selection layer {layer} is invalid")
        layers[str(layer)] = list(experts)
    digest = canonical_layer_selection_sha256(layers)
    declared_digest = payload.get("layers_sha256")
    if declared_digest is not None and declared_digest != digest:
        raise ValueError("selection layer digest differs from its exact expert IDs")
    return layers, digest


def _base_selection_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    promotion_allowed = payload.get("promotion_allowed")
    if not isinstance(promotion_allowed, bool):
        raise ValueError("selection manifest must set boolean promotion_allowed")
    purpose = payload.get("purpose")
    method = payload.get("selection_method")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("selection manifest must set purpose")
    if not isinstance(method, str) or not method:
        raise ValueError("selection manifest must set selection_method")
    try:
        source_experts = int(payload["source_experts"])
        retained_experts = int(payload["retained_experts"])
        top_k_experts = int(payload["top_k_experts"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection manifest must set integer expert counts") from error
    if not 0 < top_k_experts <= retained_experts <= source_experts:
        raise ValueError("selection manifest expert counts are invalid")
    return {
        "schema_version": payload.get("schema_version"),
        "purpose": purpose,
        "promotion_allowed": promotion_allowed,
        "selection_method": method,
        "source_experts": source_experts,
        "retained_experts": retained_experts,
        "top_k_experts": top_k_experts,
    }


def _validate_approved_selection(
    payload: Mapping[str, Any], *, require_layers: bool
) -> dict[str, Any]:
    """Validate the only schema allowed to seed a non-smoke phase."""
    metadata = _base_selection_metadata(payload)
    retained_experts = metadata["retained_experts"]
    if (
        metadata["schema_version"] != 1
        or metadata["purpose"] != STAGE1_APPROVED_PURPOSE
        or metadata["selection_method"] != STAGE1_APPROVED_METHOD
        or metadata["promotion_allowed"] is not True
        or metadata["source_experts"] != 128
        or retained_experts not in MASS_COVERAGE_BY_EXPERTS
        or metadata["top_k_experts"] != 8
        or payload.get("mass_gate_satisfied") is not True
        or payload.get("reconstruction_gate_satisfied") is not True
        or payload.get("mass_min_coverage")
        != MASS_COVERAGE_BY_EXPERTS[retained_experts]
    ):
        raise ValueError("production selection is not a canonical Stage-1 approval")
    for key in _APPROVAL_DIGESTS:
        digest = payload.get(key)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"production selection has invalid {key}")
    dataset_manifest_sha256 = payload.get("dataset_manifest_sha256")
    if not isinstance(dataset_manifest_sha256, str) or not _SHA256.fullmatch(
        dataset_manifest_sha256
    ):
        raise ValueError("production selection has invalid dataset manifest digest")
    if normalize_source_identity(payload.get("sunfish_source")) is None:
        raise ValueError("production selection has invalid Sunfish source identity")
    source_revision = payload.get("source_revision")
    if not isinstance(source_revision, str) or not _GCS_REVISION.fullmatch(
        source_revision
    ):
        raise ValueError("production selection source revision is not a GCS inventory")
    try:
        observed = int(payload["calibration_observed_input_tokens"])
        minimum = int(payload["calibration_minimum_observed_input_tokens"])
        full_records = int(payload["calibration_full_usable_records"])
        processed_records = int(payload["calibration_processed_records"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("production selection calibration counts are invalid") from error
    if (
        minimum != MIN_CALIBRATION_INPUT_TOKENS
        or observed < minimum
        or not 0 < full_records == processed_records
    ):
        raise ValueError("production selection calibration coverage is incomplete")
    if payload.get("thresholds") != reconstruction_thresholds():
        raise ValueError("production selection reconstruction thresholds changed")
    for key, relation, threshold in (
        (
            "worst_relative_rmse",
            "maximum",
            RECONSTRUCTION_MAX_RELATIVE_RMSE,
        ),
        (
            "worst_cosine_similarity",
            "minimum",
            RECONSTRUCTION_MIN_COSINE_SIMILARITY,
        ),
    ):
        metric = payload.get(key)
        if not isinstance(metric, Mapping):
            raise ValueError(f"production selection has invalid {key}")
        try:
            value = float(metric["value"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"production selection has invalid {key}") from error
        if not math.isfinite(value) or (
            relation == "maximum" and value > threshold
        ) or (relation == "minimum" and value < threshold):
            raise ValueError(f"production selection failed {key}")
    layers: dict[str, list[int]] | None = None
    layers_sha256: str | None = None
    if require_layers:
        layers, layers_sha256 = _validated_layer_selection(
            payload,
            source_experts=metadata["source_experts"],
            retained_experts=retained_experts,
        )
    result = {
        **metadata,
        "source_revision": source_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "sunfish_source": dict(payload["sunfish_source"]),
        "mass_gate_satisfied": True,
        "reconstruction_gate_satisfied": True,
        "mass_min_coverage": MASS_COVERAGE_BY_EXPERTS[retained_experts],
        **{key: payload[key] for key in _APPROVAL_DIGESTS},
        "calibration_observed_input_tokens": observed,
        "calibration_minimum_observed_input_tokens": minimum,
        "calibration_full_usable_records": full_records,
        "calibration_processed_records": processed_records,
        "thresholds": reconstruction_thresholds(),
        "worst_relative_rmse": dict(payload["worst_relative_rmse"]),
        "worst_cosine_similarity": dict(payload["worst_cosine_similarity"]),
    }
    if layers is not None and layers_sha256 is not None:
        result["layers"] = layers
        result["layers_sha256"] = layers_sha256
    return result


def _validate_stage05_selection(
    payload: Mapping[str, Any], *, require_layers: bool
) -> dict[str, Any]:
    metadata = _base_selection_metadata(payload)
    if (
        metadata["schema_version"] != 1
        or metadata["purpose"] != STAGE05_PURPOSE
        or metadata["promotion_allowed"] is not False
    ):
        raise ValueError("non-promotable selection is not the Stage-0.5 fixture")
    result = dict(metadata)
    if require_layers:
        layers, digest = _validated_layer_selection(
            payload,
            source_experts=metadata["source_experts"],
            retained_experts=metadata["retained_experts"],
        )
        result["layers"] = layers
        result["layers_sha256"] = digest
    return result


def selection_metadata_bytes(manifest_bytes: bytes) -> dict[str, Any]:
    """Validate promotion policy and layer IDs from one captured byte snapshot."""
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection manifest must contain valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("selection manifest must be a JSON object")
    base = _base_selection_metadata(payload)
    if base["promotion_allowed"]:
        return _validate_approved_selection(payload, require_layers=True)
    return _validate_stage05_selection(payload, require_layers=True)


def selection_metadata(path: str | Path) -> dict[str, Any]:
    """Read the promotion boundary embedded in a selection manifest."""
    return selection_metadata_bytes(Path(path).read_bytes())


def validate_seed_manifest_bytes(
    manifest_bytes: bytes,
    *,
    expected_sha256: str,
    init_path: str,
    phase: str,
    expected_num_experts: int | None = None,
    expected_top_k_experts: int | None = None,
) -> dict[str, Any]:
    """Validate one pinned seed sidecar and its phase/promotion policy."""
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("checkpoint.init_manifest_sha256 is not a SHA-256 digest")
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "initial seed manifest identity mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("initial seed manifest is invalid JSON") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("initial seed manifest must be a schema-1 JSON object")
    if payload.get("output") != init_path:
        raise ValueError("initial seed manifest output differs from checkpoint.init_path")
    if payload.get("saved_tree") != payload.get("target_exact_tree"):
        raise ValueError("initial seed manifest does not attest an exact target tree")
    source_revision = payload.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("initial seed manifest has no source revision")
    source_path = payload.get("source")
    if not isinstance(source_path, str):
        raise ValueError("initial seed manifest has no source checkpoint path")
    source_inventory = payload.get("source_gcs_inventory")
    source_inventory_post_load = payload.get("source_gcs_inventory_post_load")
    output_inventory = payload.get("output_gcs_inventory")
    if (
        not isinstance(source_inventory, Mapping)
        or not isinstance(source_inventory_post_load, Mapping)
        or not isinstance(output_inventory, Mapping)
    ):
        raise ValueError("initial seed manifest has no GCS object inventories")
    validated_source_inventory = validate_gcs_inventory(
        source_inventory, expected_uri=source_path
    )
    validated_source_inventory_post_load = validate_gcs_inventory(
        source_inventory_post_load, expected_uri=source_path
    )
    if validated_source_inventory_post_load != validated_source_inventory:
        raise ValueError("initial seed source inventory changed during checkpoint load")
    validate_gcs_inventory(output_inventory, expected_uri=init_path)
    if source_revision != (
        f"gcs-inventory-sha256:{validated_source_inventory['sha256']}"
    ):
        raise ValueError("initial seed source revision differs from its GCS inventory")
    if normalize_source_identity(payload.get("sunfish_source")) is None:
        raise ValueError("initial seed manifest has no valid Sunfish source identity")
    selection_sha256 = payload.get("selection_sha256")
    if not isinstance(selection_sha256, str) or not _SHA256.fullmatch(selection_sha256):
        raise ValueError("initial seed manifest has no valid selection hash")

    metadata = payload.get("selection_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("initial seed manifest has no selection metadata")
    promotion_allowed = metadata.get("promotion_allowed")
    purpose = metadata.get("purpose")
    if not isinstance(promotion_allowed, bool) or not isinstance(purpose, str):
        raise ValueError("initial seed manifest selection metadata is malformed")
    for key in ("source_experts", "retained_experts", "top_k_experts"):
        if metadata.get(key) != payload.get(key):
            raise ValueError(f"initial seed selection metadata differs for {key}")
    if expected_num_experts is not None and payload.get(
        "retained_experts"
    ) != expected_num_experts:
        raise ValueError("initial seed retained expert count differs from the model")
    if expected_top_k_experts is not None and payload.get(
        "top_k_experts"
    ) != expected_top_k_experts:
        raise ValueError("initial seed top-k differs from the model")
    if phase == "smoke":
        if promotion_allowed or purpose != STAGE05_PURPOSE:
            raise ValueError("smoke phase requires the non-promotable Stage-0.5 seed")
        attested_selection = _validate_stage05_selection(
            metadata, require_layers=True
        )
    elif not promotion_allowed:
        raise ValueError("non-promotable Stage-0.5 seed cannot initialize this phase")
    else:
        approved = _validate_approved_selection(metadata, require_layers=True)
        if (
            approved["source_revision"] != source_revision
            or normalize_source_identity(approved["sunfish_source"])
            != normalize_source_identity(payload.get("sunfish_source"))
        ):
            raise ValueError("production seed and approved selection lineage differ")
        attested_selection = approved
    layers_sha256 = payload.get("selection_layers_sha256")
    if (
        not isinstance(layers_sha256, str)
        or not _SHA256.fullmatch(layers_sha256)
        or layers_sha256 != attested_selection["layers_sha256"]
    ):
        raise ValueError("initial seed layer selection digest differs")
    pruning = payload.get("pruning")
    if (
        not isinstance(pruning, Mapping)
        or pruning.get("selection_layers_sha256") != layers_sha256
    ):
        raise ValueError("initial seed pruning is not bound to the selected expert IDs")
    return dict(payload)
