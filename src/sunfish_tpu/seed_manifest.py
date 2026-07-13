"""Dependency-free provenance checks for exact-tree Orbax seed manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sunfish_tpu.source_identity import normalize_source_identity
from sunfish_tpu.gcs_inventory import validate_gcs_inventory

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
STAGE05_PURPOSE = "stage-0.5-infrastructure-readiness-only"


def selection_metadata(path: str | Path) -> dict[str, Any]:
    """Read the promotion boundary embedded in a selection manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("selection manifest must be a JSON object")
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
        "purpose": purpose,
        "promotion_allowed": promotion_allowed,
        "selection_method": method,
        "source_experts": source_experts,
        "retained_experts": retained_experts,
        "top_k_experts": top_k_experts,
    }


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
    output_inventory = payload.get("output_gcs_inventory")
    if not isinstance(source_inventory, Mapping) or not isinstance(
        output_inventory, Mapping
    ):
        raise ValueError("initial seed manifest has no GCS object inventories")
    validated_source_inventory = validate_gcs_inventory(
        source_inventory, expected_uri=source_path
    )
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
    elif not promotion_allowed:
        raise ValueError("non-promotable Stage-0.5 seed cannot initialize this phase")
    return dict(payload)
