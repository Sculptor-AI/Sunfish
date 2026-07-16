"""Dependency-free, non-overridable Stage-1 promotion contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from sunfish.calibration_records import (
    CALIBRATION_BUCKET_PERCENT,
    calibration_bucket_token_caps,
)
from sunfish_tpu.calibration_data_inventory import (
    validate_calibration_data_inventory,
)

MIN_CALIBRATION_INPUT_TOKENS = 75_000_000
MASS_COVERAGE_32 = 0.225
MASS_COVERAGE_48 = 0.3375
MASS_COVERAGE_BY_EXPERTS = {32: MASS_COVERAGE_32, 48: MASS_COVERAGE_48}
RECONSTRUCTION_SAMPLE_TOKENS = 100_000
RECONSTRUCTION_MIN_TOKENS_PER_BUCKET = 4_000
RECONSTRUCTION_MAX_RELATIVE_RMSE = 0.15
RECONSTRUCTION_MIN_COSINE_SIMILARITY = 0.99
APPROVED_SELECTION_PURPOSE = "stage-1-router-selection-approved"
APPROVED_SELECTION_METHOD = (
    "coverage-constrained-router-mass-plus-layer-output-reconstruction"
)
CALIBRATION_SOURCE_RECEIPT_PURPOSE = "stage-1-calibration-source-receipt"
CALIBRATION_SOURCE_APPROVAL_SCOPE = "stage-1-router-calibration-promotion"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION = re.compile(
    r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|gcs-inventory-sha256:[0-9a-f]{64})$"
)
_REVIEWED_PROFILE = re.compile(r"^reviewed-stage1-[a-z0-9][a-z0-9._-]*$")


def _nonempty_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"calibration source receipt {field} must be a nonempty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"calibration source receipt {field} has an invalid entry")
    return result


def validate_calibration_source_contract(
    manifest: Mapping[str, object],
    receipt_bytes: bytes,
    *,
    expected_receipt_sha256: str,
) -> dict[str, object]:
    """Validate reviewed calibration provenance before any JAX backend access."""
    if not _SHA256.fullmatch(expected_receipt_sha256):
        raise ValueError("calibration source receipt SHA-256 is invalid")
    actual_receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise ValueError(
            "calibration source receipt bytes differ: "
            f"{actual_receipt_sha256} != {expected_receipt_sha256}"
        )
    try:
        receipt = json.loads(receipt_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("calibration source receipt is invalid JSON") from error
    if not isinstance(receipt, Mapping):
        raise ValueError("calibration source receipt must be a JSON object")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("purpose") != CALIBRATION_SOURCE_RECEIPT_PURPOSE
        or receipt.get("approval_scope") != CALIBRATION_SOURCE_APPROVAL_SCOPE
        or receipt.get("review_status") != "approved"
        or receipt.get("promotion_allowed") is not True
    ):
        raise ValueError("calibration source receipt is not an approved promotion receipt")
    for field in ("reviewed_by", "reviewed_at_utc", "approval_reference"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"calibration source receipt lacks {field}")
    source_profile = receipt.get("source_profile")
    if not isinstance(source_profile, str) or not _REVIEWED_PROFILE.fullmatch(
        source_profile
    ):
        raise ValueError("calibration source receipt has no reviewed source profile")

    raw_corpus_inventory = receipt.get("corpus_gcs_inventory")
    if not isinstance(raw_corpus_inventory, Mapping):
        raise ValueError(
            "calibration source receipt has no corpus GCS artifact inventory"
        )
    corpus_inventory = validate_calibration_data_inventory(
        manifest, raw_corpus_inventory
    )
    if manifest.get("corpus_gcs_inventory_sha256") != corpus_inventory["sha256"]:
        raise ValueError(
            "calibration manifest does not bind the corpus GCS artifact inventory"
        )

    expected_buckets = set(CALIBRATION_BUCKET_PERCENT)
    bucket_sources = receipt.get("bucket_sources")
    if not isinstance(bucket_sources, Mapping) or set(bucket_sources) != expected_buckets:
        raise ValueError("calibration source receipt bucket set is not canonical")
    normalized_sources: dict[str, dict[str, object]] = {}
    for bucket, percent in CALIBRATION_BUCKET_PERCENT.items():
        source = bucket_sources[bucket]
        if not isinstance(source, Mapping):
            raise ValueError(f"calibration source receipt bucket {bucket} is invalid")
        source_id = source.get("source_id")
        revision = source.get("source_revision")
        license_policy = source.get("license_policy")
        token_share = source.get("token_share")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"calibration source receipt bucket {bucket} lacks source_id")
        if not isinstance(revision, str) or not _IMMUTABLE_REVISION.fullmatch(revision):
            raise ValueError(
                f"calibration source receipt bucket {bucket} lacks an immutable revision"
            )
        if not isinstance(license_policy, str) or not license_policy.strip():
            raise ValueError(
                f"calibration source receipt bucket {bucket} lacks license policy"
            )
        if isinstance(token_share, bool) or not isinstance(token_share, (int, float)):
            raise ValueError(f"calibration source receipt bucket {bucket} lacks token share")
        if float(token_share) != percent / 100.0:
            raise ValueError(
                f"calibration source receipt bucket {bucket} changed token share"
            )
        normalized_sources[bucket] = {
            "source_id": source_id,
            "source_revision": revision,
            "license_policy": license_policy,
            "filters": _nonempty_strings(
                source.get("filters"), field=f"bucket_sources.{bucket}.filters"
            ),
            "decontamination": _nonempty_strings(
                source.get("decontamination"),
                field=f"bucket_sources.{bucket}.decontamination",
            ),
            "token_share": float(token_share),
        }

    if manifest.get("promotion_allowed") is not True:
        raise ValueError("calibration manifest is explicitly non-promotable")
    if manifest.get("source_profile") != source_profile:
        raise ValueError("calibration manifest source profile differs from receipt")
    if manifest.get("source_receipt_sha256") != expected_receipt_sha256:
        raise ValueError("calibration manifest does not bind the source receipt")
    if manifest.get("failures") not in ([], ()):
        raise ValueError("calibration manifest records failed source buckets")
    if manifest.get("manifest_version") != 1:
        raise ValueError("calibration manifest version is unsupported")
    total_tokens = manifest.get("total_tokens")
    total_records = manifest.get("total_records")
    record_tokens = manifest.get("record_tokens")
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < MIN_CALIBRATION_INPUT_TOKENS
    ):
        raise ValueError(
            "calibration manifest does not contain the precommitted 75M tokens"
        )
    if (
        isinstance(total_records, bool)
        or not isinstance(total_records, int)
        or total_records <= 0
    ):
        raise ValueError("calibration manifest total_records must be positive")
    if (
        isinstance(record_tokens, bool)
        or not isinstance(record_tokens, int)
        or record_tokens < 2
    ):
        raise ValueError("calibration manifest record_tokens must be at least two")
    expected_caps = calibration_bucket_token_caps(total_tokens)
    if manifest.get("bucket_token_caps") != expected_caps:
        raise ValueError("calibration manifest bucket token caps are not canonical")
    expected_shares = {
        bucket: percent / 100.0
        for bucket, percent in CALIBRATION_BUCKET_PERCENT.items()
    }
    if manifest.get("bucket_token_shares") != expected_shares:
        raise ValueError("calibration manifest token shares are not canonical")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(expected_buckets):
        raise ValueError("calibration manifest must contain one shard per source bucket")
    seen = set()
    observed_tokens = 0
    observed_records = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("calibration manifest has an invalid shard")
        bucket = shard.get("bucket")
        if bucket not in expected_buckets or bucket in seen:
            raise ValueError("calibration manifest has duplicate/unknown source buckets")
        seen.add(bucket)
        approved = normalized_sources[bucket]
        if (
            shard.get("source_id") != approved["source_id"]
            or shard.get("source_revision") != approved["source_revision"]
        ):
            raise ValueError(
                f"calibration manifest bucket {bucket} differs from approved source"
            )
        records = shard.get("records")
        tokens = shard.get("tokens")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            raise ValueError(
                f"calibration manifest bucket {bucket} has nonpositive records/tokens"
            )
        if shard.get("record_tokens") != record_tokens:
            raise ValueError(
                f"calibration manifest bucket {bucket} changed record_tokens"
            )
        if not 2 * records <= tokens <= record_tokens * records:
            raise ValueError(
                f"calibration manifest bucket {bucket} has impossible record lengths"
            )
        if tokens != expected_caps[bucket]:
            raise ValueError(
                f"calibration manifest bucket {bucket} does not meet its token cap"
            )
        observed_records += records
        observed_tokens += tokens
    if seen != expected_buckets:
        raise ValueError("calibration manifest is missing approved source buckets")
    if observed_tokens != total_tokens:
        raise ValueError("calibration manifest shard tokens differ from total_tokens")
    if observed_records != total_records:
        raise ValueError("calibration manifest shard records differ from total_records")
    return {
        "schema_version": 2,
        "promotion_allowed": True,
        "source_profile": source_profile,
        "source_receipt_sha256": expected_receipt_sha256,
        "corpus_gcs_inventory": corpus_inventory,
        "corpus_gcs_inventory_sha256": corpus_inventory["sha256"],
        "bucket_sources": normalized_sources,
        "reviewed_by": receipt["reviewed_by"],
        "reviewed_at_utc": receipt["reviewed_at_utc"],
        "approval_reference": receipt["approval_reference"],
        "total_tokens": total_tokens,
        "total_records": total_records,
        "record_tokens": record_tokens,
        "bucket_token_caps": expected_caps,
    }


def reconstruction_thresholds() -> dict[str, int | float]:
    """Return a fresh canonical threshold mapping for evidence comparison."""
    return {
        "max_relative_rmse": RECONSTRUCTION_MAX_RELATIVE_RMSE,
        "min_cosine_similarity": RECONSTRUCTION_MIN_COSINE_SIMILARITY,
        "min_total_tokens": RECONSTRUCTION_SAMPLE_TOKENS,
        "min_tokens_per_bucket": RECONSTRUCTION_MIN_TOKENS_PER_BUCKET,
    }
