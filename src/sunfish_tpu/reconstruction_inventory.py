"""Immutable inventory for Stage-1 reconstruction artifacts.

The raw tensors are large, but their small JSON manifests already contain a
SHA-256 for every tensor payload.  This module binds the exact manifest set and
those declared payload hashes to the completed calibration receipt.  The
reconstruction gate re-reads the manifest bytes and hashes the tensor payloads
while consuming them, so objects appended or relabelled after calibration
cannot silently enter the promotion sample.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PURPOSE = "stage-1-reconstruction-artifact-inventory"


def _manifest_paths(root: Any, expected_processes: int) -> list[tuple[int, Any]]:
    if expected_processes <= 0:
        raise ValueError("artifact inventory process count must be positive")
    paths: list[tuple[int, Any]] = []
    for process_index in range(expected_processes):
        host = root / f"host-{process_index:05d}"
        paths.extend(
            (process_index, path) for path in sorted(host.glob("step-*.json"))
        )
    return paths


def _normalized_entry(
    *,
    root: Any,
    process_index: int,
    path: Any,
    manifest_bytes: bytes,
    run_id: str,
    calibration_run_sha256: str,
    allowed_buckets: Collection[str],
    field_names: Collection[str],
) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid reconstruction manifest JSON at {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid reconstruction manifest schema at {path}")
    if payload.get("run_id") != run_id:
        raise ValueError(f"reconstruction manifest run ID differs at {path}")
    if payload.get("calibration_run_sha256") != calibration_run_sha256:
        raise ValueError(f"reconstruction manifest calibration lineage differs at {path}")
    if payload.get("process_index") != process_index:
        raise ValueError(f"reconstruction manifest process differs at {path}")
    artifact_id = payload.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or not _ARTIFACT_ID.fullmatch(artifact_id)
        or path.name != f"{artifact_id}.json"
    ):
        raise ValueError(f"reconstruction artifact ID differs from its path at {path}")
    bucket = payload.get("bucket")
    if bucket not in allowed_buckets:
        raise ValueError(f"unknown reconstruction bucket at {path}: {bucket!r}")
    try:
        tokens = int(payload["tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid reconstruction token count at {path}") from error
    if tokens <= 0:
        raise ValueError(f"invalid reconstruction token count at {path}")
    fields = payload.get("fields")
    if not isinstance(fields, Mapping) or set(fields) != set(field_names):
        raise ValueError(f"reconstruction fields differ at {path}")
    normalized_fields: dict[str, dict[str, Any]] = {}
    for name in sorted(field_names):
        field = fields[name]
        if not isinstance(field, Mapping):
            raise ValueError(f"reconstruction field metadata differs at {path}")
        field_path = field.get("path")
        digest = field.get("sha256")
        try:
            byte_count = int(field["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"reconstruction field byte count differs at {path}") from error
        if (
            not isinstance(field_path, str)
            or not field_path
            or (path.parent / field_path).name != field_path
            or field_path != f"{artifact_id}.{name}.bin"
            or byte_count <= 0
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError(f"reconstruction field identity differs at {path}")
        normalized_fields[name] = {
            "path": field_path,
            "bytes": byte_count,
            "sha256": digest,
        }
    relative_path = f"host-{process_index:05d}/{path.name}"
    return {
        "path": relative_path,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "process_index": process_index,
        "artifact_id": artifact_id,
        "bucket": bucket,
        "tokens": tokens,
        "fields": normalized_fields,
    }


def build_artifact_inventory(
    root: Any,
    *,
    run_id: str,
    calibration_run_sha256: str,
    expected_processes: int,
    allowed_buckets: Collection[str],
    field_names: Collection[str],
) -> dict[str, Any]:
    """Build a canonical inventory from the exact completed manifest set."""
    if not _SHA256.fullmatch(calibration_run_sha256):
        raise ValueError("artifact inventory requires the calibration-run digest")
    entries = [
        _normalized_entry(
            root=root,
            process_index=process_index,
            path=path,
            manifest_bytes=path.read_bytes(),
            run_id=run_id,
            calibration_run_sha256=calibration_run_sha256,
            allowed_buckets=allowed_buckets,
            field_names=field_names,
        )
        for process_index, path in _manifest_paths(root, expected_processes)
    ]
    if not entries:
        raise ValueError("reconstruction artifact inventory is empty")
    tokens_by_bucket = {bucket: 0 for bucket in sorted(allowed_buckets)}
    for entry in entries:
        tokens_by_bucket[entry["bucket"]] += int(entry["tokens"])
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "run_id": run_id,
        "calibration_run_sha256": calibration_run_sha256,
        "raw_artifact_prefix": str(root),
        "expected_processes": expected_processes,
        "manifest_count": len(entries),
        "total_tokens": sum(tokens_by_bucket.values()),
        "tokens_by_bucket": tokens_by_bucket,
        "manifests": entries,
    }


def validate_artifact_inventory(
    payload: Mapping[str, Any],
    *,
    root: Any,
    run_id: str,
    calibration_run_sha256: str,
    expected_processes: int,
    allowed_buckets: Collection[str],
    field_names: Collection[str],
) -> dict[str, Any]:
    """Validate canonical inventory structure and aggregate arithmetic."""
    if (
        payload.get("schema_version") != 1
        or payload.get("purpose") != PURPOSE
        or payload.get("run_id") != run_id
        or payload.get("calibration_run_sha256") != calibration_run_sha256
        or payload.get("raw_artifact_prefix") != str(root)
        or payload.get("expected_processes") != expected_processes
    ):
        raise ValueError("reconstruction artifact inventory lineage differs")
    manifests = payload.get("manifests")
    if not isinstance(manifests, Sequence) or isinstance(manifests, (str, bytes)):
        raise ValueError("reconstruction artifact inventory manifests are invalid")
    normalized_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in manifests:
        if not isinstance(raw, Mapping):
            raise ValueError("reconstruction artifact inventory entry is invalid")
        try:
            process_index = int(raw["process_index"])
            tokens = int(raw["tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("reconstruction artifact inventory counts are invalid") from error
        path = raw.get("path")
        artifact_id = raw.get("artifact_id")
        bucket = raw.get("bucket")
        manifest_sha256 = raw.get("manifest_sha256")
        expected_path = f"host-{process_index:05d}/{artifact_id}.json"
        if (
            not 0 <= process_index < expected_processes
            or tokens <= 0
            or not isinstance(path, str)
            or not isinstance(artifact_id, str)
            or not _ARTIFACT_ID.fullmatch(artifact_id)
            or path != expected_path
            or path in seen
            or bucket not in allowed_buckets
            or not isinstance(manifest_sha256, str)
            or not _SHA256.fullmatch(manifest_sha256)
        ):
            raise ValueError("reconstruction artifact inventory entry identity differs")
        fields = raw.get("fields")
        if not isinstance(fields, Mapping) or set(fields) != set(field_names):
            raise ValueError("reconstruction artifact inventory fields differ")
        for name in field_names:
            field = fields[name]
            if (
                not isinstance(field, Mapping)
                or field.get("path") != f"{artifact_id}.{name}.bin"
                or not isinstance(field.get("bytes"), int)
                or field["bytes"] <= 0
                or not isinstance(field.get("sha256"), str)
                or not _SHA256.fullmatch(field["sha256"])
            ):
                raise ValueError("reconstruction artifact inventory field differs")
        seen.add(path)
        normalized_entries.append(dict(raw))
    if [entry["path"] for entry in normalized_entries] != sorted(seen):
        raise ValueError("reconstruction artifact inventory is not canonically ordered")
    tokens_by_bucket = {bucket: 0 for bucket in sorted(allowed_buckets)}
    for entry in normalized_entries:
        tokens_by_bucket[entry["bucket"]] += int(entry["tokens"])
    if (
        payload.get("manifest_count") != len(normalized_entries)
        or payload.get("total_tokens") != sum(tokens_by_bucket.values())
        or payload.get("tokens_by_bucket") != tokens_by_bucket
    ):
        raise ValueError("reconstruction artifact inventory aggregates differ")
    return dict(payload)


def verify_live_artifact_inventory(root: Any, inventory: Mapping[str, Any]) -> None:
    """Require the current manifest set and bytes to equal the pinned inventory."""
    expected_processes = int(inventory["expected_processes"])
    expected = {
        entry["path"]: entry["manifest_sha256"]
        for entry in inventory["manifests"]
    }
    live = {
        f"host-{process_index:05d}/{path.name}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for process_index, path in _manifest_paths(root, expected_processes)
    }
    if live != expected:
        raise ValueError("live reconstruction artifact manifest inventory changed")


def paths_for_process(
    root: Any, inventory: Mapping[str, Any], process_index: int
) -> list[Any]:
    """Return the pinned manifest paths for one process in canonical order."""
    return [
        root / entry["path"]
        for entry in inventory["manifests"]
        if entry["process_index"] == process_index
    ]
