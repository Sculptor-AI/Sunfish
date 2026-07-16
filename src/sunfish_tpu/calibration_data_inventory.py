"""Generation-bound inventory for the exact Stage-1 corpus shard objects.

The calibration manifest already records SHA-256 digests for every ``.bin``
and ``.idx`` file, but hashing the full corpus on every preflight is wasteful.
This inventory binds the same exact artifact set to GCS generations, sizes,
and CRC32Cs.  The runner re-lists that bounded set before JAX initialization,
then the data source verifies the manifest SHA-256 values while opening it.

``manifest.json`` and the reviewed receipt are deliberately not inventoried
here: their byte hashes are pinned separately, and excluding them avoids a
hash cycle while the receipt and manifest bind this inventory's digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from sunfish_tpu.gcs_inventory import (
    build_gcs_inventory,
    gcs_inventory_from_objects,
)

PURPOSE = "stage-1-calibration-corpus-artifact-inventory"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_BYTES = 4
_OFFSET_BYTES = 8


def _artifact_specs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("calibration corpus manifest has no shards")
    specs: dict[str, dict[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("calibration corpus manifest has an invalid shard")
        records = shard.get("records")
        tokens = shard.get("tokens")
        bucket = shard.get("bucket")
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("calibration corpus shard has no bucket")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            raise ValueError("calibration corpus shard counts are invalid")
        for kind, size, digest_key in (
            ("bin", tokens * _TOKEN_BYTES, "sha256_bin"),
            ("idx", records * _OFFSET_BYTES, "sha256_idx"),
        ):
            name = shard.get(kind)
            digest = shard.get(digest_key)
            if (
                not isinstance(name, str)
                or not name
                or name.startswith("/")
                or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
            ):
                raise ValueError(
                    f"calibration corpus shard has an unsafe {kind} artifact name"
                )
            if name in specs:
                raise ValueError(f"duplicate calibration corpus artifact {name}")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError(
                    f"calibration corpus artifact {name} has no exact SHA-256"
                )
            specs[name] = {
                "bucket": bucket,
                "kind": kind,
                "size": size,
                "sha256": digest,
            }
    return specs


def _finalize_inventory(
    directory: str,
    manifest: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonicalize GCS metadata together with manifest-declared SHA-256s."""
    generic = gcs_inventory_from_objects(directory, objects)
    specs = _artifact_specs(manifest)
    by_name = {item["name"]: item for item in generic["objects"]}
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        extra = sorted(set(by_name) - set(specs))
        raise ValueError(
            "calibration corpus inventory artifact set differs: "
            f"missing={missing}, extra={extra}"
        )
    artifacts = []
    for name in sorted(specs):
        metadata = by_name[name]
        spec = specs[name]
        if metadata["size"] != spec["size"]:
            raise ValueError(
                f"calibration corpus inventory size differs for {name}: "
                f"{metadata['size']} != {spec['size']}"
            )
        artifacts.append(
            {
                "bucket": spec["bucket"],
                "kind": spec["kind"],
                "name": name,
                "generation": metadata["generation"],
                "size": metadata["size"],
                "crc32c": metadata["crc32c"],
                "sha256": spec["sha256"],
            }
        )
    return _canonicalize_inventory_payload(generic["uri"], artifacts)


def _canonicalize_inventory_payload(
    directory: str, artifacts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    metadata = [
        {
            "name": item.get("name"),
            "generation": item.get("generation"),
            "size": item.get("size"),
            "crc32c": item.get("crc32c"),
        }
        for item in artifacts
    ]
    generic = gcs_inventory_from_objects(directory, metadata)
    raw_by_name: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        name = item.get("name")
        if not isinstance(name, str) or name in raw_by_name:
            raise ValueError("calibration corpus inventory artifact names are invalid")
        raw_by_name[name] = item
    rows = []
    for metadata_item in generic["objects"]:
        raw = raw_by_name[metadata_item["name"]]
        bucket = raw.get("bucket")
        kind = raw.get("kind")
        declared_sha256 = raw.get("sha256")
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("calibration corpus inventory artifact has no bucket")
        if kind not in {"bin", "idx"}:
            raise ValueError("calibration corpus inventory artifact kind is invalid")
        if not isinstance(declared_sha256, str) or not _SHA256.fullmatch(
            declared_sha256
        ):
            raise ValueError(
                "calibration corpus inventory artifact has no declared SHA-256"
            )
        rows.append(
            {
                "bucket": bucket,
                "kind": kind,
                "name": metadata_item["name"],
                "generation": metadata_item["generation"],
                "size": metadata_item["size"],
                "crc32c": metadata_item["crc32c"],
                "sha256": declared_sha256,
            }
        )
    canonical = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "uri": generic["uri"],
        "artifacts": rows,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **canonical,
        "artifact_count": len(rows),
        "total_bytes": sum(item["size"] for item in rows),
        "sha256": digest,
    }


def validate_calibration_data_inventory_payload(
    inventory: Mapping[str, Any], *, expected_directory: str | None = None
) -> dict[str, Any]:
    """Validate a persisted inventory without needing the source manifest."""
    if inventory.get("schema_version") != 1 or inventory.get("purpose") != PURPOSE:
        raise ValueError("unsupported calibration corpus inventory schema/purpose")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("calibration corpus inventory artifacts must be a list")
    if any(not isinstance(item, Mapping) for item in artifacts):
        raise ValueError("calibration corpus inventory has an invalid artifact")
    directory = (
        expected_directory if expected_directory is not None else inventory.get("uri")
    )
    if not isinstance(directory, str):
        raise ValueError("calibration corpus inventory has no GCS directory")
    canonical = _canonicalize_inventory_payload(directory, artifacts)
    if dict(inventory) != canonical:
        raise ValueError("calibration corpus inventory is not canonical")
    return canonical


def validate_calibration_data_inventory(
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    expected_directory: str | None = None,
) -> dict[str, Any]:
    """Validate a canonical GCS inventory against every manifest artifact."""
    directory = (
        expected_directory if expected_directory is not None else inventory.get("uri")
    )
    canonical = validate_calibration_data_inventory_payload(
        inventory, expected_directory=directory if isinstance(directory, str) else None
    )
    expected = _artifact_specs(manifest)
    actual = {item["name"]: item for item in canonical["artifacts"]}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "calibration corpus inventory artifact set differs: "
            f"missing={missing}, extra={extra}"
        )
    for name, spec in expected.items():
        artifact = actual[name]
        if any(
            artifact[key] != spec[key]
            for key in ("bucket", "kind", "size", "sha256")
        ):
            raise ValueError(
                f"calibration corpus inventory manifest row differs for {name}"
            )
    return canonical


def build_calibration_data_inventory(
    directory: str,
    manifest: Mapping[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """List GCS once and retain only the manifest's exact ``bin/idx`` set."""
    specs = _artifact_specs(manifest)
    full = build_gcs_inventory(directory, client=client)
    by_name = {item["name"]: item for item in full["objects"]}
    missing = sorted(set(specs) - set(by_name))
    if missing:
        raise ValueError(
            f"calibration corpus GCS artifacts are missing: {missing}"
        )
    selected = [by_name[name] for name in sorted(specs)]
    return _finalize_inventory(directory, manifest, selected)


def calibration_data_inventory_from_objects(
    directory: str,
    manifest: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a canonical corpus inventory from already-fetched GCS metadata."""
    return _finalize_inventory(directory, manifest, objects)


def verify_live_calibration_data_inventory(
    directory: str,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Fail unless current GCS generations/CRCs equal the reviewed inventory."""
    expected_valid = validate_calibration_data_inventory(
        manifest, expected, expected_directory=directory
    )
    actual = build_calibration_data_inventory(directory, manifest, client=client)
    if actual["sha256"] != expected_valid["sha256"]:
        raise RuntimeError(
            "calibration corpus GCS artifact inventory changed at "
            f"{directory}: {actual['sha256']} != {expected_valid['sha256']}"
        )
    return actual


def _write_immutable_json(path_string: str, payload: Mapping[str, Any]) -> None:
    from etils import epath

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = epath.Path(path_string)
    if output.exists() and output.read_text() != encoded:
        raise FileExistsError(
            f"immutable calibration corpus inventory changed at {output}"
        )
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if not _SHA256.fullmatch(args.manifest_sha256):
            raise ValueError("--manifest-sha256 must be 64 lowercase hex digits")
        from etils import epath

        manifest_bytes = (
            epath.Path(args.data_directory) / "manifest.json"
        ).read_bytes()
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_sha256 != args.manifest_sha256:
            raise ValueError(
                "calibration manifest bytes differ: "
                f"{actual_manifest_sha256} != {args.manifest_sha256}"
            )
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, Mapping):
            raise ValueError("calibration manifest must be a JSON object")
        payload = build_calibration_data_inventory(args.data_directory, manifest)
        _write_immutable_json(args.output, payload)
    except (
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"sunfish-stage1-data-inventory: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
