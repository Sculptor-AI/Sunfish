"""Immutable GCS directory inventory using object generations and CRC32C."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_GCS = re.compile(r"^gs://([^/]+)/(.+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _location(uri: str) -> tuple[str, str, str]:
    match = _GCS.fullmatch(uri.rstrip("/"))
    if match is None:
        raise ValueError("inventory URI must be a non-root gs://bucket/prefix")
    bucket, prefix = match.groups()
    prefix = prefix.rstrip("/") + "/"
    return bucket, prefix, f"gs://{bucket}/{prefix.rstrip('/')}"


def _finalize_inventory(
    uri: str, objects: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    bucket, prefix, normalized_uri = _location(uri)
    rows = []
    seen = set()
    for raw in objects:
        name = raw.get("name")
        generation = raw.get("generation")
        size = raw.get("size")
        crc32c = raw.get("crc32c")
        if not isinstance(name, str) or not name or name.startswith("/"):
            raise ValueError("inventory object names must be relative nonempty paths")
        if name in seen:
            raise ValueError(f"duplicate inventory object {name}")
        seen.add(name)
        if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
            raise ValueError(f"inventory object {name} has invalid generation")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"inventory object {name} has invalid size")
        if not isinstance(crc32c, str) or not crc32c:
            raise ValueError(f"inventory object {name} has no CRC32C")
        rows.append(
            {
                "name": name,
                "generation": generation,
                "size": size,
                "crc32c": crc32c,
            }
        )
    rows.sort(key=lambda item: item["name"])
    if not rows:
        raise ValueError(f"GCS inventory is empty: {normalized_uri}")
    canonical = {
        "schema_version": 1,
        "purpose": "gcs-generation-crc32c-inventory",
        "uri": normalized_uri,
        "bucket": bucket,
        "prefix": prefix,
        "objects": rows,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **canonical,
        "object_count": len(rows),
        "total_bytes": sum(item["size"] for item in rows),
        "sha256": digest,
    }


def validate_gcs_inventory(
    payload: Mapping[str, Any], *, expected_uri: str
) -> dict[str, Any]:
    """Validate and canonicalize a persisted inventory without network I/O."""
    if payload.get("schema_version") != 1 or payload.get("purpose") != (
        "gcs-generation-crc32c-inventory"
    ):
        raise ValueError("unsupported GCS inventory schema/purpose")
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("GCS inventory objects must be a list")
    canonical = _finalize_inventory(expected_uri, objects)
    for key in (
        "uri",
        "bucket",
        "prefix",
        "object_count",
        "total_bytes",
        "sha256",
    ):
        if payload.get(key) != canonical[key]:
            raise ValueError(f"GCS inventory differs for {key}")
    if not _SHA256.fullmatch(str(payload.get("sha256", ""))):
        raise ValueError("GCS inventory SHA-256 is invalid")
    return canonical


def gcs_inventory_from_objects(
    uri: str, objects: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build a canonical inventory from already-fetched object metadata."""
    return _finalize_inventory(uri, objects)


def build_gcs_inventory(
    uri: str, *, anonymous: bool = False, client: Any | None = None
) -> dict[str, Any]:
    """List one directory prefix and bind every object's generation/CRC."""
    bucket, prefix, normalized_uri = _location(uri)
    if client is None:
        from google.cloud import storage

        client = (
            storage.Client.create_anonymous_client()
            if anonymous
            else storage.Client()
        )
    try:
        blobs = list(client.list_blobs(bucket, prefix=prefix))
    except Exception as error:
        raise RuntimeError(
            f"failed to list GCS inventory {normalized_uri}: {error}"
        ) from error
    rows = []
    for blob in blobs:
        if blob.name == prefix and int(blob.size or 0) == 0:
            continue
        if not blob.name.startswith(prefix):
            raise RuntimeError(f"GCS returned an object outside {normalized_uri}")
        if not isinstance(blob.crc32c, str) or not blob.crc32c:
            raise RuntimeError(f"GCS object has no CRC32C: {blob.name}")
        rows.append(
            {
                "name": blob.name[len(prefix) :],
                "generation": int(blob.generation),
                "size": int(blob.size),
                "crc32c": blob.crc32c,
            }
        )
    return _finalize_inventory(normalized_uri, rows)


def verify_live_gcs_inventory(
    uri: str, expected: Mapping[str, Any], *, anonymous: bool = False
) -> dict[str, Any]:
    expected_valid = validate_gcs_inventory(expected, expected_uri=uri)
    actual = build_gcs_inventory(uri, anonymous=anonymous)
    if actual["sha256"] != expected_valid["sha256"]:
        raise RuntimeError(
            f"GCS object inventory changed at {uri}: "
            f"{actual['sha256']} != {expected_valid['sha256']}"
        )
    return actual


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--output", required=True, help="immutable local or gs:// JSON")
    parser.add_argument("--anonymous", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = build_gcs_inventory(args.uri, anonymous=args.anonymous)
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        from etils import epath

        output = epath.Path(args.output)
        if output.exists() and output.read_text() != encoded:
            raise FileExistsError(f"immutable inventory changed at {output}")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded)
    except (FileExistsError, RuntimeError, TypeError, ValueError) as error:
        print(f"sunfish-gcs-inventory: {error}", file=sys.stderr)
        return 2
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
