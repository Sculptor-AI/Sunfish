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


def compare_gcs_inventory_contents(
    source: Mapping[str, Any], staged: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a staged prefix to source names, sizes, and CRC32Cs.

    Object generations and URIs are intentionally different after a GCS copy.
    Everything that identifies the copied bytes remains exact.
    """
    source_uri = source.get("uri")
    staged_uri = staged.get("uri")
    if not isinstance(source_uri, str) or not isinstance(staged_uri, str):
        raise ValueError("source and staged inventories must declare their URIs")
    if source_uri == staged_uri:
        raise ValueError("source and staged inventory URIs must differ")
    source_valid = validate_gcs_inventory(source, expected_uri=source_uri)
    staged_valid = validate_gcs_inventory(staged, expected_uri=staged_uri)

    def content_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": item["name"],
                "size": item["size"],
                "crc32c": item["crc32c"],
            }
            for item in inventory["objects"]
        ]

    source_rows = content_rows(source_valid)
    staged_rows = content_rows(staged_valid)
    if source_rows != staged_rows:
        raise ValueError(
            "staged GCS content differs from source object names/sizes/CRC32Cs"
        )
    content_sha256 = hashlib.sha256(
        json.dumps(source_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    canonical = {
        "schema_version": 1,
        "purpose": "gcs-staged-content-match",
        "source_uri": source_valid["uri"],
        "source_inventory_sha256": source_valid["sha256"],
        "staged_uri": staged_valid["uri"],
        "staged_inventory_sha256": staged_valid["sha256"],
        "object_count": len(source_rows),
        "total_bytes": source_valid["total_bytes"],
        "content_sha256": content_sha256,
        "matched": True,
    }
    return {
        **canonical,
        "sha256": hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def probe_gcs_object_reads(
    uri: str,
    inventory: Mapping[str, Any],
    *,
    path_factory: Any | None = None,
) -> int:
    """Read one byte from every nonempty inventoried object.

    The caller should inventory again afterward and compare the inventory hash;
    the CLI does so automatically. This is a bounded IAM/range-read probe, not
    a checkpoint download.
    """
    canonical = validate_gcs_inventory(inventory, expected_uri=uri)
    if path_factory is None:
        from etils import epath

        path_factory = epath.Path
    reads = 0
    for item in canonical["objects"]:
        if item["size"] == 0:
            continue
        path = path_factory(f"{canonical['uri']}/{item['name']}")
        try:
            with path.open("rb") as source:
                payload = source.read(1)
        except Exception as error:
            raise RuntimeError(f"failed bounded read probe for {path}: {error}") from error
        if len(payload) != 1:
            raise RuntimeError(f"short bounded read probe for {path}")
        reads += 1
    return reads


def _write_immutable_json(path_string: str, payload: Mapping[str, Any]) -> None:
    from etils import epath

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = epath.Path(path_string)
    if output.exists() and output.read_text() != encoded:
        raise FileExistsError(f"immutable inventory artifact changed at {output}")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--output", required=True, help="immutable local or gs:// JSON")
    parser.add_argument("--anonymous", action="store_true")
    parser.add_argument(
        "--expected-sha256",
        help="fail unless the live generation-bound inventory has this SHA-256",
    )
    parser.add_argument(
        "--probe-readable",
        action="store_true",
        help="range-read one byte from every nonempty object and re-inventory",
    )
    parser.add_argument(
        "--match-content-of",
        help="source inventory whose names/sizes/CRC32Cs must match",
    )
    parser.add_argument(
        "--match-output",
        help="immutable staged-content match receipt (requires --match-content-of)",
    )
    args = parser.parse_args(argv)
    try:
        if bool(args.match_content_of) != bool(args.match_output):
            raise ValueError(
                "--match-content-of and --match-output must be supplied together"
            )
        payload = build_gcs_inventory(args.uri, anonymous=args.anonymous)
        if args.expected_sha256 is not None:
            if not _SHA256.fullmatch(args.expected_sha256):
                raise ValueError("--expected-sha256 must be 64 lowercase hex digits")
            if payload["sha256"] != args.expected_sha256:
                raise RuntimeError(
                    "live GCS inventory differs from --expected-sha256: "
                    f"{payload['sha256']} != {args.expected_sha256}"
                )
        if args.probe_readable:
            probe_gcs_object_reads(args.uri, payload)
            after_probe = build_gcs_inventory(args.uri, anonymous=args.anonymous)
            if after_probe["sha256"] != payload["sha256"]:
                raise RuntimeError("GCS inventory changed during bounded read probe")
            payload = after_probe
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.match_content_of:
            from etils import epath

            source_inventory = json.loads(epath.Path(args.match_content_of).read_text())
            receipt = compare_gcs_inventory_contents(source_inventory, payload)
            _write_immutable_json(args.match_output, receipt)
        _write_immutable_json(args.output, payload)
    except (
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"sunfish-gcs-inventory: {error}", file=sys.stderr)
        return 2
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
