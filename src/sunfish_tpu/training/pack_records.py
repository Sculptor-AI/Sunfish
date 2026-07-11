"""Pack tokenized JSONL examples into immutable Sunfish training shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sunfish.datashards import ShardWriter, write_manifest
from sunfish_tpu.training.record_format import TrainingRecord, encode_record

_FIELDS = {
    "prompt",
    "response",
    "bucket_id",
    "prompt_loss_mask",
    "response_loss_mask",
}


def pack_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    records_per_shard: int,
    source: str,
) -> dict[str, object]:
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, object]] = []
    writer: ShardWriter | None = None
    total_records = 0
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = _record_from_json(payload)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid record on JSONL line {line_number}: {error}") from error
            if writer is None:
                writer = ShardWriter(output_dir / f"records-{len(shards):05d}")
            writer.add(encode_record(record))
            total_records += 1
            if writer.records == records_per_shard:
                shards.append(writer.close())
                writer = None
    if writer is not None:
        shards.append(writer.close())
    if not total_records:
        raise ValueError("input contains no training records")

    input_sha256 = _file_sha256(input_path)
    manifest_hash = write_manifest(
        output_dir,
        shards,
        source=source,
        input_file=input_path.name,
        input_sha256=input_sha256,
        record_envelope="sunfish-training-record-v1",
    )
    return {
        "output": str(output_dir),
        "records": total_records,
        "shards": len(shards),
        "manifest_sha256": manifest_hash,
        "input_sha256": input_sha256,
    }


def _record_from_json(payload: object) -> TrainingRecord:
    if not isinstance(payload, dict):
        raise TypeError("record must be a JSON object")
    unknown = set(payload) - _FIELDS
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    if "prompt" not in payload or "response" not in payload:
        raise ValueError("prompt and response are required")
    return TrainingRecord(
        prompt=_integer_tuple(payload["prompt"], "prompt"),
        response=_integer_tuple(payload["response"], "response"),
        bucket_id=_integer(payload.get("bucket_id", 0), "bucket_id"),
        prompt_loss_mask=_optional_bool_tuple(
            payload.get("prompt_loss_mask"), "prompt_loss_mask"
        ),
        response_loss_mask=_optional_bool_tuple(
            payload.get("response_loss_mask"), "response_loss_mask"
        ),
    )


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_bool_tuple(value: object, name: str) -> tuple[bool, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must be a list of booleans")
    return tuple(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="tokenized JSONL")
    parser.add_argument("--output", required=True, type=Path, help="immutable shard directory")
    parser.add_argument("--records-per-shard", type=int, default=8192)
    parser.add_argument("--source", required=True, help="provenance label stored in manifest")
    args = parser.parse_args()
    print(
        json.dumps(
            pack_jsonl(
                args.input,
                args.output,
                records_per_shard=args.records_per_shard,
                source=args.source,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
