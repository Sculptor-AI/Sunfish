"""Streaming DiffusionGemma text-only and expert-pruning converter.

The converter operates directly on the safetensors byte format. Untouched
tensors are copied byte-for-byte and selected expert rows are copied as
contiguous byte ranges, so the 51+ GB source checkpoint is never loaded into
RAM. Output shards retain their source names and a fresh index is generated.

The first required use is a 128-expert/Top-8 text-only control. Only after that
control reproduces upstream logits and generations should a per-layer expert
selection manifest be supplied for a smaller checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sunfish.checkpoint_audit import DTYPE_BITS
from sunfish.source_tree import workspace_source_identity

_HEADER_LENGTH_BYTES = 8
_MAX_HEADER_BYTES = 100_000_000
_INDEX_NAME = "model.safetensors.index.json"
_MANIFEST_NAME = "sunfish_conversion.json"
_EXPERT_TENSOR = re.compile(
    r"^model\.decoder\.layers\.(\d+)\."
    r"(experts\.(?:down_proj|gate_up_proj)|"
    r"router\.(?:per_expert_scale|proj\.weight))$"
)
_VISION_PREFIXES = (
    "model.encoder.embed_vision.",
    "model.encoder.vision_tower.",
)
_EXPECTED_PRUNABLE_SUFFIXES = frozenset(
    {
        "experts.down_proj",
        "experts.gate_up_proj",
        "router.per_expert_scale",
        "router.proj.weight",
    }
)


@dataclass(frozen=True)
class RawTensor:
    """One source tensor and its location in a safetensors data buffer."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TensorCopy:
    """Copy plan for one tensor; rows=None means a byte-identical full copy."""

    source: RawTensor
    output_shape: tuple[int, ...]
    rows: tuple[int, ...] | None = None

    @property
    def output_nbytes(self) -> int:
        if self.rows is None:
            return self.source.nbytes
        return self.row_nbytes * len(self.rows)

    @property
    def row_nbytes(self) -> int:
        if not self.source.shape or self.source.shape[0] <= 0:
            raise ValueError(f"cannot slice scalar tensor {self.source.name!r}")
        if self.source.nbytes % self.source.shape[0]:
            raise ValueError(f"tensor {self.source.name!r} has packed/non-row-aligned data")
        return self.source.nbytes // self.source.shape[0]


@dataclass(frozen=True)
class ShardPlan:
    source_path: Path
    tensors: tuple[TensorCopy, ...]
    metadata: object | None


@dataclass(frozen=True)
class ConversionPlan:
    config: dict[str, object]
    source_index: dict[str, object]
    shards: tuple[ShardPlan, ...]
    selection: dict[int, tuple[int, ...]]
    source_experts: int
    retained_experts: int
    top_k: int
    text_only: bool
    dropped_tensors: tuple[str, ...]

    @property
    def output_tensors(self) -> int:
        return sum(len(shard.tensors) for shard in self.shards)

    @property
    def output_parameters(self) -> int:
        return sum(
            _numel(tensor.output_shape)
            for shard in self.shards
            for tensor in shard.tensors
        )

    @property
    def output_bytes(self) -> int:
        return sum(
            tensor.output_nbytes
            for shard in self.shards
            for tensor in shard.tensors
        )


def _numel(shape: Sequence[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _read_raw_header(path: Path) -> tuple[list[RawTensor], object | None, int]:
    """Read and validate a safetensors header, returning its data start."""
    with path.open("rb") as handle:
        prefix = handle.read(_HEADER_LENGTH_BYTES)
        if len(prefix) != _HEADER_LENGTH_BYTES:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", prefix)
        if not 0 < header_length <= _MAX_HEADER_BYTES:
            raise ValueError(f"{path} has an implausible header length {header_length}")
        payload = handle.read(header_length)
        if len(payload) != header_length:
            raise ValueError(f"{path} has a truncated safetensors header")
        header = json.loads(payload)

    tensors: list[RawTensor] = []
    metadata = header.get("__metadata__")
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = str(entry["dtype"])
        shape = tuple(int(dim) for dim in entry["shape"])
        offsets = entry["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"tensor {name!r} has invalid data_offsets")
        start, end = (int(offsets[0]), int(offsets[1]))
        if start < 0 or end < start:
            raise ValueError(f"tensor {name!r} has invalid byte range {offsets!r}")
        if dtype not in DTYPE_BITS:
            raise ValueError(f"tensor {name!r} uses unsupported dtype {dtype!r}")
        expected_bits = _numel(shape) * DTYPE_BITS[dtype]
        if expected_bits % 8 or end - start != expected_bits // 8:
            raise ValueError(f"tensor {name!r} byte range does not match dtype and shape")
        tensors.append(RawTensor(name, dtype, shape, start, end))

    data_start = _HEADER_LENGTH_BYTES + header_length
    required_size = data_start + max((tensor.end for tensor in tensors), default=0)
    if path.stat().st_size < required_size:
        raise ValueError(f"{path} has truncated tensor data")
    return tensors, metadata, data_start


def _load_json_object(path: Path) -> dict[str, object]:
    return _load_json_object_bytes(path.read_bytes(), source=str(path))


def _load_json_object_bytes(payload: bytes, *, source: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def load_selection_manifest(
    path: Path,
    *,
    source_experts: int,
    retained_experts: int,
    top_k_experts: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Load one selection path exactly once, then validate its layer IDs."""
    return load_selection_manifest_bytes(
        path.read_bytes(),
        source_experts=source_experts,
        retained_experts=retained_experts,
        top_k_experts=top_k_experts,
        source=str(path),
    )


def load_selection_manifest_bytes(
    manifest_bytes: bytes,
    *,
    source_experts: int,
    retained_experts: int,
    top_k_experts: int | None = None,
    source: str = "selection manifest bytes",
) -> dict[int, tuple[int, ...]]:
    """Validate layer IDs from an already captured immutable byte snapshot."""
    payload = _load_json_object_bytes(manifest_bytes, source=source)
    if "source_experts" in payload and int(payload["source_experts"]) != source_experts:
        raise ValueError("selection source_experts does not match the checkpoint")
    if "retained_experts" in payload and int(payload["retained_experts"]) != retained_experts:
        raise ValueError("selection retained_experts does not match the conversion")
    if (
        top_k_experts is not None
        and "top_k_experts" in payload
        and int(payload["top_k_experts"]) != top_k_experts
    ):
        raise ValueError("selection top_k_experts does not match the conversion")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("selection must contain a layers object")

    selection: dict[int, tuple[int, ...]] = {}
    for raw_layer, raw_experts in raw_layers.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid layer id {raw_layer!r}") from error
        if layer < 0 or not isinstance(raw_experts, list):
            raise ValueError(f"invalid selection for layer {raw_layer!r}")
        experts = tuple(int(expert) for expert in raw_experts)
        if len(experts) != retained_experts:
            raise ValueError(
                f"layer {layer} selects {len(experts)} experts; expected {retained_experts}"
            )
        if tuple(sorted(set(experts))) != experts:
            raise ValueError(f"layer {layer} expert ids must be unique and increasing")
        if any(expert < 0 or expert >= source_experts for expert in experts):
            raise ValueError(f"layer {layer} selects an out-of-range expert")
        selection[layer] = experts
    return selection


# Kept as a private alias for callers/tests written before the JAX seed
# materializer shared this exact manifest contract.
_selection_from_file = load_selection_manifest


def _source_shards(source: Path, source_index: Mapping[str, object]) -> list[Path]:
    raw_weight_map = source_index.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError(f"{source / _INDEX_NAME} has no weight_map")
    names = sorted({str(shard) for shard in raw_weight_map.values()})
    paths = [source / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing shards: {', '.join(missing)}")
    return paths


def build_plan(
    source: Path,
    *,
    retained_experts: int,
    top_k: int,
    selection_path: Path | None = None,
    text_only: bool = True,
) -> ConversionPlan:
    """Validate a source checkpoint and construct a header-only copy plan."""
    config_path = source / "config.json"
    index_path = source / _INDEX_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")
    config = _load_json_object(config_path)
    source_index = _load_json_object(index_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config.json has no text_config object")
    source_experts = int(text_config.get("num_experts", 0))
    if source_experts <= 0:
        raise ValueError("text_config.num_experts must be positive")
    if not 0 < retained_experts <= source_experts:
        raise ValueError("retained_experts must be in [1, source_experts]")
    if not 0 < top_k <= retained_experts:
        raise ValueError("top_k must be in [1, retained_experts]")
    if retained_experts < source_experts and selection_path is None:
        raise ValueError("a per-layer selection manifest is required when pruning experts")

    selection = (
        load_selection_manifest(
            selection_path,
            source_experts=source_experts,
            retained_experts=retained_experts,
            top_k_experts=top_k,
        )
        if selection_path is not None
        else {}
    )

    source_weight_map = source_index["weight_map"]
    assert isinstance(source_weight_map, dict)
    seen_names: set[str] = set()
    seen_suffixes: dict[int, set[str]] = {}
    layers: set[int] = set()
    dropped: list[str] = []
    shard_plans: list[ShardPlan] = []

    for shard_path in _source_shards(source, source_index):
        raw_tensors, metadata, _ = _read_raw_header(shard_path)
        copies: list[TensorCopy] = []
        for tensor in raw_tensors:
            if tensor.name in seen_names:
                raise ValueError(f"tensor {tensor.name!r} appears in multiple shards")
            seen_names.add(tensor.name)
            expected_shard = source_weight_map.get(tensor.name)
            if expected_shard != shard_path.name:
                raise ValueError(
                    f"index maps {tensor.name!r} to {expected_shard!r}, not {shard_path.name!r}"
                )
            if text_only and tensor.name.startswith(_VISION_PREFIXES):
                dropped.append(tensor.name)
                continue

            match = _EXPERT_TENSOR.fullmatch(tensor.name)
            if match is None:
                copies.append(TensorCopy(tensor, tensor.shape))
                continue

            layer = int(match.group(1))
            suffix = match.group(2)
            layers.add(layer)
            seen_suffixes.setdefault(layer, set()).add(suffix)
            if not tensor.shape or tensor.shape[0] != source_experts:
                raise ValueError(
                    f"{tensor.name!r} axis 0 is not the configured expert count"
                )
            rows = selection.get(layer, tuple(range(source_experts)))
            if retained_experts < source_experts and layer not in selection:
                raise ValueError(f"selection is missing decoder layer {layer}")
            if len(rows) != retained_experts:
                raise ValueError(f"layer {layer} has an invalid selection length")
            if retained_experts == source_experts and rows == tuple(range(source_experts)):
                copies.append(TensorCopy(tensor, tensor.shape))
            else:
                output_shape = (retained_experts, *tensor.shape[1:])
                copies.append(TensorCopy(tensor, output_shape, rows))
        shard_plans.append(ShardPlan(shard_path, tuple(copies), metadata))

    index_names = {str(name) for name in source_weight_map}
    if seen_names != index_names:
        missing = sorted(index_names - seen_names)
        extra = sorted(seen_names - index_names)
        raise ValueError(f"checkpoint/index tensor mismatch; missing={missing}, extra={extra}")
    for layer in sorted(layers):
        if seen_suffixes[layer] != _EXPECTED_PRUNABLE_SUFFIXES:
            missing = sorted(_EXPECTED_PRUNABLE_SUFFIXES - seen_suffixes[layer])
            raise ValueError(f"decoder layer {layer} is missing prunable tensors: {missing}")

    configured_layers = int(text_config.get("num_hidden_layers", len(layers)))
    expected_layers = set(range(configured_layers))
    if layers != expected_layers:
        raise ValueError(
            f"prunable tensor layers are {sorted(layers)}; expected {sorted(expected_layers)}"
        )
    if retained_experts < source_experts and set(selection) != expected_layers:
        raise ValueError("selection layers do not exactly match decoder layers")

    output_config = json.loads(json.dumps(config))
    output_text_config = output_config["text_config"]
    assert isinstance(output_text_config, dict)
    output_text_config["num_experts"] = retained_experts
    output_text_config["top_k_experts"] = top_k
    if text_only:
        output_config["vision_config"] = None

    return ConversionPlan(
        config=output_config,
        source_index=source_index,
        shards=tuple(shard_plans),
        selection={
            layer: selection.get(layer, tuple(range(source_experts)))
            for layer in sorted(expected_layers)
        },
        source_experts=source_experts,
        retained_experts=retained_experts,
        top_k=top_k,
        text_only=text_only,
        dropped_tensors=tuple(sorted(dropped)),
    )


def plan_report(plan: ConversionPlan) -> dict[str, object]:
    return {
        "source_experts": plan.source_experts,
        "retained_experts": plan.retained_experts,
        "top_k": plan.top_k,
        "text_only": plan.text_only,
        "output_shards": sum(bool(shard.tensors) for shard in plan.shards),
        "output_tensors": plan.output_tensors,
        "output_parameters": plan.output_parameters,
        "output_bytes": plan.output_bytes,
        "dropped_tensors": len(plan.dropped_tensors),
        "layers": len(plan.selection),
    }


def _copy_bytes(
    source: BinaryIO,
    output: BinaryIO,
    *,
    start: int,
    length: int,
    chunk_bytes: int,
) -> None:
    source.seek(start)
    remaining = length
    while remaining:
        block = source.read(min(remaining, chunk_bytes))
        if not block:
            raise ValueError("source shard ended during tensor copy")
        output.write(block)
        remaining -= len(block)


def _encode_header(shard: ShardPlan) -> bytes:
    header: dict[str, object] = {}
    if shard.metadata is not None:
        header["__metadata__"] = shard.metadata
    offset = 0
    for tensor in shard.tensors:
        end = offset + tensor.output_nbytes
        header[tensor.source.name] = {
            "dtype": tensor.source.dtype,
            "shape": list(tensor.output_shape),
            "data_offsets": [offset, end],
        }
        offset = end
    payload = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * ((-len(payload)) % 8)
    return payload


def _write_shard(shard: ShardPlan, output_path: Path, *, chunk_bytes: int) -> None:
    _, _, source_data_start = _read_raw_header(shard.source_path)
    header = _encode_header(shard)
    with shard.source_path.open("rb") as source, output_path.open("xb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        for tensor in shard.tensors:
            if tensor.rows is None:
                _copy_bytes(
                    source,
                    output,
                    start=source_data_start + tensor.source.start,
                    length=tensor.source.nbytes,
                    chunk_bytes=chunk_bytes,
                )
                continue
            for row in tensor.rows:
                _copy_bytes(
                    source,
                    output,
                    start=source_data_start + tensor.source.start + row * tensor.row_nbytes,
                    length=tensor.row_nbytes,
                    chunk_bytes=chunk_bytes,
                )


def _copy_auxiliary_files(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if path.name == _INDEX_NAME or path.suffix == ".safetensors":
            continue
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def convert(plan: ConversionPlan, output: Path, *, chunk_mib: int = 16) -> dict[str, object]:
    """Execute a validated plan into a new, previously nonexistent directory."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {output}")
    if chunk_mib <= 0:
        raise ValueError("chunk_mib must be positive")
    source_identity = workspace_source_identity(Path(__file__).resolve().parents[2])
    output.mkdir(parents=True)
    _copy_auxiliary_files(plan.shards[0].source_path.parent, output)

    weight_map: dict[str, str] = {}
    for shard in plan.shards:
        if not shard.tensors:
            continue
        _write_shard(shard, output / shard.source_path.name, chunk_bytes=chunk_mib << 20)
        for tensor in shard.tensors:
            weight_map[tensor.source.name] = shard.source_path.name

    metadata = dict(plan.source_index.get("metadata", {}))
    metadata["total_size"] = plan.output_bytes
    metadata["total_parameters"] = plan.output_parameters
    output_index = {"metadata": metadata, "weight_map": dict(sorted(weight_map.items()))}
    (output / _INDEX_NAME).write_text(json.dumps(output_index, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(plan.config, indent=2) + "\n")

    manifest = {
        "format_version": 1,
        **plan_report(plan),
        "selection": {str(layer): list(experts) for layer, experts in plan.selection.items()},
        "dropped_tensor_names": list(plan.dropped_tensors),
        "sunfish_source": source_identity,
    }
    (output / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="upstream checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--retained-experts", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--selection", type=Path, help="per-layer expert selection JSON")
    parser.add_argument(
        "--keep-vision",
        action="store_true",
        help="retain vision tensors and vision_config (text-only is the default)",
    )
    parser.add_argument("--chunk-mib", type=int, default=16, help="copy buffer size")
    parser.add_argument("--dry-run", action="store_true", help="validate headers and print plan only")
    args = parser.parse_args()

    plan = build_plan(
        args.source,
        retained_experts=args.retained_experts,
        top_k=args.top_k,
        selection_path=args.selection,
        text_only=not args.keep_vision,
    )
    if args.dry_run:
        print(json.dumps(plan_report(plan), indent=2))
        return
    print(json.dumps(convert(plan, args.output, chunk_mib=args.chunk_mib), indent=2))


if __name__ == "__main__":
    main()
