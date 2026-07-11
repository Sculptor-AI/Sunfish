"""Dependency-free safetensors checkpoint auditor.

Reads only the JSON headers of ``*.safetensors`` shards (8-byte length prefix
followed by a JSON table of name/dtype/shape/offsets), so a full 25B-parameter
checkpoint can be audited in milliseconds without loading a single weight.

This is the "recount the materialized weights" tool the README promises: it
groups tensors by configurable regex rules, recounts parameters per group, and
lets the design math in ``model_budget.py`` be checked against reality before
any conversion code is trusted.

Tensor naming upstream is not guessed at here — run with ``--list`` first,
then adjust the group rules to the real names.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_HEADER_LENGTH_BYTES = 8
_MAX_HEADER_BYTES = 100_000_000

DTYPE_BITS = {
    "F64": 64, "I64": 64, "U64": 64,
    "F32": 32, "I32": 32, "U32": 32,
    "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8, "I8": 8, "U8": 8, "BOOL": 8,
    "F4": 4, "NF4": 4,
}

# Starting-point grouping only; override with --rules once real names are known.
DEFAULT_GROUP_RULES: tuple[tuple[str, str], ...] = (
    ("vision", r"vision|image|pixel"),
    ("routed_experts", r"\bexperts?\b|\bexperts?[._]"),
    ("router", r"router|gating|gate_proj_router"),
    ("shared_expert", r"shared_expert"),
    ("embedding_or_head", r"embed|lm_head|logits"),
)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str

    @property
    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def bits(self) -> int:
        if self.dtype not in DTYPE_BITS:
            raise ValueError(f"unknown dtype {self.dtype!r} for tensor {self.name!r}")
        return DTYPE_BITS[self.dtype] * self.numel


def read_safetensors_header(path: Path) -> dict[str, TensorInfo]:
    """Parse one shard's header without reading any tensor data."""
    with path.open("rb") as handle:
        prefix = handle.read(_HEADER_LENGTH_BYTES)
        if len(prefix) != _HEADER_LENGTH_BYTES:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", prefix)
        if not 0 < header_length <= _MAX_HEADER_BYTES:
            raise ValueError(f"{path} has an implausible header length {header_length}")
        header = json.loads(handle.read(header_length))

    tensors: dict[str, TensorInfo] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = TensorInfo(
            name=name,
            dtype=entry["dtype"],
            shape=tuple(int(dim) for dim in entry["shape"]),
            shard=path.name,
        )
    return tensors


def read_checkpoint(directory: Path) -> dict[str, TensorInfo]:
    """Merge headers from every shard in a checkpoint directory."""
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors shards under {directory}")
    tensors: dict[str, TensorInfo] = {}
    for shard in shards:
        for name, info in read_safetensors_header(shard).items():
            if name in tensors:
                raise ValueError(f"tensor {name!r} appears in multiple shards")
            tensors[name] = info
    return tensors


def classify(name: str, rules: Sequence[tuple[str, str]]) -> str:
    for group, pattern in rules:
        if re.search(pattern, name):
            return group
    return "other"


def audit(
    tensors: Mapping[str, TensorInfo],
    *,
    rules: Sequence[tuple[str, str]] = DEFAULT_GROUP_RULES,
) -> dict[str, object]:
    """Group tensors and recount parameters; totals come from real shapes only."""
    groups: dict[str, dict[str, int]] = {}
    for info in tensors.values():
        entry = groups.setdefault(
            classify(info.name, rules), {"tensors": 0, "parameters": 0, "bytes": 0}
        )
        entry["tensors"] += 1
        entry["parameters"] += info.numel
        entry["bytes"] += info.bits // 8
    return {
        "groups": dict(sorted(groups.items())),
        "total_tensors": len(tensors),
        "total_parameters": sum(info.numel for info in tensors.values()),
        "total_bytes": sum(info.bits // 8 for info in tensors.values()),
    }


def _load_rules(path: Path) -> tuple[tuple[str, str], ...]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(
        isinstance(item, list) and len(item) == 2 for item in data
    ):
        raise ValueError("rules file must be a JSON list of [group, regex] pairs")
    return tuple((group, pattern) for group, pattern in data)


def _format_listing(tensors: Iterable[TensorInfo]) -> str:
    lines = [
        f"{info.name}\t{info.dtype}\t{list(info.shape)}\t{info.shard}"
        for info in sorted(tensors, key=lambda info: info.name)
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--rules", type=Path, help="JSON list of [group, regex] pairs")
    parser.add_argument(
        "--list", action="store_true", help="print every tensor name/dtype/shape and exit"
    )
    args = parser.parse_args()

    tensors = read_checkpoint(args.dir)
    if args.list:
        print(_format_listing(tensors.values()))
        return
    rules = _load_rules(args.rules) if args.rules else DEFAULT_GROUP_RULES
    print(json.dumps(audit(tensors, rules=rules), indent=2))


if __name__ == "__main__":
    main()
