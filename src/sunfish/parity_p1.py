"""Parity harness P1: full static equivalence check (docs/parity_harness.md).

Dependency-free and local: pure file I/O over the two checkpoints.

P1.1 tokenizer files byte-identical
P1.2 configs equal except exactly text_config.num_experts/top_k (unchanged
     for the 128/8 control) and vision_config: null
P1.3 EVERY control tensor's bytes hash-equal to its source counterpart, and
     the control tensor set == source set minus the conversion manifest's
     dropped names (sampling is false economy — external review + wire seq 2)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

_HEADER_LEN = struct.Struct("<Q")


def _tensor_table(directory: Path) -> dict[str, tuple[Path, int, int]]:
    """name -> (shard path, absolute byte start, absolute byte end)."""
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    table: dict[str, tuple[Path, int, int]] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        shard = directory / shard_name
        with shard.open("rb") as handle:
            (hlen,) = _HEADER_LEN.unpack(handle.read(8))
            header = json.loads(handle.read(hlen))
        data_start = 8 + hlen
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            table[name] = (shard, data_start + start, data_start + end)
    return table


def _hash_range(path: Path, start: int, end: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            block = handle.read(min(remaining, 1 << 24))
            if not block:
                raise ValueError(f"unexpected EOF in {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def run_p1(source: Path, control: Path) -> dict[str, object]:
    report: dict[str, object] = {"checks": {}}
    checks = report["checks"]

    # P1.1 — tokenizer byte identity
    tokenizer_ok = True
    for name in ("tokenizer.json", "tokenizer_config.json"):
        a, b = source / name, control / name
        same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        tokenizer_ok &= same
    checks["p1.1_tokenizer_identical"] = tokenizer_ok

    # P1.2 — config diff is exactly the permitted set
    source_config = json.loads((source / "config.json").read_text())
    control_config = json.loads((control / "config.json").read_text())
    permitted = {("vision_config",)}
    diffs = []

    def walk(a, b, path=()):
        keys = set(a) | set(b)
        for key in sorted(keys):
            sub = path + (key,)
            if key not in a or key not in b:
                diffs.append(sub)
            elif isinstance(a[key], dict) and isinstance(b[key], dict):
                walk(a[key], b[key], sub)
            elif a[key] != b[key]:
                diffs.append(sub)

    walk(source_config, control_config)
    unexpected = [".".join(d) for d in diffs if d not in permitted]
    checks["p1.2_config_diff_exact"] = not unexpected
    if unexpected:
        report["unexpected_config_diffs"] = unexpected
    checks["p1.2_vision_config_null"] = control_config.get("vision_config") is None

    # P1.3 — tensor set and full byte-hash equality
    manifest = json.loads((control / "sunfish_conversion.json").read_text())
    dropped = set(manifest["dropped_tensor_names"])
    source_table = _tensor_table(source)
    control_table = _tensor_table(control)

    expected_names = set(source_table) - dropped
    set_ok = set(control_table) == expected_names
    checks["p1.3_tensor_set_exact"] = set_ok
    if not set_ok:
        report["missing"] = sorted(expected_names - set(control_table))[:10]
        report["extra"] = sorted(set(control_table) - expected_names)[:10]

    mismatched = []
    for count, name in enumerate(sorted(control_table)):
        s_path, s_start, s_end = source_table[name]
        c_path, c_start, c_end = control_table[name]
        if (s_end - s_start) != (c_end - c_start) or _hash_range(
            s_path, s_start, s_end
        ) != _hash_range(c_path, c_start, c_end):
            mismatched.append(name)
        if count % 100 == 0:
            print(f"  hashed {count}/{len(control_table)} tensors", flush=True)
    checks["p1.3_all_tensor_hashes_equal"] = not mismatched
    if mismatched:
        report["mismatched_tensors"] = mismatched[:20]

    report["tensors_compared"] = len(control_table)
    report["passed"] = all(checks.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="write JSON report here")
    args = parser.parse_args()
    report = run_p1(args.source, args.control)
    payload = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
