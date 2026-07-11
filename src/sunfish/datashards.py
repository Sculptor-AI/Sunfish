"""Immutable, indexed token shards with process-disjoint resumable sampling.

The storage format the input pipeline is built on (external review item 5,
reference/tpu-docs/data-loading.md):

- ``<name>.bin``  — concatenated uint32-LE token records
- ``<name>.idx``  — uint64-LE END offset (in tokens) of each record, so
  record i spans ``[idx[i-1], idx[i])``; random access is two seeks
- ``manifest.json`` — file names, record/token counts, sha256 of every file,
  provenance; its hash is the dataset identity the checkpoint pins

``ShardedRecordSource`` exposes ``__getitem__``/``__len__`` (exactly what
Grain's ``MapDataset.source`` wants); ``ResumableSampler`` provides
deterministic shuffling, process-disjoint assignment, and ``get_state``/
``set_state`` for exact resume. Dependency-free stdlib Python by design —
it must run on TPU hosts, CPU boxes, and this laptop identically.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
from array import array
from pathlib import Path

_TOKEN = struct.Struct("<I")
_OFFSET = struct.Struct("<Q")
MANIFEST_VERSION = 1


class ShardWriter:
    """Stream records into a .bin/.idx pair; immutable once closed."""

    def __init__(self, path_stem: Path):
        self._bin_path = path_stem.with_suffix(".bin")
        self._idx_path = path_stem.with_suffix(".idx")
        for path in (self._bin_path, self._idx_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite immutable shard {path}")
        self._bin = self._bin_path.open("xb")
        self._idx = self._idx_path.open("xb")
        self.tokens = 0
        self.records = 0

    def add(self, token_ids: list[int] | array) -> None:
        data = array("I", token_ids)
        if data.itemsize != 4:  # exotic platforms
            raise RuntimeError("platform uint32 array required")
        self._bin.write(data.tobytes())
        self.tokens += len(data)
        self.records += 1
        self._idx.write(_OFFSET.pack(self.tokens))

    def close(self) -> dict[str, object]:
        self._bin.close()
        self._idx.close()
        return {
            "bin": self._bin_path.name,
            "idx": self._idx_path.name,
            "records": self.records,
            "tokens": self.tokens,
            "sha256_bin": _file_sha256(self._bin_path),
            "sha256_idx": _file_sha256(self._idx_path),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(output_dir: Path, shards: list[dict], **provenance: object) -> str:
    """Write manifest.json; returns its sha256 (the dataset identity)."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "shards": shards,
        "total_records": sum(int(s["records"]) for s in shards),
        "total_tokens": sum(int(s["tokens"]) for s in shards),
        **provenance,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "manifest.json").write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ShardedRecordSource:
    """Random access over every record in a manifest's shards.

    ``__getitem__``/``__len__`` — pluggable directly into Grain. Verifies
    file hashes against the manifest on open (verify=True) so a silently
    modified shard cannot poison a resumed run.
    """

    def __init__(self, directory: Path, *, verify: bool = True):
        self.directory = Path(directory)
        payload = (self.directory / "manifest.json").read_text(encoding="utf-8")
        self.manifest = json.loads(payload)
        self.manifest_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError("unsupported manifest version")

        self._records: list[tuple[Path, int, int]] = []  # (bin_path, start, end) in tokens
        for shard in self.manifest["shards"]:
            bin_path = self.directory / shard["bin"]
            idx_path = self.directory / shard["idx"]
            if verify:
                for path, key in ((bin_path, "sha256_bin"), (idx_path, "sha256_idx")):
                    if _file_sha256(path) != shard[key]:
                        raise ValueError(f"{path.name} does not match manifest hash")
            offsets = array("Q")
            offsets.frombytes(idx_path.read_bytes())
            if len(offsets) != shard["records"]:
                raise ValueError(f"{idx_path.name} record count mismatch")
            start = 0
            for end in offsets:
                self._records.append((bin_path, start, int(end)))
                start = int(end)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> array:
        bin_path, start, end = self._records[index]
        with bin_path.open("rb") as handle:
            handle.seek(start * _TOKEN.size)
            data = array("I")
            data.frombytes(handle.read((end - start) * _TOKEN.size))
        return data


class ResumableSampler:
    """Deterministic, process-disjoint, exactly-resumable record sampler.

    Each epoch is a seeded permutation of all record ids; process ``p`` of
    ``n`` takes positions ``p::n`` (disjoint and exhaustive by construction).
    ``get_state``/``set_state`` capture (seed, epoch, cursor) — sufficient
    because shards are immutable and identified by manifest hash.
    """

    def __init__(
        self,
        num_records: int,
        *,
        seed: int,
        process_index: int,
        process_count: int,
        manifest_hash: str,
    ):
        if not 0 <= process_index < process_count:
            raise ValueError("process_index must be in [0, process_count)")
        if num_records <= 0:
            raise ValueError("num_records must be positive")
        self.num_records = num_records
        self.seed = seed
        self.process_index = process_index
        self.process_count = process_count
        self.manifest_hash = manifest_hash
        self.epoch = 0
        self.cursor = 0  # position within THIS process's slice of the epoch
        self._permutation: list[int] | None = None

    def _epoch_permutation(self) -> list[int]:
        if self._permutation is None:
            material = f"sunfish-datashards:{self.seed}:{self.epoch}".encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "little"))
            permutation = list(range(self.num_records))
            rng.shuffle(permutation)
            self._permutation = permutation
        return self._permutation

    def _slice(self) -> range:
        return range(self.process_index, self.num_records, self.process_count)

    def __iter__(self):
        return self

    def __next__(self) -> int:
        positions = self._slice()
        if self.cursor >= len(positions):
            self.epoch += 1
            self.cursor = 0
            self._permutation = None
        record = self._epoch_permutation()[positions[self.cursor]]
        self.cursor += 1
        return record

    def get_state(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "process_index": self.process_index,
            "process_count": self.process_count,
        }

    def set_state(self, state: dict[str, object]) -> None:
        if state["manifest_hash"] != self.manifest_hash:
            raise ValueError("checkpointed loader state belongs to a different dataset")
        if (state["process_index"], state["process_count"]) != (
            self.process_index,
            self.process_count,
        ):
            raise ValueError("loader state belongs to a different process topology")
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._permutation = None
