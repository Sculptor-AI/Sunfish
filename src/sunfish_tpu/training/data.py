"""GCS/local indexed input pipeline with exact Kauldron iterator resume."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import struct
import sys
import threading
import time
from array import array
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

from etils import epath
from grain import python as grain
from kauldron.data import iterators as kd_iterators
from kauldron.data.py import base as kd_data_base
import numpy as np

from sunfish.datashards import MANIFEST_VERSION
from sunfish_tpu.training.packing import pack_training_record
from sunfish_tpu.training.record_format import TrainingRecord, decode_record

_OFFSET = struct.Struct("<Q")
_TOKEN_BYTES = 4
_INPUT_WAIT_LOCK = threading.Lock()
_INPUT_WAIT_SAMPLES: list[float] = []

# Kauldron batches before applying Grain multiprocessing prefetch.  Keep this
# small and explicit: it overlaps direct GCS range-read latency without adding
# a persistent/local-disk cache or an unbounded host-memory queue.
GRAIN_PREFETCH_BATCHES_PER_WORKER = 2


def consume_input_wait_metrics() -> dict[str, float | int]:
    """Return and reset main-process iterator wait samples since last log."""
    with _INPUT_WAIT_LOCK:
        samples = tuple(_INPUT_WAIT_SAMPLES)
        _INPUT_WAIT_SAMPLES.clear()
    total = sum(samples)
    return {
        "samples": len(samples),
        "total_seconds": total,
        "mean_seconds": total / len(samples) if samples else 0.0,
        "max_seconds": max(samples, default=0.0),
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class TimedPyGrainIterator(kd_iterators.PyGrainIterator):
    """Checkpoint-compatible PyGrain iterator with main-process wait timing."""

    def __kd_ocp_restore_post__(self, value: grain.DatasetIterator):
        """Preserve wait instrumentation across Kauldron's Orbax restore hook."""
        restored = super().__kd_ocp_restore_post__(value)
        if not isinstance(restored, kd_iterators.PyGrainIterator):
            raise TypeError(
                "Kauldron PyGrain restore returned "
                f"{type(restored)!r}, expected PyGrainIterator"
            )
        return type(self)(source=restored.source, iter=restored.iter)

    def __next__(self):
        started = time.perf_counter()
        try:
            return super().__next__()
        finally:
            elapsed = time.perf_counter() - started
            with _INPUT_WAIT_LOCK:
                _INPUT_WAIT_SAMPLES.append(elapsed)


def manifest_sha256(directory: str) -> str:
    """Return the byte-exact identity of ``manifest.json`` on local disk/GCS."""
    payload = (epath.Path(directory) / "manifest.json").read_bytes()
    return hashlib.sha256(payload).hexdigest()


class EPathShardedRecordSource(grain.RandomAccessDataSource):
    """Random-access uint32 records backed by local paths or ``gs://``.

    Index files are read once.  Bin records use seek/range reads and only the
    requested record is materialized.  Full shard hashing is optional because
    it is expensive on every TPU host; run identity always pins and verifies
    the manifest bytes, and immutable GCS prefixes are the production policy.
    """

    def __init__(
        self,
        directory: str,
        *,
        expected_manifest_sha256: str,
        verify_shard_hashes: bool,
    ):
        self.directory = directory
        root = epath.Path(directory)
        manifest_bytes = (root / "manifest.json").read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "dataset manifest identity mismatch: "
                f"expected {expected_manifest_sha256}, got {self.manifest_sha256}"
            )
        manifest = json.loads(manifest_bytes)
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError("unsupported data manifest version")

        # Keep one compact uint64 offset array per shard. A Python tuple per
        # record costs an order of magnitude more host RAM and becomes
        # prohibitive for the full coding/agent corpus.
        self._shards: list[tuple[str, array]] = []
        self._cumulative_records: list[int] = []
        self._records_read = 0
        self._payload_bytes_read = 0
        self._range_read_seconds = 0.0
        total_records = 0
        total_tokens = 0
        for shard in manifest.get("shards", ()):
            bin_path = root / shard["bin"]
            idx_path = root / shard["idx"]
            if verify_shard_hashes:
                for path, key in ((bin_path, "sha256_bin"), (idx_path, "sha256_idx")):
                    digest = _path_sha256(path)
                    if digest != shard[key]:
                        raise ValueError(f"{path} does not match manifest hash")

            index_bytes = idx_path.read_bytes()
            if len(index_bytes) % _OFFSET.size:
                raise ValueError(f"{idx_path} byte length is not a uint64 index")
            offsets = array("Q")
            if offsets.itemsize != _OFFSET.size:
                raise RuntimeError("platform uint64 array required")
            offsets.frombytes(index_bytes)
            if sys.byteorder != "little":
                offsets.byteswap()
            if len(offsets) != int(shard["records"]):
                raise ValueError(f"{idx_path} record count mismatch")
            previous = 0
            for end in offsets:
                if end < previous:
                    raise ValueError(f"{idx_path} offsets are not monotonic")
                previous = int(end)
            if previous != int(shard["tokens"]):
                raise ValueError(f"{idx_path} final offset differs from shard token count")
            self._shards.append((str(bin_path), offsets))
            total_records += len(offsets)
            self._cumulative_records.append(total_records)
            total_tokens += previous

        if total_records != int(manifest.get("total_records", -1)):
            raise ValueError("manifest total_records mismatch")
        if total_tokens != int(manifest.get("total_tokens", -1)):
            raise ValueError("manifest total_tokens mismatch")
        if not total_records:
            raise ValueError("training dataset contains no records")

    def __len__(self) -> int:
        return self._cumulative_records[-1]

    @property
    def read_metrics(self) -> dict[str, float | int]:
        """Best-effort per-process range-read counters for readiness evidence.

        Multiprocessing workers own independent source copies, so production
        training should treat these as host/source diagnostics rather than a
        globally synchronized metric. The Stage-0.5 input smoke runs without
        workers and records an exact per-host baseline.
        """
        seconds = self._range_read_seconds
        return {
            "records_read": self._records_read,
            "payload_bytes_read": self._payload_bytes_read,
            "range_read_seconds": seconds,
            "mean_range_read_ms": (
                seconds * 1000.0 / self._records_read if self._records_read else 0.0
            ),
            "payload_mib_per_second": (
                self._payload_bytes_read / (1024.0 * 1024.0) / seconds
                if seconds > 0.0
                else 0.0
            ),
        }

    def __getitem__(self, record_key: int) -> array:
        if not 0 <= record_key < len(self):
            raise IndexError(record_key)
        shard_index = bisect_right(self._cumulative_records, record_key)
        shard_start = (
            self._cumulative_records[shard_index - 1] if shard_index else 0
        )
        path_string, offsets = self._shards[shard_index]
        local_index = record_key - shard_start
        start = int(offsets[local_index - 1]) if local_index else 0
        end = int(offsets[local_index])
        byte_count = (end - start) * _TOKEN_BYTES
        started = time.perf_counter()
        with epath.Path(path_string).open("rb") as source:
            source.seek(start * _TOKEN_BYTES)
            payload = source.read(byte_count)
        elapsed = time.perf_counter() - started
        if len(payload) != byte_count:
            raise IOError(f"short range read for record {record_key} from {path_string}")
        self._records_read += 1
        self._payload_bytes_read += len(payload)
        self._range_read_seconds += elapsed
        words = array("I")
        words.frombytes(payload)
        if sys.byteorder != "little":
            words.byteswap()
        return words


class TrainingRecordSource(grain.RandomAccessDataSource):
    """Decode opaque shard records into fixed-shape DiffusionGemma batches."""

    def __init__(
        self,
        *,
        directory: str,
        expected_manifest_sha256: str,
        verify_shard_hashes: bool,
        prompt_length: int,
        canvas_size: int,
        num_canvases: int,
        vocab_size: int,
        pad_token: int,
        eos_token: int,
    ):
        self._source = EPathShardedRecordSource(
            directory,
            expected_manifest_sha256=expected_manifest_sha256,
            verify_shard_hashes=verify_shard_hashes,
        )
        self.prompt_length = prompt_length
        self.canvas_size = canvas_size
        self.num_canvases = num_canvases
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.eos_token = eos_token

    @property
    def manifest_sha256(self) -> str:
        return self._source.manifest_sha256

    @property
    def read_metrics(self) -> dict[str, float | int]:
        return self._source.read_metrics

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, record_key: int) -> dict[str, np.ndarray]:
        record = decode_record(self._source[record_key])
        return pack_diffusion_example(
            record,
            record_id=record_key,
            prompt_length=self.prompt_length,
            canvas_size=self.canvas_size,
            num_canvases=self.num_canvases,
            vocab_size=self.vocab_size,
            pad_token=self.pad_token,
            eos_token=self.eos_token,
        )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class SunfishData(kd_data_base.DataSourceBase):
    """Kauldron/Grain pipeline whose iterator state Orbax checkpoints.

    ``DataSourceBase`` performs process slicing before global shuffling, so
    every host reads a disjoint, exhaustive record subset.  Kauldron wraps the
    resulting Grain iterator with ``PyGrainCheckpointHandler`` and saves it in
    the same checkpoint transaction as model and optimizer state.
    """

    directory: str
    expected_manifest_sha256: str
    prompt_length: int
    canvas_size: int
    num_canvases: int
    vocab_size: int
    pad_token: int
    eos_token: int
    verify_shard_hashes: bool = False

    @functools.cached_property
    def data_source(self) -> grain.RandomAccessDataSource:
        return TrainingRecordSource(
            directory=self.directory,
            expected_manifest_sha256=self.expected_manifest_sha256,
            verify_shard_hashes=self.verify_shard_hashes,
            prompt_length=self.prompt_length,
            canvas_size=self.canvas_size,
            num_canvases=self.num_canvases,
            vocab_size=self.vocab_size,
            pad_token=self.pad_token,
            eos_token=self.eos_token,
        )

    def __iter__(self) -> kd_iterators.Iterator:
        base = super().__iter__()
        if not isinstance(base, kd_iterators.PyGrainIterator):
            raise TypeError(f"expected PyGrainIterator, got {type(base)!r}")
        return TimedPyGrainIterator(source=base.source, iter=base.iter)


def pack_diffusion_example(
    record: TrainingRecord,
    *,
    record_id: int,
    prompt_length: int,
    canvas_size: int,
    num_canvases: int,
    vocab_size: int,
    pad_token: int,
    eos_token: int,
) -> dict[str, np.ndarray]:
    """Create the exact fields consumed by the training model."""
    packed = pack_training_record(
        record,
        record_id=record_id,
        prompt_length=prompt_length,
        canvas_size=canvas_size,
        num_canvases=num_canvases,
        vocab_size=vocab_size,
        pad_token=pad_token,
        eos_token=eos_token,
    )

    return {
        "prompt": np.asarray(packed.prompt, dtype=np.int32),
        "canvas": np.asarray(packed.canvas, dtype=np.int32)[:, None],
        "canvas_id": np.asarray(packed.canvas_id, dtype=np.int32),
        "canvas_mask": np.asarray(packed.canvas_mask, dtype=np.bool_),
        "canvas_loss_mask": np.asarray(packed.canvas_loss_mask, dtype=np.bool_),
        "encoder_target": np.asarray(packed.encoder_target, dtype=np.int32),
        "encoder_target_mask": np.asarray(
            packed.encoder_target_mask, dtype=np.bool_
        ),
        "bucket_id": np.asarray(packed.bucket_id, dtype=np.uint32),
        "record_id": np.asarray(packed.record_id, dtype=np.uint32),
    }


def _path_sha256(path: epath.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
