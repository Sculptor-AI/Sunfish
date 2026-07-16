"""Bounded asynchronous reconstruction-artifact drain for calibration."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = (
    "shared_pre_router_residual",
    "topk_indices",
    "final_scaled_topk_weights",
)


class ReconstructionDrain:
    """Keep at most two device/host artifact batches resident at once."""

    def __init__(
        self,
        *,
        output_dir: str,
        process_index: int,
        run_id: str,
        calibration_run_sha256: str,
        max_tokens: int,
        initial_tokens: int = 0,
        max_pending: int = 2,
    ):
        if (
            process_index < 0
            or max_tokens < 0
            or initial_tokens < 0
            or initial_tokens > max_tokens
        ):
            raise ValueError("process_index/token limits are invalid")
        if not _ARTIFACT_ID.fullmatch(run_id) or not _SHA256.fullmatch(
            calibration_run_sha256
        ):
            raise ValueError("reconstruction artifact lineage is invalid")
        if max_pending != 2:
            raise ValueError("the reconstruction contract requires double buffering")
        self.output_dir = output_dir
        self.process_index = process_index
        self.run_id = run_id
        self.calibration_run_sha256 = calibration_run_sha256
        self.max_tokens = max_tokens
        self.tokens_submitted = initial_tokens
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"sunfish-reconstruction-{process_index}",
        )
        self._pending: deque[Future] = deque()
        self._completed: list[str] = []
        self._closed = False

    @property
    def pending(self) -> int:
        return len(self._pending)

    def submit(
        self,
        artifacts: dict[str, Any],
        *,
        bucket: str,
        artifact_id: str,
        valid_tokens: int,
    ) -> bool:
        if self._closed:
            raise RuntimeError("reconstruction drain is closed")
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError("invalid reconstruction artifact ID")
        if set(artifacts) != set(_FIELDS):
            raise ValueError(f"reconstruction fields must be exactly {_FIELDS}")
        if valid_tokens < 0:
            raise ValueError("valid_tokens must be non-negative")
        remaining = self.max_tokens - self.tokens_submitted
        take = min(valid_tokens, remaining)
        if take <= 0:
            return False
        while len(self._pending) >= 2:
            self._completed.append(self._pending.popleft().result())

        staged = {}
        for name in _FIELDS:
            value = artifacts[name][:take]
            copy_async = getattr(value, "copy_to_host_async", None)
            if copy_async is not None:
                copy_async()
            staged[name] = value
        self.tokens_submitted += take
        self._pending.append(
            self._executor.submit(
                _write_artifact,
                self.output_dir,
                self.process_index,
                self.run_id,
                self.calibration_run_sha256,
                bucket,
                artifact_id,
                take,
                staged,
            )
        )
        return True

    def close(self) -> list[str]:
        if self._closed:
            return []
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True
        return list(self._completed)

    def flush(self) -> list[str]:
        """Finish current uploads without closing the reusable double buffer."""
        if self._closed:
            raise RuntimeError("reconstruction drain is closed")
        completed = [future.result() for future in self._pending]
        self._completed.extend(completed)
        self._pending.clear()
        return completed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _write_artifact(
    output_dir: str,
    process_index: int,
    run_id: str,
    calibration_run_sha256: str,
    bucket: str,
    artifact_id: str,
    tokens: int,
    staged: dict[str, Any],
) -> str:
    import numpy as np
    from etils import epath

    root = epath.Path(output_dir) / f"host-{process_index:05d}"
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "shared_pre_router_residual": np.asarray(
            staged["shared_pre_router_residual"]
        ),
        "topk_indices": np.asarray(staged["topk_indices"], dtype=np.uint8),
        "final_scaled_topk_weights": np.asarray(
            staged["final_scaled_topk_weights"], dtype=np.float16
        ),
    }
    if arrays["shared_pre_router_residual"].dtype.itemsize != 2:
        raise ValueError("reconstruction residuals must use a two-byte dtype")
    if arrays["topk_indices"].dtype != np.uint8:
        raise ValueError("reconstruction choices must use uint8")
    if arrays["final_scaled_topk_weights"].dtype != np.float16:
        raise ValueError("reconstruction weights must use float16")
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "calibration_run_sha256": calibration_run_sha256,
        "bucket": bucket,
        "artifact_id": artifact_id,
        "process_index": process_index,
        "tokens": tokens,
        "fields": {},
    }
    for name, array in arrays.items():
        payload = array.tobytes(order="C")
        path = root / f"{artifact_id}.{name}.bin"
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"immutable reconstruction artifact changed: {path}"
                )
        else:
            path.write_bytes(payload)
        metadata["fields"][name] = {
            "path": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = root / f"{artifact_id}.json"
    encoded = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if manifest.exists():
        if manifest.read_text() != encoded:
            raise FileExistsError(
                f"immutable reconstruction manifest changed: {manifest}"
            )
    else:
        manifest.write_text(encoded)
    return str(manifest)
