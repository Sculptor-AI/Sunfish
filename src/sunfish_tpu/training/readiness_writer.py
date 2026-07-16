"""Immutable per-step TPU readiness evidence for the production trainer."""

from __future__ import annotations

import dataclasses
import contextlib
import json
import math
import os
import re
from collections.abc import Mapping
from functools import cached_property
from typing import Any

from etils import epath
import jax
from kauldron.train import metric_writer
import numpy as np

from sunfish_tpu.training.data import (
    GRAIN_PREFETCH_BATCHES_PER_WORKER,
    consume_input_wait_metrics,
)

_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _local_device_memory_snapshot() -> dict[str, Any]:
    devices = []
    for local_index, device in enumerate(jax.local_devices()):
        stats = device.memory_stats()
        if not isinstance(stats, Mapping):
            raise RuntimeError(
                f"local device {local_index} did not expose TPU memory_stats"
            )
        values = {}
        for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
            value = stats.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise RuntimeError(
                    f"local device {local_index} has invalid memory_stats[{name!r}]"
                )
            values[name] = int(value)
        if not (
            0 <= values["bytes_in_use"] <= values["peak_bytes_in_use"]
            <= values["bytes_limit"]
            and values["bytes_limit"] > 0
        ):
            raise RuntimeError(
                f"local device {local_index} reported inconsistent TPU memory_stats"
            )
        devices.append(
            {
                "local_device_index": local_index,
                "device_id": int(device.id),
                "platform": str(device.platform),
                **values,
            }
        )
    if not devices:
        raise RuntimeError("readiness evidence found no local TPU devices")
    return {
        "schema_version": 1,
        "purpose": "cloud-tpu-device-memory-snapshot",
        "devices": devices,
    }


def _json_scalar(value: Any) -> int | float | bool:
    value = np.asarray(jax.device_get(value))
    if value.shape != ():
        raise ValueError(f"readiness scalar has non-scalar shape {value.shape}")
    result = value.item()
    if isinstance(result, (np.bool_, bool)):
        return bool(result)
    if isinstance(result, (np.integer, int)):
        return int(result)
    result = float(result)
    if not math.isfinite(result):
        raise ValueError(f"readiness scalar is non-finite: {result}")
    return result


def _write_immutable_json(path: epath.Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"readiness evidence changed at {path}")
        return
    path.write_text(encoded)


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class ReadinessMetricWriter(metric_writer.KDMetricWriter):
    """Default Kauldron writer plus immutable scalar/input-wait JSON evidence."""

    readiness_enabled: bool = False
    run_id: str = ""
    config_sha256: str = ""
    dataset_manifest_sha256: str = ""
    seed_manifest_sha256: str = ""
    source_git_commit: str = ""
    source_tree_sha256: str = ""

    @cached_property
    def _lineage(self) -> dict[str, Any]:
        lineage = {
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "seed_manifest_sha256": self.seed_manifest_sha256,
            "sunfish_source": {
                "git_commit": self.source_git_commit,
                "source_tree_sha256": self.source_tree_sha256,
            },
        }
        if self.readiness_enabled:
            if not _ATTEMPT_ID.fullmatch(self.run_id):
                raise RuntimeError("readiness run_id is missing or invalid")
            for name in (
                "config_sha256",
                "dataset_manifest_sha256",
                "seed_manifest_sha256",
                "source_tree_sha256",
            ):
                value = str(
                    lineage["sunfish_source"][name]
                    if name == "source_tree_sha256"
                    else lineage[name]
                )
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise RuntimeError(f"readiness {name} is missing or invalid")
            if not re.fullmatch(r"[0-9a-f]{40}", self.source_git_commit):
                raise RuntimeError("readiness source_git_commit is missing or invalid")
        return lineage

    @cached_property
    def _attempt_id(self) -> str:
        attempt_id = os.environ.get("SUNFISH_ATTEMPT_ID", "")
        if self.readiness_enabled and not _ATTEMPT_ID.fullmatch(attempt_id):
            raise RuntimeError(
                "SUNFISH_ATTEMPT_ID is required for readiness evidence and must "
                "contain only letters, numbers, dot, underscore, or dash"
            )
        return attempt_id

    @cached_property
    def _readiness_root(self) -> epath.Path:
        return epath.Path(self.workdir) / "readiness" / self._attempt_id

    def write_step_metrics(self, *, step, aux, schedules, log_summaries, timer=None):
        pause = (
            timer.pause("readiness_evidence_and_metrics")
            if self.readiness_enabled and timer is not None
            else contextlib.nullcontext()
        )
        # Kauldron starts the next step's clock before calling the writer. Keep
        # synchronous GCS evidence and metric serialization outside that clock
        # so the following input/compute measurement is not contaminated.
        with pause:
            if self.readiness_enabled:
                _write_immutable_json(
                    self._readiness_root
                    / "input-wait"
                    / f"host-{int(jax.process_index()):05d}"
                    / f"step-{int(step):09d}.json",
                    {
                        "schema_version": 1,
                        "attempt_id": self._attempt_id,
                        **self._lineage,
                        "process_index": int(jax.process_index()),
                        "process_count": int(jax.process_count()),
                        "step": int(step),
                        "input_wait": consume_input_wait_metrics(),
                        "local_cache_policy": "none-direct-gcs-range-reads",
                        "local_cache_bytes": 0,
                        "memory_prefetch_policy": "grain-mp-prefetch-bounded",
                        "per_worker_prefetch_batches": (
                            GRAIN_PREFETCH_BATCHES_PER_WORKER
                        ),
                        "device_memory": _local_device_memory_snapshot(),
                    },
                )
            return super().write_step_metrics(
                step=step,
                aux=aux,
                schedules=schedules,
                log_summaries=log_summaries,
                timer=timer,
            )

    def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
        if self.readiness_enabled and int(jax.process_index()) == 0:
            normalized = {key: _json_scalar(value) for key, value in scalars.items()}
            _write_immutable_json(
                self._readiness_root / "metrics" / f"step-{int(step):09d}.json",
                {
                    "schema_version": 1,
                    "attempt_id": self._attempt_id,
                    **self._lineage,
                    "process_index": 0,
                    "process_count": int(jax.process_count()),
                    "step": int(step),
                    "scalars": normalized,
                },
            )
        return super().write_scalars(step, scalars)
