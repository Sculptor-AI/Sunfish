"""Analyze immutable production-trainer evidence for readiness gates 4 and 8."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from sunfish_tpu.source_identity import normalize_source_identity


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_HBM_PEAK_FRACTION = 0.90


def _evidence_lineage(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    run_id = payload.get("run_id")
    config = payload.get("config_sha256")
    dataset = payload.get("dataset_manifest_sha256")
    seed = payload.get("seed_manifest_sha256")
    source = normalize_source_identity(payload.get("sunfish_source"))
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(config, str)
        or not _SHA256.fullmatch(config)
        or not isinstance(dataset, str)
        or not _SHA256.fullmatch(dataset)
        or not isinstance(seed, str)
        or not _SHA256.fullmatch(seed)
        or source is None
    ):
        return None
    return {
        "run_id": run_id,
        "config_sha256": config,
        "dataset_manifest_sha256": dataset,
        "seed_manifest_sha256": seed,
        "sunfish_source": {
            "git_commit": source[0],
            "source_tree_sha256": source[1],
        },
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def analyze_smoke_evidence(
    metric_payloads: Sequence[Mapping[str, Any]],
    input_wait_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_processes: int,
    min_steps: int = 100,
    min_relative_loss_reduction: float = 0.10,
    max_p95_input_wait_ratio: float = 0.10,
    steady_state_start: int = 20,
) -> dict[str, Any]:
    """Return fail-closed gate-4/gate-8 evidence from one trainer attempt."""
    if expected_processes <= 0 or min_steps <= 0:
        raise ValueError("expected_processes and min_steps must be positive")
    if not 0.0 <= min_relative_loss_reduction < 1.0:
        raise ValueError("min_relative_loss_reduction must be in [0, 1)")
    if not 0.0 <= max_p95_input_wait_ratio <= 1.0:
        raise ValueError("max_p95_input_wait_ratio must be in [0, 1]")

    errors: list[str] = []
    metrics_by_step: dict[int, Mapping[str, Any]] = {}
    attempt_ids: set[str] = set()
    metric_lineages: list[dict[str, Any] | None] = []
    for payload in metric_payloads:
        metric_lineages.append(_evidence_lineage(payload))
        if payload.get("schema_version") != 1:
            errors.append("metric evidence has an unsupported schema")
            continue
        if payload.get("process_index") != 0:
            errors.append("metric evidence was not emitted by process 0")
        if payload.get("process_count") != expected_processes:
            errors.append("metric evidence reports the wrong process count")
        step = payload.get("step")
        if not isinstance(step, int) or step < 0:
            errors.append("metric evidence has an invalid step")
            continue
        if step in metrics_by_step:
            errors.append(f"duplicate metric evidence for step {step}")
            continue
        metrics_by_step[step] = payload
        attempt_ids.add(str(payload.get("attempt_id", "")))
    if len(attempt_ids) != 1 or "" in attempt_ids:
        errors.append(f"metric evidence attempt IDs differ: {sorted(attempt_ids)}")
    valid_metric_lineages = [item for item in metric_lineages if item is not None]
    if len(valid_metric_lineages) != len(metric_lineages) or not metric_lineages:
        errors.append("metric evidence has missing or invalid run lineage")
    elif any(item != valid_metric_lineages[0] for item in valid_metric_lineages[1:]):
        errors.append("metric evidence run lineages differ")
    lineage = valid_metric_lineages[0] if valid_metric_lineages else None

    steps = sorted(metrics_by_step)
    if len(steps) < min_steps:
        errors.append(f"only {len(steps)} metric steps found; require {min_steps}")
    if steps and steps[0] != 0:
        errors.append(
            f"fresh tiny-overfit metric evidence starts at step {steps[0]}, expected 0"
        )
    if steps and steps != list(range(steps[0], steps[-1] + 1)):
        errors.append("metric steps are not contiguous")

    losses: list[float] = []
    gradient_norms: list[float] = []
    update_norms: list[float] = []
    wall_seconds: dict[int, float] = {}
    for step in steps:
        scalars = metrics_by_step[step].get("scalars")
        if not isinstance(scalars, Mapping):
            errors.append(f"step {step} has no scalar mapping")
            continue
        required = (
            "losses/total",
            "metrics/gradient_norm",
            "metrics/update_norm",
            "perf_stats/steps_per_sec",
        )
        missing = [key for key in required if key not in scalars]
        if missing:
            errors.append(f"step {step} is missing scalars {missing}")
            continue
        values = [float(scalars[key]) for key in required]
        if not all(math.isfinite(value) for value in values):
            errors.append(f"step {step} has non-finite scalars")
            continue
        loss, gradient_norm, update_norm, steps_per_second = values
        if loss < 0 or gradient_norm < 0 or update_norm < 0:
            errors.append(f"step {step} has a negative loss/norm")
        if steps_per_second <= 0:
            errors.append(f"step {step} has non-positive throughput")
            continue
        losses.append(loss)
        gradient_norms.append(gradient_norm)
        update_norms.append(update_norm)
        wall_seconds[step] = 1.0 / steps_per_second

    loss_reduction = None
    first_loss = None
    final_loss = None
    if len(losses) >= min_steps:
        window = min(10, max(1, len(losses) // 4))
        first_loss = statistics.median(losses[:window])
        final_loss = statistics.median(losses[-window:])
        loss_reduction = (first_loss - final_loss) / max(abs(first_loss), 1e-12)
        if loss_reduction < min_relative_loss_reduction:
            errors.append(
                f"loss reduction {loss_reduction:.4f} is below "
                f"{min_relative_loss_reduction:.4f}"
            )
    if gradient_norms and max(gradient_norms) <= 0.0:
        errors.append("all gradient norms are zero")
    if update_norms and max(update_norms) <= 0.0:
        errors.append("all update norms are zero")

    waits_by_step: dict[int, dict[int, Mapping[str, Any]]] = {}
    wait_lineages: list[dict[str, Any] | None] = []
    for payload in input_wait_payloads:
        wait_lineages.append(_evidence_lineage(payload))
        if payload.get("schema_version") != 1:
            errors.append("input-wait evidence has an unsupported schema")
            continue
        if str(payload.get("attempt_id", "")) not in attempt_ids:
            errors.append("input-wait evidence has a different attempt ID")
            continue
        step = payload.get("step")
        process = payload.get("process_index")
        if not isinstance(step, int) or not isinstance(process, int):
            errors.append("input-wait evidence has invalid step/process")
            continue
        if payload.get("process_count") != expected_processes:
            errors.append("input-wait evidence reports the wrong process count")
        host_map = waits_by_step.setdefault(step, {})
        if process in host_map:
            errors.append(f"duplicate input-wait evidence for step {step} host {process}")
        host_map[process] = payload

    valid_wait_lineages = [item for item in wait_lineages if item is not None]
    if len(valid_wait_lineages) != len(wait_lineages) or not wait_lineages:
        errors.append("input-wait evidence has missing or invalid run lineage")
    elif any(item != valid_wait_lineages[0] for item in valid_wait_lineages[1:]):
        errors.append("input-wait evidence run lineages differ")
    elif lineage is None or valid_wait_lineages[0] != lineage:
        errors.append("input-wait evidence lineage differs from metric evidence")

    ratios: list[float] = []
    analyzed_steps: list[int] = []
    expected_indices = set(range(expected_processes))
    hbm_peak_fractions: list[float] = []
    hbm_device_counts: set[int] = set()
    hbm_baseline: dict[tuple[int, int], tuple[int, int]] | None = None
    previous_hbm_peaks: dict[tuple[int, int], int] = {}
    for step in steps:
        hosts = waits_by_step.get(step, {})
        if set(hosts) != expected_indices:
            errors.append(
                f"step {step} HBM evidence host set is {sorted(hosts)}, "
                f"expected {sorted(expected_indices)}"
            )
            continue
        step_devices: dict[tuple[int, int], tuple[int, int, int]] = {}
        global_device_ids: dict[int, tuple[int, int]] = {}
        complete_step = True
        for process, payload in sorted(hosts.items()):
            memory = payload.get("device_memory")
            if not isinstance(memory, Mapping) or memory.get("schema_version") != 1:
                errors.append(
                    f"step {step} host {process} has missing/invalid HBM evidence"
                )
                complete_step = False
                continue
            if memory.get("purpose") != "cloud-tpu-device-memory-snapshot":
                errors.append(f"step {step} host {process} changed the HBM policy")
                complete_step = False
                continue
            devices = memory.get("devices")
            if not isinstance(devices, list) or not devices:
                errors.append(f"step {step} host {process} has no HBM device snapshots")
                complete_step = False
                continue
            valid_devices = 0
            device_ids = set()
            for local_index, device in enumerate(devices):
                if not isinstance(device, Mapping) or device.get(
                    "local_device_index"
                ) != local_index:
                    errors.append(
                        f"step {step} host {process} has invalid HBM device ordering"
                    )
                    complete_step = False
                    continue
                device_id = device.get("device_id")
                if (
                    isinstance(device_id, bool)
                    or not isinstance(device_id, int)
                    or device_id in device_ids
                    or device.get("platform") != "tpu"
                ):
                    errors.append(
                        f"step {step} host {process} has invalid TPU device identity"
                    )
                    complete_step = False
                    continue
                device_ids.add(device_id)
                device_key = (process, local_index)
                prior_owner = global_device_ids.get(device_id)
                if prior_owner is not None and prior_owner != device_key:
                    errors.append(
                        f"step {step} TPU device ID {device_id} is duplicated by "
                        f"hosts/local-indices {prior_owner} and {device_key}"
                    )
                    complete_step = False
                else:
                    global_device_ids[device_id] = device_key
                values = []
                for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                    value = device.get(name)
                    if isinstance(value, bool) or not isinstance(value, int):
                        errors.append(
                            f"step {step} host {process} device {local_index} "
                            f"has invalid HBM {name}"
                        )
                        complete_step = False
                        break
                    values.append(value)
                else:
                    bytes_in_use, peak_bytes, bytes_limit = values
                    if not 0 <= bytes_in_use <= peak_bytes <= bytes_limit or bytes_limit <= 0:
                        errors.append(
                            f"step {step} host {process} device {local_index} "
                            "has inconsistent HBM counters"
                        )
                        complete_step = False
                        continue
                    step_devices[device_key] = (device_id, bytes_limit, peak_bytes)
                    fraction = peak_bytes / bytes_limit
                    hbm_peak_fractions.append(fraction)
                    valid_devices += 1
                    if fraction > _MAX_HBM_PEAK_FRACTION:
                        errors.append(
                            f"step {step} host {process} device {local_index} "
                            f"peak HBM fraction {fraction:.4f} exceeds "
                            f"{_MAX_HBM_PEAK_FRACTION:.4f}"
                        )
            if valid_devices == len(devices):
                hbm_device_counts.add(valid_devices)
            else:
                complete_step = False
        if complete_step:
            identity = {
                key: (device_id, bytes_limit)
                for key, (device_id, bytes_limit, _peak) in step_devices.items()
            }
            if hbm_baseline is None:
                hbm_baseline = identity
            elif identity != hbm_baseline:
                errors.append(
                    f"step {step} HBM device-ID/bytes-limit set differs from baseline"
                )
            if identity == hbm_baseline:
                for key, (_device_id, _bytes_limit, peak_bytes) in step_devices.items():
                    previous_peak = previous_hbm_peaks.get(key)
                    if previous_peak is not None and peak_bytes < previous_peak:
                        errors.append(
                            f"step {step} host {key[0]} device {key[1]} peak HBM "
                            f"decreased from {previous_peak} to {peak_bytes}"
                        )
                    previous_hbm_peaks[key] = max(
                        peak_bytes, previous_peak if previous_peak is not None else 0
                    )
    if len(hbm_device_counts) != 1:
        errors.append(
            "HBM evidence does not report one consistent positive local-device count"
        )
    if hbm_baseline is None:
        errors.append("HBM evidence has no complete device baseline")

    for step in steps:
        if step < steady_state_start or step not in wall_seconds:
            continue
        hosts = waits_by_step.get(step, {})
        if set(hosts) != expected_indices:
            errors.append(
                f"step {step} input-wait hosts are {sorted(hosts)}, "
                f"expected {sorted(expected_indices)}"
            )
            continue
        host_waits: list[float] = []
        for process, payload in sorted(hosts.items()):
            wait = payload.get("input_wait")
            if not isinstance(wait, Mapping) or int(wait.get("samples", 0)) < 1:
                errors.append(f"step {step} host {process} has no input-wait sample")
                continue
            seconds = float(wait.get("total_seconds", -1.0))
            if not math.isfinite(seconds) or seconds < 0:
                errors.append(f"step {step} host {process} has invalid wait time")
                continue
            if payload.get("local_cache_bytes") != 0:
                errors.append(f"step {step} host {process} used a local disk cache")
            if payload.get("local_cache_policy") != "none-direct-gcs-range-reads":
                errors.append(f"step {step} host {process} changed the direct-GCS policy")
            if payload.get("memory_prefetch_policy") != "grain-mp-prefetch-bounded":
                errors.append(f"step {step} host {process} changed the prefetch policy")
            if payload.get("per_worker_prefetch_batches") != 2:
                errors.append(f"step {step} host {process} changed the prefetch bound")
            host_waits.append(seconds)
        if len(host_waits) == expected_processes:
            ratios.append(max(host_waits) / wall_seconds[step])
            analyzed_steps.append(step)

    p95_wait_ratio = _percentile(ratios, 0.95) if ratios else None
    if not ratios:
        errors.append("no steady-state input-wait ratios could be computed")
    elif p95_wait_ratio > max_p95_input_wait_ratio:
        errors.append(
            f"p95 input-wait ratio {p95_wait_ratio:.4f} exceeds "
            f"{max_p95_input_wait_ratio:.4f}"
        )

    gate4_errors = [
        error
        for error in errors
        if not any(
            token in error
            for token in (
                "input-wait",
                "hosts are",
                "local disk cache",
                "direct-GCS",
                "prefetch",
            )
        )
        and "p95" not in error
        and "steady-state" not in error
    ]
    gate8_errors = [error for error in errors if error not in gate4_errors]
    return {
        "schema_version": 1,
        "attempt_id": next(iter(attempt_ids), ""),
        "expected_processes": expected_processes,
        **(lineage or {}),
        "passed": not errors,
        "gates": {
            "4": {
                "passed": not gate4_errors,
                "errors": gate4_errors,
                "metric_steps": len(steps),
                "first_loss_median": first_loss,
                "final_loss_median": final_loss,
                "relative_loss_reduction": loss_reduction,
                "max_gradient_norm": max(gradient_norms, default=None),
                "max_update_norm": max(update_norms, default=None),
                "required_relative_loss_reduction": min_relative_loss_reduction,
                "max_peak_hbm_fraction": max(hbm_peak_fractions, default=None),
                "max_allowed_peak_hbm_fraction": _MAX_HBM_PEAK_FRACTION,
                "local_device_count": (
                    next(iter(hbm_device_counts)) if len(hbm_device_counts) == 1 else None
                ),
            },
            "8": {
                "passed": not gate8_errors,
                "errors": gate8_errors,
                "steady_state_steps": analyzed_steps,
                "p95_input_wait_ratio": p95_wait_ratio,
                "max_p95_input_wait_ratio": max_p95_input_wait_ratio,
                "local_cache_policy": "none-direct-gcs-range-reads",
                "memory_prefetch_policy": "grain-mp-prefetch-bounded",
                "per_worker_prefetch_batches": 2,
            },
        },
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True, help="local or gs://.../readiness/ATTEMPT")
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--min-relative-loss-reduction", type=float, default=0.10)
    parser.add_argument("--max-p95-input-wait-ratio", type=float, default=0.10)
    parser.add_argument("--steady-state-start", type=int, default=20)
    args = parser.parse_args(argv)

    from etils import epath

    root = epath.Path(args.attempt_root)
    metrics = [json.loads(path.read_text()) for path in sorted((root / "metrics").glob("step-*.json"))]
    waits = [
        json.loads(path.read_text())
        for path in sorted((root / "input-wait").glob("host-*/step-*.json"))
    ]
    payload = analyze_smoke_evidence(
        metrics,
        waits,
        expected_processes=args.expected_processes,
        min_steps=args.min_steps,
        min_relative_loss_reduction=args.min_relative_loss_reduction,
        max_p95_input_wait_ratio=args.max_p95_input_wait_ratio,
        steady_state_start=args.steady_state_start,
    )
    output = root / "smoke-summary.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != encoded:
        print(f"sunfish-smoke-evidence: immutable output changed at {output}", file=sys.stderr)
        return 2
    if not output.exists():
        output.write_text(encoded)
    print(encoded, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
