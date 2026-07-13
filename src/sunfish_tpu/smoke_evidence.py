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
                errors.append(f"step {step} host {process} used unbounded local cache")
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
        if not any(token in error for token in ("input-wait", "hosts are", "local cache"))
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
            },
            "8": {
                "passed": not gate8_errors,
                "errors": gate8_errors,
                "steady_state_steps": analyzed_steps,
                "p95_input_wait_ratio": p95_wait_ratio,
                "max_p95_input_wait_ratio": max_p95_input_wait_ratio,
                "local_cache_policy": "none-direct-gcs-range-reads",
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
