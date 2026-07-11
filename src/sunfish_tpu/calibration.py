"""Router-calibration accumulation core (docs/calibration_hook.md rev 2).

The pure math of stage 1, separated from model integration so it is unit-
testable on CPU JAX: masked segment-sums of pre-truncation router softmaxes
into an f32[buckets, layers, experts] accumulator, per-host flush into the
dependency-free ``sunfish.router_stats`` schema, and the spec's sanity
invariants.

Flush topology (spec): each host sums its own shards and flushes its LOCAL
accumulator; the global total is the offline ``RouterStatsAccumulator.merge``
fold. Never psum-then-flush-everywhere (host-count multiplication bug).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from sunfish.router_stats import RouterStatsAccumulator

PAD_BUCKET = -1


@dataclasses.dataclass(frozen=True)
class CalibrationState:
    """Functional accumulator state threaded through the jitted step."""

    mass: object  # f32[num_buckets, num_layers, num_experts]
    tokens: object  # i32[num_buckets]


def init_state(num_buckets: int, num_layers: int, num_experts: int) -> CalibrationState:
    import jax.numpy as jnp

    if min(num_buckets, num_layers, num_experts) <= 0:
        raise ValueError("bucket/layer/expert counts must be positive")
    return CalibrationState(
        mass=jnp.zeros((num_buckets, num_layers, num_experts), jnp.float32),
        tokens=jnp.zeros((num_buckets,), jnp.int32),
    )


def accumulate(state: CalibrationState, router_probs, bucket_ids) -> CalibrationState:
    """Fold one batch of router softmaxes into the accumulator.

    router_probs: f32[tokens, layers, experts] — plain pre-truncation softmax
        (NO per_expert_scale; router.scale is inherent — spec rev 2).
    bucket_ids:   i32[tokens] — PAD_BUCKET (-1) contributes nothing.

    Pure and jit/donate-friendly: returns a new state.
    """
    import jax.numpy as jnp

    num_buckets = state.tokens.shape[0]
    valid = bucket_ids != PAD_BUCKET
    clamped = jnp.where(valid, bucket_ids, 0)
    onehot = jnp.asarray(
        (clamped[:, None] == jnp.arange(num_buckets)[None, :]) & valid[:, None],
        state.mass.dtype,
    )  # [tokens, buckets], zero rows for padding
    mass = state.mass + jnp.einsum("tb,tle->ble", onehot, router_probs)
    tokens = state.tokens + jnp.sum(onehot, axis=0).astype(state.tokens.dtype)
    return CalibrationState(mass=mass, tokens=tokens)


def flush_host(
    state: CalibrationState,
    *,
    bucket_names: list[str],
    process_index: int,
    shard_id: str,
    output_dir: Path,
    check: bool = True,
) -> Path:
    """Write this host's accumulator as router_stats JSON (spec file naming)."""
    import numpy as np

    mass = np.asarray(state.mass)
    tokens = np.asarray(state.tokens)
    if len(bucket_names) != mass.shape[0]:
        raise ValueError("bucket_names length must match accumulator buckets")
    if check:
        errors = sanity_errors(mass, tokens, bucket_names)
        if errors:
            raise ValueError("calibration sanity failed: " + "; ".join(errors))

    accumulator = RouterStatsAccumulator(
        num_layers=mass.shape[1], num_experts=mass.shape[2]
    )
    for b, bucket in enumerate(bucket_names):
        if tokens[b] == 0:
            continue
        for layer in range(mass.shape[1]):
            accumulator.update(
                bucket=bucket, layer=layer, probabilities=mass[b, layer].tolist()
            )
        accumulator.count_tokens(bucket=bucket, tokens=int(tokens[b]))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"router_stats.host{process_index}.shard{shard_id}.json"
    path.write_text(accumulator.to_json(), encoding="utf-8")
    return path


def sanity_errors(mass, tokens, bucket_names: list[str], tolerance: float = 0.01) -> list[str]:
    """Spec invariant 1: per (bucket, layer), mass sums to ~tokens."""
    errors = []
    for b, bucket in enumerate(bucket_names):
        if tokens[b] == 0:
            if float(mass[b].sum()) != 0.0:
                errors.append(f"{bucket}: mass without tokens")
            continue
        for layer in range(mass.shape[1]):
            ratio = float(mass[b, layer].sum()) / float(tokens[b])
            if not (1 - tolerance) <= ratio <= (1 + tolerance):
                errors.append(f"{bucket}/layer{layer}: mass/token ratio {ratio:.4f}")
    return errors


def merge_flushes(directory: Path) -> RouterStatsAccumulator:
    """Offline fold of every host/shard flush (spec invariant 3 by addition)."""
    paths = sorted(directory.glob("router_stats.host*.shard*.json"))
    if not paths:
        raise FileNotFoundError(f"no router_stats flushes under {directory}")
    merged = RouterStatsAccumulator.from_json(paths[0].read_text(encoding="utf-8"))
    for path in paths[1:]:
        merged.merge(RouterStatsAccumulator.from_json(path.read_text(encoding="utf-8")))
    return merged


def selection_inputs(merged: RouterStatsAccumulator, layer: int) -> dict[str, list[float]]:
    """Adapter to sunfish.expert_selection.select_experts bucket_mass input."""
    return merged.layer_bucket_mass(layer)
