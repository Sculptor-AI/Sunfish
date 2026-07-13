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
from pathlib import Path

from sunfish.router_stats import RouterStatsAccumulator

PAD_BUCKET = -1
PHASES = ("prefill", "denoise_high", "denoise_mid", "denoise_low")
WORKLOADS = (
    "code_completion",
    "repo_edit",
    "tool_calls",
    "agent_trajectory",
    "general_control",
    "reasoning_control",
)


@dataclasses.dataclass(frozen=True)
class CalibrationState:
    """Functional accumulator state threaded through the jitted step."""

    mass: object  # f32[num_buckets, num_layers, num_experts]
    tokens: object  # i32[num_buckets]

    def tree_flatten(self):
        """Make the state a first-class JAX pytree without importing JAX eagerly."""
        return (self.mass, self.tokens), None

    @classmethod
    def tree_unflatten(cls, _metadata, children):
        mass, tokens = children
        return cls(mass=mass, tokens=tokens)


def _register_state_pytree() -> None:
    """Register when the heavy runtime is present; keep module discovery cheap."""
    try:
        import jax
    except ModuleNotFoundError:
        return
    jax.tree_util.register_pytree_node_class(CalibrationState)


_register_state_pytree()


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

    router_probs = jnp.asarray(router_probs, dtype=jnp.float32)
    bucket_ids = jnp.asarray(bucket_ids, dtype=jnp.int32)
    if router_probs.ndim != 3:
        raise ValueError("router_probs must have shape [tokens, layers, experts]")
    if bucket_ids.ndim != 1 or bucket_ids.shape[0] != router_probs.shape[0]:
        raise ValueError("bucket_ids must have one entry per router-probability token")
    if tuple(router_probs.shape[1:]) != tuple(state.mass.shape[1:]):
        raise ValueError("router_probs layer/expert axes must match accumulator state")

    num_buckets = state.tokens.shape[0]
    # Invalid non-padding IDs are excluded here and will make phase/workload
    # coverage fail at flush. Callers should produce only PAD_BUCKET or an
    # in-range ID; keeping this data-dependent path JIT-safe avoids callbacks.
    valid = (bucket_ids >= 0) & (bucket_ids < num_buckets)
    clamped = jnp.where(valid, bucket_ids, 0)
    onehot = jnp.asarray(
        (clamped[:, None] == jnp.arange(num_buckets)[None, :]) & valid[:, None],
        state.mass.dtype,
    )  # [tokens, buckets], zero rows for padding
    mass = state.mass + jnp.einsum("tb,tle->ble", onehot, router_probs)
    tokens = state.tokens + jnp.sum(onehot, axis=0).astype(state.tokens.dtype)
    return CalibrationState(mass=mass, tokens=tokens)


def call_with_router_probabilities(
    forward,
    *args,
    expected_layers: int,
    **kwargs,
):
    """Run one Linen encoder/decoder forward and capture plain router softmaxes.

    The pinned Gemma MoE does not expose pre-top-k probabilities. Flax method
    interception observes each ``MoE._router`` input without changing upstream
    source. Capture happens while JAX traces the forward, so the returned
    ``[tokens, layers, experts]`` stack remains an ordinary JIT value.
    """
    if expected_layers <= 0:
        raise ValueError("expected_layers must be positive")
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    captured = []

    def interceptor(next_fun, method_args, method_kwargs, context):
        result = next_fun(*method_args, **method_kwargs)
        module = context.module
        if (
            context.method_name == "_router"
            and hasattr(module, "num_experts")
            and hasattr(module, "num_experts_per_datapoint")
        ):
            router_logits = (
                method_args[0]
                if method_args
                else method_kwargs["router_logits"]
            )
            captured.append(
                jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
            )
        return result

    with nn.intercept_methods(interceptor):
        output = forward(*args, **kwargs)
    if len(captured) != expected_layers:
        raise RuntimeError(
            f"captured {len(captured)} router calls, expected {expected_layers}; "
            "wrap exactly one encoder or decoder forward"
        )
    flattened = [
        probabilities.reshape((-1, probabilities.shape[-1]))
        for probabilities in captured
    ]
    token_counts = {probabilities.shape[0] for probabilities in flattened}
    expert_counts = {probabilities.shape[1] for probabilities in flattened}
    if len(token_counts) != 1 or len(expert_counts) != 1:
        raise RuntimeError("router probability shapes differ across layers")
    return output, jnp.stack(flattened, axis=1)


def call_with_router_artifacts(
    forward,
    *args,
    expected_layers: int,
    **kwargs,
):
    """Capture mass plus the bounded reconstruction-gate inputs for one forward.

    Returns ``(output, artifacts)`` where artifacts contains:

    - ``probabilities[tokens,layers,experts]``: plain pre-top-k f32 softmax;
    - ``shared_pre_router_residual[tokens,layers,hidden]``: the unnormalized
      residual supplied to each MoE branch;
    - ``topk_indices[tokens,layers,k]``;
    - ``final_scaled_topk_weights[tokens,layers,k]``: renormalized selected
      router weights multiplied by ``per_expert_scale``.
    """
    if expected_layers <= 0:
        raise ValueError("expected_layers must be positive")
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    residuals = []
    probabilities = []
    choices_out = []
    final_weights = []

    def is_moe(module):
        return (
            hasattr(module, "num_experts")
            and hasattr(module, "num_experts_per_datapoint")
        )

    def interceptor(next_fun, method_args, method_kwargs, context):
        module = context.module
        if context.method_name == "__call__" and is_moe(module):
            x = method_args[0] if method_args else method_kwargs["x"]
            unnormalized = (
                method_args[1]
                if len(method_args) > 1 and method_args[1] is not None
                else method_kwargs.get("unnormalized_x", x)
            )
            residuals.append(unnormalized)

        result = next_fun(*method_args, **method_kwargs)
        if context.method_name == "_router" and is_moe(module):
            logits = (
                method_args[0]
                if method_args
                else method_kwargs["router_logits"]
            )
            router_probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
            router_weights, choices = result
            selected = jnp.take_along_axis(router_weights, choices, axis=-1)
            expert_scale = module.per_expert_scale.astype(selected.dtype)
            scaled = selected * jnp.take(expert_scale, choices, axis=0)
            probabilities.append(router_probs)
            choices_out.append(choices)
            final_weights.append(scaled)
        return result

    with nn.intercept_methods(interceptor):
        output = forward(*args, **kwargs)
    counts = {
        "residuals": len(residuals),
        "probabilities": len(probabilities),
        "choices": len(choices_out),
        "weights": len(final_weights),
    }
    if set(counts.values()) != {expected_layers}:
        raise RuntimeError(
            f"router artifact calls differ from expected {expected_layers}: {counts}"
        )

    def flatten_and_stack(values, name):
        flattened = [value.reshape((-1,) + value.shape[2:]) for value in values]
        token_counts = {value.shape[0] for value in flattened}
        trailing_shapes = {value.shape[1:] for value in flattened}
        if len(token_counts) != 1 or len(trailing_shapes) != 1:
            raise RuntimeError(f"{name} shapes differ across layers")
        return jnp.stack(flattened, axis=1)

    return output, {
        "probabilities": flatten_and_stack(probabilities, "probability"),
        "shared_pre_router_residual": flatten_and_stack(residuals, "residual"),
        "topk_indices": flatten_and_stack(choices_out, "top-k index"),
        "final_scaled_topk_weights": flatten_and_stack(
            final_weights, "top-k weight"
        ),
    }


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


def phase_coverage_errors(merged: RouterStatsAccumulator) -> list[str]:
    """Validate the exact rev-2 taxonomy and phase/workload coverage."""
    errors: list[str] = []
    counts = {(phase, workload): 0 for phase in PHASES for workload in WORKLOADS}
    for bucket in merged.buckets:
        parts = bucket.split("/")
        if len(parts) not in {2, 3}:
            errors.append(f"malformed calibration bucket {bucket!r}")
            continue
        phase, workload, *position = parts
        if phase not in PHASES or workload not in WORKLOADS:
            errors.append(f"unknown calibration bucket {bucket!r}")
            continue
        if phase == "prefill" and position:
            errors.append(f"prefill bucket must not have a position suffix: {bucket!r}")
            continue
        if position and (phase == "prefill" or position[0] not in {"pos0", "pos1", "pos2"}):
            errors.append(f"invalid calibration position suffix: {bucket!r}")
            continue
        counts[(phase, workload)] += merged.tokens(bucket)
    for (phase, workload), tokens in counts.items():
        if tokens <= 0:
            errors.append(f"missing tokens for {phase}/{workload}")
    return errors


def merge_flushes(
    directory: Path, *, require_phase_coverage: bool = True
) -> RouterStatsAccumulator:
    """Offline fold of every host/shard flush and validate global coverage."""
    paths = sorted(directory.glob("router_stats.host*.shard*.json"))
    if not paths:
        raise FileNotFoundError(f"no router_stats flushes under {directory}")
    merged = RouterStatsAccumulator.from_json(paths[0].read_text(encoding="utf-8"))
    for path in paths[1:]:
        merged.merge(RouterStatsAccumulator.from_json(path.read_text(encoding="utf-8")))
    if require_phase_coverage:
        errors = phase_coverage_errors(merged)
        if errors:
            raise ValueError("merged calibration coverage failed: " + "; ".join(errors))
    return merged


def selection_inputs(merged: RouterStatsAccumulator, layer: int) -> dict[str, list[float]]:
    """Adapter to sunfish.expert_selection.select_experts bucket_mass input."""
    return merged.layer_bucket_mass(layer)
