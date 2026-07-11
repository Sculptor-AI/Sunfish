# Router-calibration hook — interface contract (stage 1)

Owner: Claude (spec). Implementation: first-come per AGENTS.md.
Consumers: `sunfish.router_stats.RouterStatsAccumulator`,
`sunfish.expert_selection`, the stage-1 gate in PLAN.md.
Revision 2 — incorporates Codex review (channel [6]): corrected
`per_expert_scale` semantics, functional-state/flush rules, i32 counts,
bucket arithmetic, masking, and the reconstruction residual definition.

## What the hook measures

For every MoE layer, the **plain 128-way router softmax, before top-k
truncation**. Two scaling facts matter and were previously misstated:

- `router.scale` rescales the normalized hidden state *before* the router
  projection, so it already shapes the softmax — nothing extra to apply.
- `router.per_expert_scale` is applied only **after** softmax → top-k →
  renormalization, to the selected experts' weights. It does not affect
  ranking and must NOT be folded into the mass artifact.

So: the **mass artifact** logs the pre-truncation softmax; the
**reconstruction artifact** (below) uses the final scaled top-k weights,
because those are what the pruned model must reproduce.

## On-device accumulation (no per-token I/O)

Functional JAX state threaded through the jitted step (donation is fine as
long as the updated accumulator is returned):

- `mass: f32[num_buckets, 30, 128]` — summed router probabilities
- `tokens: i32[num_buckets]` — token counts (i64 silently degrades to i32
  under default `jax_enable_x64=False`; use i32 on device, bound tokens per
  flush well below 2^31, and widen to Python ints on the host)

**Masking:** batches are not assumed bucket-homogeneous. Updates are masked
segment-sums keyed by each token's bucket id; padding tokens carry bucket id
-1 and contribute nothing to mass or counts.

**Flush topology (pick per-host, not global):** each host sums its own
addressable shards and flushes its local accumulator; the global total comes
from the offline `RouterStatsAccumulator.merge` fold. Do NOT `psum` globally
and then let every host flush — merged results would be multiplied by host
count. (If a global `psum` is ever preferred, then process 0 alone flushes.)

## Bucket taxonomy

Bucket string = `"{phase}/{workload}"`, plus optional position suffix.

- **Phases**: `prefill` (causal encoder tokens), `denoise_high` /
  `denoise_mid` / `denoise_low` (top/middle/bottom thirds of the remaining-
  step schedule at the time of the forward pass).
- **Workloads**: `code_completion`, `repo_edit`, `tool_calls`,
  `agent_trajectory`, `general_control`, `reasoning_control`.
- **Canvas position** (denoise phases only, flag-gated): `/pos0..2` tertile
  suffix. Off by default.

Bucket count: 24 without positions; **60** with positions (6 prefill +
3 denoise phases × 6 workloads × 3 positions — prefill has no canvas
position). Allocate exactly; the mass array is ≤1 MB either way.

## Flush destinations (canonical prefixes — must match infra/gcp layout)

- Aggregates: `gs://<bucket>/sunfish/calib/<run_id>/router_stats.host{H}.shard{S}.json`
- Raw debug sample + reconstruction artifact:
  `gs://<bucket>/sunfish/calib/raw/<run_id>/...` (this prefix carries the
  lifecycle auto-delete backstop; aggregates are tiny and kept).

Offline: load all host/shard JSONs, fold with `.merge()`, feed
`layer_bucket_mass(layer)` into `expert_selection.select_per_layer` with the
weights/floor from `configs/sunfish-8b-a3b.toml [calibration]`.

## Secondary artifact — reconstruction-gate sample

The stage-1 gate requires a layer-output reconstruction check. For a bounded
subsample (~100k tokens, stratified across buckets), log per token and layer:

- the **shared pre-router residual** — the hidden state from which BOTH the
  router branch (its norm + scale + projection) and the expert branch can be
  re-run. A post-norm or post-expert input is insufficient: after pruning,
  replacement experts and renormalized routings must be recomputed from this
  residual plus checkpoint weights.
- the original top-8 expert indices (u8) and **final scaled top-k weights**
  (f16, i.e. after renormalization and `per_expert_scale`).

Budget ≈ 17 GB (100k × 30 × (2816×2 + 24) bytes). **Draining is bounded and
asynchronous**: double-buffered device→host copies with host-side GCS
uploads; never a synchronous callback in the forward path, and never more
than 2 batches of artifact resident on device.

## Sanity invariants (assert at flush)

1. `sum(mass[b, l, :]) / tokens[b] ∈ [0.99, 1.01]` for every bucket b, layer
   l with `tokens[b] > 0`.
2. Every workload bucket has tokens in every phase.
3. Merged totals equal the sum of per-host flushes exactly (merge is
   addition; any mismatch means a double-psum/double-flush bug).
