# Mesh and partition policy (design — review item 7)

Owner: Claude (spec). Implementation lands with the trainer. All shapes from
the verified audit (`reference/upstream/README.md`). Assumes the requested
v4-64 slice: 32 chips (v4 megacore exposes ~1 device/chip), likely 8 hosts ×
4 local devices — **both numbers are verified by readiness test #1, not
assumed**; every mesh below is parameterized on the measured counts.

## Memory ground truth (bf16 weights)

| Model | Total | Expert banks | Dense (everything else) |
| --- | --- | --- | --- |
| Student 32E (8.11B) | 16.2 GB | 11.4 GB (32 × 357 MB) | 4.8 GB |
| Teacher 128E (25.25B) | 50.5 GB | 45.7 GB (128 × 357 MB) | 4.8 GB |

Per-chip HBM (v4): 32 GB.

## Phase A — recovery/SFT/RL with frozen or LoRA base (the main line)

Student replicated, batch sharded. Simplest correct thing first:

- Mesh: `('data',)` = all 32 devices.
- Every parameter: `P()` (replicated; 16.2 GB/device leaves ~14 GB for
  activations, LoRA/router optimizer state, and prefix caches).
- Batch dims: `P('data')`. Loader delivers `global_batch/process_count`
  per host (see `reference/tpu-docs/data-loading.md`).
- LoRA/router trainable states are MBs — replicated with the base.

Rationale: no all-to-alls, no gather latency, trivially correct collectives;
at 3.1B-active the arithmetic intensity is already good. We do not spend
engineering on model parallelism the memory budget doesn't require.

## Teacher (trace generation / online distillation) — must shard

50.5 GB doesn't fit one device. Shard the expert banks only:

- Mesh: `('data', 'expert')` = (8, 4).
- `experts.gate_up_proj [128,1408,2816]`, `experts.down_proj [128,2816,704]`,
  `router.per_expert_scale [128]`: `P('expert', ...)` on the expert dim →
  32 experts/group ≈ 11.4 GB + 4.8 GB dense replicated ≈ 16 GB/device.
- Dense weights, embed, router.proj, self-conditioning: `P()` (replicated).
- Routed dispatch crosses the `expert` axis (all-to-all). Forward-only, so
  the cost is bounded; measured in the phase-2 pilot before committing
  replica counts.

## Phase B — full-parameter unfreeze (evidence-gated)

AdamW fp32 (m, v) + fp32 master ≈ 97 GB of state → FSDP:

- Mesh: `('data',)` = 32; shard params + optimizer along it.
- **The student's expert count equals the device count**: expert tensors
  shard `P('data')` on dim 0 → exactly one expert (357 MB) + its optimizer
  state per device. No padding, no uneven shards.
- Dense 2D weights `[out, in]`: `P('data', None)` on dim 0 (row sharding);
  embed/head `[262144, 2816]`: `P('data', None)`. Norms/scalars replicated.
- Per device: ~3 GB fp32 state + ~0.5 GB owned params + gathered bf16
  working weights layer-by-layer. Comfortable in 32 GB.

## Non-negotiables carried from the external review

- Mesh built AFTER `jax.distributed.initialize()`, from measured global
  devices (`jax.make_mesh`).
- Checkpoint restore specifies TARGET shardings (Orbax abstract state), so
  the same GCS checkpoint loads under phase-A replication or phase-B FSDP —
  and readiness test #3 (sharded load without per-host full replication)
  uses exactly this path.
- No spec in this file survives contact with readiness tests #1/#3 without
  re-measurement; the tests are the authority, this document is the intent.

## Open questions (answered by the gauntlet, not by assumption)

1. v4 megacore: is `jax.device_count()` 32 or 64 on this slice? (Affects
   axis sizes only; specs are count-parameterized.)
2. Host count / local device count (8×4 assumed).
3. Teacher all-to-all cost at (8,4) vs (4,8) — measured in the trace pilot.
