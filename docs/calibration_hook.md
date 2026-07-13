# Router-calibration hook — interface contract (stage 1)

Owner: Claude (spec). Implementation: first-come per AGENTS.md.
Consumers: `sunfish.router_stats.RouterStatsAccumulator`,
`sunfish.expert_selection`, the stage-1 gate in PLAN.md.
Revision 3 — incorporates Codex review (channel [6]): corrected
`per_expert_scale` semantics, functional-state/flush rules, i32 counts,
bucket arithmetic, masking, reconstruction residual definition, executable
full-teacher runner, fixed-window corpus, and reconstruction promotion gate.

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

The pinned Gemma MoE does not return its pre-top-k logits. The implementation
uses Flax 0.12.7 `intercept_methods` around exactly one encoder or decoder
forward, observes every `MoE._router` input, recomputes the identical f32
softmax, and returns `[tokens, 30, 128]` as an ordinary JIT value. It fails if
the call count is not exactly 30, so a wrapper that invokes the backbone more
than once cannot silently mix phases. Upstream Gemma source is not patched.

**Flush topology:** the executable runner uses the allowed global variant:
GSPMD reduces the data-sharded contribution into one replicated accumulator,
and process 0 alone flushes it. Never let every host flush that global value —
the offline merge would multiply results by host count. The core hook also
retains a per-host-flush mode for isolated tests, but one run must use exactly
one topology.

## Bucket taxonomy

Bucket string = `"{phase}/{workload}"`, with a required denoising-position
suffix for the mass artifact.

- **Phases**: `prefill` (causal encoder tokens), `denoise_high` /
  `denoise_mid` / `denoise_low` (top/middle/bottom thirds of the remaining-
  step schedule at the time of the forward pass).
- **Workloads**: `code_completion`, `repo_edit`, `tool_calls`,
  `agent_trajectory`, `general_control`, `reasoning_control`.
- **Canvas position** (denoise phases only): `/pos0..2` tertile suffix. This is
  mandatory because PLAN.md makes position coverage part of the Stage-1 gate.

Bucket count: **60** (6 prefill + 3 denoise phases × 6 workloads × 3
positions — prefill has no canvas position). The mass array remains below 1
MB. The bounded reconstruction sample uses the coarser 24 phase/workload
buckets; each sampled canvas contains all position tertiles.

## Flush destinations (canonical prefixes — must match infra/gcp layout)

- Aggregates: `gs://<bucket>/sunfish/calib/<run_id>/router_stats.host{H}.shard{S}.json`
- Raw debug sample + reconstruction artifact:
  `gs://<bucket>/sunfish/calib/raw/<run_id>/...` (this prefix carries the
  lifecycle auto-delete backstop; aggregates are tiny and kept).

Offline: load all host/shard JSONs, fold with `.merge()`, feed
`layer_bucket_mass(layer)` into `expert_selection.select_per_layer` with the
weights/floor from `configs/sunfish-8b-a3b.toml [calibration]`.
`sunfish_tpu.calibration.merge_flushes` fails closed unless the merged artifact
contains positive token coverage for all 4 phases × 6 workloads; position
suffixes are validated and aggregated into their phase/workload coverage.

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

Budget ≈ 17 GB (100k × 30 × (2816×2 + 24) bytes). The 100k sample is
stratified across all 24 phase/workload buckets, including prefill.
**Draining is bounded and
asynchronous**: double-buffered device→host copies with host-side GCS
uploads; never a synchronous callback in the forward path, and never more
than 2 batches of artifact resident on device.

The reconstruction job replays each shared residual through the exact pinned
Gemma 4 FFW twice: the full 128/8 teacher and the selected 32/8 candidate. It
recomputes and byte-checks the recorded teacher top-k indices and f16 final
weights before using the sample. The metric covers the complete FFW output
(shared dense branch + routed branch + all post norms), not merely retained
router mass. For every one of 24 buckets × 30 layers, both are binding:

- relative RMSE `sqrt(sum((candidate-teacher)^2) / sum(teacher^2)) <= 0.15`;
- cosine similarity `>= 0.99`.

At least 100,000 total tokens and 4,000 tokens per bucket are required. The
48/8 fallback uses the same normalized output bounds; its router-mass floor is
0.3375. A passing job writes `selection-approved.json` with hashes of the
mass candidate, reconstruction run identity, and full result. Nothing else
sets `promotion_allowed=true`.

The calibration identity, mass candidates, reconstruction identity, and
approved selection all carry the same launcher-verified Git commit and
source-tree SHA-256. Reconstruction refuses a candidate or calibration
artifact from a different code tree, preventing an approval from silently
crossing implementations.

## Executable Stage-1 path

Assemble the full corpus off Chase's laptop. Documents are split into fixed
768-token records before writing, so the manifest's 75M tokens are the tokens
the TPU runner actually consumes. The total is allocated with the exact
workload shares in `docs/data.md`:

```bash
python scripts/assemble_calibration.py \
  --tokenizer /path/to/tokenizer.json \
  --output /scratch/sunfish-calibration-75m \
  --total-tokens 75000000 \
  --record-tokens 768
```

After the Stage-0.5 ledger passes, launch the full 128-expert collector on all
workers (replace every placeholder with immutable GCS paths/digests):

```bash
export TEACHER_CHECKPOINT=gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it
.venv-tpu-controller/bin/sunfish-gcs-inventory \
  --uri "$TEACHER_CHECKPOINT" --anonymous \
  --output /tmp/diffusiongemma-teacher-inventory.json
export TEACHER_REVISION="gcs-inventory-sha256:$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' \
  /tmp/diffusiongemma-teacher-inventory.json)"
export STAGE05_LEDGER="$SUNFISH_READINESS/stage05-readiness-ledger.json"
export STAGE05_LEDGER_SHA256="$(.venv-tpu-controller/bin/python -c \
  'import hashlib,sys; from etils import epath; print(hashlib.sha256(epath.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "$STAGE05_LEDGER")"

scripts/launch_tpu_pod.sh \
  --run-id sunfish-calibration-v1 \
  --attempt-id sunfish-calibration-v1-001 \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-calibrate \
  --source-checkpoint "$TEACHER_CHECKPOINT" \
  --source-revision "$TEACHER_REVISION" \
  --source-anonymous \
  --readiness-ledger "$STAGE05_LEDGER" \
  --readiness-ledger-sha256 "$STAGE05_LEDGER_SHA256" \
  --data-directory gs://YOUR_BUCKET/sunfish/data/calibration-75m \
  --data-manifest-sha256 CALIBRATION_MANIFEST_SHA256 \
  --output-dir gs://YOUR_BUCKET/sunfish/calib \
  --raw-output-dir gs://YOUR_BUCKET/sunfish/calib/raw \
  --run-id sunfish-calibration-v1 \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES" \
  --min-source-tokens 75000000 \
  --min-coverage 0.225 \
  --fallback-min-coverage 0.3375 \
  --reconstruction-tokens 100000
```

The calibration runner revalidates every ordered gate, evidence hash,
topology, source identity, config-bundle pin, and Stage-0 parity pin before it
loads the teacher. It records the ledger path and byte hash in
`calibration-run.json`; reconstruction re-reads the same ledger and refuses an
edited, replaced, failed, or different-topology receipt.

If the 32E mass candidate passes, run its reconstruction gate. If it fails,
use `selection-mass-candidate-48e.json` only when that fallback's mass gate
passed:

```bash
scripts/launch_tpu_pod.sh \
  --run-id sunfish-reconstruction-32e-v1 \
  --attempt-id sunfish-reconstruction-32e-v1-001 \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-reconstruction-gate \
  --source-checkpoint "$TEACHER_CHECKPOINT" \
  --source-anonymous \
  --calibration-dir gs://YOUR_BUCKET/sunfish/calib/sunfish-calibration-v1 \
  --raw-dir gs://YOUR_BUCKET/sunfish/calib/raw/sunfish-calibration-v1 \
  --candidate gs://YOUR_BUCKET/sunfish/calib/sunfish-calibration-v1/selection-mass-candidate.json \
  --output-dir gs://YOUR_BUCKET/sunfish/calib/reconstruction \
  --run-id sunfish-reconstruction-32e-v1 \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES" \
  --max-relative-rmse 0.15 \
  --min-cosine-similarity 0.99 \
  --min-total-tokens 100000 \
  --min-tokens-per-bucket 4000
```

This approval is for the storage-pruning rung (32/8 or 48/8) only. Its
`top_k_experts=8` is enforced by both converters and seed materialization; it
cannot initialize a 32/4 run until the later top-k ablation emits a new
approved sidecar for that rung.

## Sanity invariants (assert at flush)

1. `sum(mass[b, l, :]) / tokens[b] ∈ [0.99, 1.01]` for every bucket b, layer
   l with `tokens[b] > 0`.
2. Every workload bucket has tokens in every phase.
3. Merged totals equal the sum of per-host flushes exactly (merge is
   addition; any mismatch means a double-psum/double-flush bug).
