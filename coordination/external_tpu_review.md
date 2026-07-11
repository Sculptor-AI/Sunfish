# External TPU review — the canonical pre-window punch list (2026-07-11)

Source: Chase's TRC contact (the allocation owner), reviewing the public
repo. Verdict accepted in full: **this is a research plan + stage-0 tooling,
not yet executable multi-host TPU training infrastructure.** The trainer
layer was always scheduled (PLAN stages 1-7); the review turns it into a
precise punch list and catches three traps we had not seen.

## P0 defects in EXISTING code (fix before anything else)

1. **Init-order bug in `sunfish_tpu/tpu_preflight.py`**: it calls
   `jax.devices()` / computations without `jax.distributed.initialize()`
   first. On a multi-host slice this is not merely incomplete — touching the
   backend first makes later distributed init impossible. Every entrypoint
   must call `jax.distributed.initialize()` before ANY backend access, then
   verify: global device count, process count, unique process_index values,
   local device count, and a real cross-device collective (psum), not 1+1.
2. **Kauldron eager-import trap**: `kauldron.main` calls `jax.devices()` at
   module import time. `python -m kauldron.main` on a pod is unsafe. We need
   a custom launcher that initializes distributed JAX before importing
   anything Kauldron-adjacent.
3. **No all-host launcher**: bootstrap/preflight run in one shell. Multi-host
   requires `gcloud compute tpus tpu-vm ssh --worker=all` (same program,
   same run ID/config/paths, every worker) with per-host logs.

## Architecture corrections (accepted)

4. **Conversion moves OFF TPU nodes** (they are storage-constrained):
   stage 0 conversion runs once on a CPU VM with a persistent disk (or is
   already done locally — it is), converted checkpoint + manifest upload to
   GCS, TPU hosts restore from GCS. TPU workers never hold 50-120 GB local
   copies. (infra/tpu/README's 150 GB local-scratch instruction is
   superseded.)
5. **Process-sharded GCS input pipeline** (to build): tokenize/pack off-TPU
   (exists: scripts/assemble_calibration.py); immutable shards + manifest
   (names, counts, checksums, provenance) in GCS; per-host independent
   streams (Grain + ArrayRecord or equivalent seekable format), sharded by
   process_index/process_count; each host reads only global_batch/process_count;
   static shapes; bounded local cache. Checkpoints must carry FULL loader
   state (manifest hash, seed, epoch, shard sequence, current shard, record
   offset, packing-buffer state) — a scalar cursor is insufficient.
6. **Distributed checkpoint smoke** replaces the local PyTree test: init
   distributed on every host, mesh over all global devices, arrays in the
   intended training shardings, all processes in one Orbax save, restore
   with explicit shardings, compare addressable shards, then resume one real
   optimizer update vs an uninterrupted control.
7. **Explicit mesh + partition specs** for: embeddings/head, attention
   projections, the routed-expert axis, optimizer state, batch axis.
   "Topology pending" in TOML is a placeholder, not a policy.
8. **Pin everything**: jax/jaxlib/libtpu, gemma, hackable-diffusion,
   kauldron, orbax-checkpoint, grain, etils, google-cloud-storage. Lock file
   + pip freeze saved with every run.

## The minimum bar — 8 ordered readiness tests (now PLAN gates)

1. Topology: every host up, global device count correct, one collective
   returns the expected global result.
2. GCS data: every process reads disjoint record IDs; one synthetic epoch,
   no omissions, no duplicates.
3. Sharded load: converted checkpoint into the global mesh WITHOUT
   temporarily replicating the full model per host.
4. Training smoke: overfit a tiny GCS-resident dataset ~100-500 updates.
5. Distributed checkpoint: save/restore sharded model+optimizer+RNG+loader
   state from GCS.
6. Exact resume: next batch, loss, gradients, updated params identical to an
   uninterrupted control after restart.
7. Preemption: kill a real run between checkpoints; automatic recovery, no
   manual GCS cleanup.
8. Input throughput: TPU not starved by GCS reads; local disk bounded.

No v4-64 job runs before all eight pass, in order.

## Lane split

- **Codex (infra lane)**: P0 items 1-3 (preflight rewrite, Kauldron-safe
  launcher, all-host launch scripts), item 6 (distributed checkpoint/resume
  test), item 8 (dependency pinning + lockfile).
- **Claude (plan/data lane)**: item 5 (input-pipeline implementation over
  the existing assembler output + loader-state checkpoint schema), item 7
  (mesh/partition-spec design doc for the MoE), PLAN/runbook reconciliation
  (item 4), readiness-test gate integration.
- **Trainer core** (model init from converted checkpoint, train step,
  optimizer, LoRA/router phases): joint; design doc first, then split by
  module. This is the multi-week core engineering between now and the
  window.
