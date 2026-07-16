# Third independent audit (2026-07-16) — findings relayed by Chase, verified by C

An independent auditor reviewed the whole repo and verified pinned third-party
APIs against the actual packages (optax 0.2.8, kauldron 1.4.4, orbax 0.12.1,
grain 0.2.18, hackable-diffusion 1.0.1, gemma@09e7b48). C has independently
re-verified the repo-side and kauldron-side mechanisms marked ✔ below.

## Blocker 1 — Phase-A HBM: gradients were never budgeted ✔ (C-verified)

- `docs/sharding_plan.md:23` budgets Phase A as 16.2 GB replicated params +
  ~14 GB for activations — no gradient line item. ✔
- kauldron 1.4.4 `train/train_step.py` `_step()` takes `jax.grad` w.r.t. the
  FULL params tree (line ~292) and adds `grads=params_grads` to the metric
  context (line ~323). ✔ (verified in the downloaded 1.4.4 wheel)
- `kauldron_config.py:131-136` `gradient_norm = TreeReduce(Norm(tensor="grads"))`
  reads every grad leaf inside the jitted step, so XLA DCE cannot prune the
  ~16.2 GB bf16 frozen-leaf grad tree. `partial_updates`/`set_to_zero`
  (kauldron_config.py:176,181) masks only the optimizer update, not the grad. ✔
- Static floor ≈ 32.4 GB vs 32 GB/chip HBM on v4 → gate-4 smoke OOMs at first
  step; recovery config (K=4, 262K-vocab encoder+decoder logits) is far worse.
- Fixes in preference order (auditor's, C concurs):
  1. Scope the grad-norm metric to trainable leaves (kontext-filtered Norm over
     lora/router paths) — DCE then prunes frozen grads; smoke fits (~19 GB).
  2. Add upstream-style `nn.remat` wrapper for the recovery phase (upstream
     SFT config wraps the model in remat; `make_gemma_network` does not).
  3. Recovery: consider K=2 or chunked vocab projection for the encoder AR loss.
  4. Heavy hammer: run Phase A under the Phase-B sharded policy.

## Blocker 2 — topology hard-pinned to v4-64 ✔ (repo-side verified)

- All training TOMLs: expected 32/8/4, `global_batch_size = 32`. ✔
- On a v4-32 grant: preflight correctly refuses; re-render gives 16/4/4, but
  batch 32 then = 2 examples/device → doubles every per-device activation
  figure above. Recovery needs `global_batch_size = 16` (or sharding) on v4-32
  even after the metric fix.
- Minor: `configs/sunfish-8b-a3b.toml:60` `baseline_global_devices = 8` is a
  third topology in-tree.

## High — run lifetime tied to the IAP SSH session ✔ (repo-side verified)

- Trainer runs as a child of the `gcloud ... ssh --worker=all` session; tunnel
  drop kills the process group on every worker. Gate 7 makes this survivable,
  but a long recovery run will hit it. Mitigate: setsid/detached worker launch
  with reattach, or controller tmux + ServerAliveInterval + attempt-N budget.
- `infra/tpu/job.env.example:34` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` but
  `tpu_host_entrypoint.sh` exports only SUNFISH_*/EXPECTED_* vars — the value
  never reaches workers. Wire it through or delete it. ✔

## Medium

- Data plane: `EPathShardedRecordSource.__getitem__` does open+seek+read per
  record over GCS (no cache, `per_worker_buffer_size=2`) — fine at smoke scale,
  request-rate-bound at production scale. Gate-8 mandates zero local cache;
  decide whether that stays a production requirement or is a gate-2 purity test.
- `verify_shard_hashes=true` re-hashes every shard per Grain worker at startup;
  correct that production TOMLs set false ✔ (verified: recovery/router false).
- Stage-1 teacher loads 50.5 GB from `gs://gemma-data/...` on every worker —
  a live external runtime dependency in an otherwise fully pinned system.
  Consider staging an inventoried copy into the project bucket; at minimum
  verify worker read access during the gauntlet, not at Stage 1.

## Low

- `TimedPyGrainIterator` (training/data.py:49) is lost after checkpoint
  restore: kauldron's `PyGrainIterator.__kd_ocp_restore_post__` returns a plain
  PyGrainIterator, so input-wait sampling stops post-resume and a gate-8-style
  analysis of a resumed attempt fails. One-line fix: override
  `__kd_ocp_restore_post__` to re-wrap. ✔ (no override exists in-tree)
- First-step compile of the 25B teacher variants and K=4 vmapped LoRA trainer
  may take tens of minutes — do not mistake it for a hang.
- `test_reconstruction_drain.py`: two tests error instead of skip in a
  dependency-free env (etils imported inside the drain thread).
- `optax.apply_if_finite` non-finite counter persists across resumes (by
  design; noted).

## Auditor also verified (no action)

Pinned private API surface all line up at the pinned versions:
`WrappedDiffusionGemmaNetwork` → `{"logits": ...}`, mask_helpers signatures,
`_moe._Weight`, router param names vs ROUTER optimizer mask,
`step_prefix="ckpt"` vs runtime/preemption `ckpt_{step}` paths, Orbax 0.12.1
`commit_success.txt`, `KDMetricWriter.write_step_metrics`,
`DataSourceBase.shard_by_process`, `ds_for_current_process`,
hackable-diffusion loss contract. Grain spawns (not forks) workers.
