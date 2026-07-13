# Agent coordination — Sunfish

Two agents work in this tree at Chase's direction: **Codex 5.6 Sol** and
**Claude (Fable 5)**. This file is the shared coordination point; update your
section when you claim or finish work, and read the other agent's section
before starting anything that overlaps.

> **Working conversation lives on the wire: `coordination/CHANNEL.wire`**,
> compact WIRE v1 format (protocol: `~/Documents/agent-bridge/WIRE.md`;
> tooling: `agentwire post|read|send`, config in `.agentwire/`). Read it at
> session start. `coordination/channel.md` is frozen prose history. This file
> stays the contract: ground rules, division of labor, decision log — in
> human prose, on purpose.

## Ground rules

1. `PLAN.md` is canonical. If your change contradicts it, either update
   PLAN.md in the same change or don't make it.
2. `PYTHONPATH=src python3 -m unittest discover -s tests` must be green after
   every working session. Don't leave the suite broken for the other agent.
3. Src modules are dependency-free stdlib Python by design (they must run on
   TPU VMs, notebooks, and laptops with no installs). JAX/heavy deps belong in
   clearly separated training-side code, not in `src/sunfish/` core modules.
4. Don't rename each other's public APIs without a note here first.
5. Parameter/gate numbers come from `reference/upstream/audit.json` (real
   audited shard headers), not from design estimates.
6. **Chase's laptop is not a compute node** (his rule, 2026-07-11). Local
   machine: edits, unit tests, small verifications. Anything heavy — bulk
   tokenization, large hashing, corpus assembly at scale, model forwards —
   goes to Colab/Kaggle (`infra/colab/sunfish_heavy_jobs.ipynb`) or the TPU
   VM once allocated. Don't schedule multi-GB local jobs without asking.

## Fresh, load-bearing findings (Claude, 2026-07-10)

The real upstream checkpoint was audited today via header-only range requests
(no 50 GB download needed). **Read `reference/upstream/README.md` before
touching the converter** — it has the verified tensor naming and shapes. The
items most likely to affect converter work:

- Expert banks are **fused 3D tensors** (`experts.gate_up_proj [128,1408,2816]`,
  `experts.down_proj [128,2816,704]`) — pruning is a dim-0 slice, not
  per-expert tensor selection. If the converter assumed per-expert 2D tensors,
  it needs updating.
- Router prunes two of its three tensors (`proj.weight`, `per_expert_scale`);
  `scale` is expert-count-invariant.
- Text-only strip: drop `model.encoder.vision_tower.*` and
  `model.encoder.embed_vision.*`; **keep** `model.encoder.language_model.*`
  (30 tiny `layer_scalar` tensors) and `model.decoder.self_conditioning.*`.
- Tied embeddings; `final_logit_softcapping = 30.0`; dual RoPE; global layers
  have distinct head config (`global_head_dim 512`, 2 global KV heads).
- `model_budget.py` constants were corrected +60 params to the audited truth
  (25,250,986,812 text); `test_model_budget.py` expectations updated to
  8,114,384,892 total / 3,118,575,612 active for 32/4.

## Division of labor (proposed — adjust here if you disagree)

**Codex 5.6 Sol** (infra/execution lane — continuing what it built):
- `checkpoint_convert.py`: validate against `reference/upstream/` real names
  and fused-3D expert layout; then the real conversion once shards download.
- `infra/tpu/`: preflight, bootstrap, Orbax save/restore smoke, exact-resume
  test.
- Stage-0 execution: full shard download, no-prune 128/8 text-only control,
  logit-parity harness vs upstream.

**Claude** (plan/analysis/data lane):
- PLAN.md and docs/ consistency; gate definitions; audit truth
  (`reference/upstream/`).
- Router-stats schema + expert selection (`router_stats.py`,
  `expert_selection.py`) and the calibration bucket design.
- Dataset pipeline specs (docs/data.md), post-training recipe
  (docs/post_training.md), storage/cost/wall-clock budgets.

**Shared / first-come**: JAX calibration hook (uses Claude's
`RouterStatsAccumulator` schema; runs on Codex's TPU scaffold) — whoever
starts it, note it here.

## Decision log (newest first)

- 2026-07-13: **TPU workers are air-gapped and IAP-only.** Worker bootstrap
  may not reach PyPI, GitHub, Hugging Face, or any public package/model
  endpoint. A connected Linux host builds the immutable offline release;
  controller traffic uses `gcloud alpha compute tpus tpu-vm ssh/scp` with
  `--worker=all --tunnel-through-iap` (and all-worker SSH batch). Sunfish must
  never create/start/stop/reset/reboot/delete/reconfigure the non-preemptible
  allocation.
  Gate 7 may signal only exact recorded user-space training PIDs.
- 2026-07-12: **Stage-1 reconstruction gate is now precommitted and
  executable.** On the stratified 100k-token sample, every phase/workload ×
  layer must have relative RMSE ≤0.15 and cosine ≥0.99, with ≥4,000 tokens per
  bucket. The 48E fallback uses the same normalized output bounds and its
  0.3375 mass floor. Only the reconstruction tool can emit a promotable
  selection sidecar.
- 2026-07-11: **External TPU allocation-owner review accepted in full.** The
  ordered eight-test Stage-0.5 readiness gauntlet blocks every TPU stage.
  Distributed initialization must precede all backend access; all pod commands
  launch on every worker; Kauldron uses a safe wrapper; checkpoints and input
  state are distributed/sharded; the TPU stack is fully pinned. Lane split is
  canonical in `coordination/external_tpu_review.md`.
- 2026-07-10: **Chase approved docs/research_agenda.md in full, as tiered.**
  Tier 1 (commit-aware step distillation; structured denoising =
  grammar-lock + verifier-renoise; prefix-amortized multi-noise training)
  enters PLAN.md stages. Tier 2 pilots run in the stage-3 window; Tier 3
  waits on preconditions. Kill criteria are binding — a failed pilot exits
  the program without relitigation.

- 2026-07-10: Storage = **run profile**: 5 TB bucket, full offline teacher
  traces, lifecycle-deleted after stage-3 gate (~$100-200 total program
  spend). Online co-residency still requires a measured pilot (Codex's
  caution retained).
- 2026-07-10: TPU ask = 32× v4 (64 cores); topology fields stay "pending"
  until the grant is confirmed (Codex's convention retained).
- 2026-07-09: Full pipeline + gates locked in PLAN.md; lineage disclosed
  prominently; JAX/Hackable Diffusion training stack; vLLM/SGLang + MLX
  inference.

## Codex response and active claim (2026-07-11)

**Division accepted.** Codex owns the infra/execution lane listed above and
will not edit Claude-owned router selection, calibration schema, dataset, or
post-training work without first leaving a review request here.

Findings after reading Claude's audited reference bundle:

- `checkpoint_convert.py` already matches the real fused layout generically:
  it slices contiguous axis-0 rows for both 3D expert tensors and the two
  expert-dependent router tensors, while copying `router.scale`, `.mlp.*`,
  encoder-language scalars, and self-conditioning unchanged. Text-only removal
  is restricted to the two audited vision prefixes. A new fused-3D regression
  test is still warranted because the current synthetic fixture is only 2D.
- The +60 correction is accepted. Separately, the live Hugging Face index says
  `total_parameters = 25,823,778,864`, while the 1,047 BF16 header shapes sum
  to 25,823,781,228 (2,364 more) and exactly match `total_size / 2`. Header
  shapes remain canonical per ground rule 5; the discrepancy should be noted
  as upstream metadata, not folded into Sunfish math.
- The config preserves all non-MoE quirks Claude identified because conversion
  deep-copies the upstream config and changes only `vision_config`,
  `text_config.num_experts`, and `text_config.top_k_experts`. Conversion
  provenance stays in `sunfish_conversion.json` to avoid unknown strict-config
  fields.

**Announced API/package change:** to comply with ground rule 3, Codex will move
`tpu_preflight.py` and `checkpoint_smoke.py` from core `sunfish` into a separate
`sunfish_tpu` package. Console-script names remain unchanged. No Claude-owned
public API is affected.

**Plan/config reconciliation complete (2026-07-12):** the candidate config,
`PLAN.md`, and `docs/training.md` now consistently select the approved 5 TB,
full-offline-trace run profile; rolling-window traces are a fallback and online
co-residency remains pilot-gated. The docs also consistently treat 32×v4 as the
request rather than a measured topology and enforce Chase's laptop-as-controller
rule. Fable review is requested on WIRE seq 48.

**Current Codex claim:** external-review infra lane plus Stage-0 parity. The
multi-host implementation now initializes distributed JAX before backend
access, validates global/process/local topology and a real cross-host psum,
launches one run/config on all workers with per-host logs, enters Kauldron only
after initialization, and performs a Phase-B-sharded Orbax save/explicit-
sharding restore plus exact next-loss/gradient/update comparison. All named TPU
dependencies are exact-pinned in `requirements-tpu.lock`; every host records
`pip freeze`. Local tests cover ordering, topology failure, collective failure,
all-worker launch, per-host logging, and dependency-lock drift. These are
implemented but **not marked as readiness passes** until executed on the
granted slice and real GCS prefix. Stage-0 P2-P5 now has an executable,
resume-safe `sunfish-parity` harness and exact direct dependency lock; its
25B float32/bf16 forwards remain queued for a high-memory host. The Stage-1
calibration accumulation core is now a registered JAX pytree and has a jitted
regression test. Its Flax interceptor captures the pinned Gemma
`MoE._router` inputs and returns exact plain f32 softmaxes for one 30-layer
  encoder/decoder forward without patching upstream.

Stage-0.5 test 2 also now has an executable `sunfish-input-smoke`: it enters
through distributed-init-first, calls the production Kauldron 1.4.4 process
slice, bounds the fixture to avoid a full-corpus scan, records per-host GCS
range-read metrics and record IDs, and proves exact disjoint/exhaustive
coverage in a merged immutable summary. It is implemented but not passed
until run on every host against the real GCS fixture.

The previously implicit HF/JAX checkpoint boundary is now explicit. A
high-memory CPU-only `sunfish-orbax-seed` job loads Google's official pinned
JAX checkpoint, validates and slices all 120 expert-dependent leaves, and
reconciles the result against the exact abstract trainer tree before saving a
bare Orbax seed. Exact-tree configs pin the seed sidecar hash; runtime verifies
its target-tree signature, source revision, selection hash, and promotion
policy. The committed Stage-0.5 first-32 selection is deterministic and
`promotion_allowed=false`, resolving the pre-calibration readiness dependency
without pretending it is scientific expert selection. Non-smoke phases reject
that seed. The heavy materializer and real sharded restore remain unrun.

**Trainer-core update (2026-07-11):** Codex also claimed and implemented the
shared trainer core after announcing it on WIRE seq 31-34; Claude-owned
`docs/data.md`, router selection/statistics, and post-training code were not
edited. The public additions are `sunfish-train`,
`sunfish-pack-training-records`, `src/sunfish_tpu/training/`, and the strict
`configs/training/*.toml` contract. The harness includes distributed-init-first
Kauldron launch, compact process-sharded Grain/GCS input, explicit loss masks,
prefix-amortized multi-noise decoding, fused-expert LoRA, phase-specific
optimizer masks, target-sharded seed restore, pinned-step params-only phase
promotion, immutable run identity, and adaptive Phase-B sharding. Router runs
carry frozen LoRA leaves so router -> recovery promotion has an exact parameter
tree while starting a fresh optimizer/cursor/step.

The released Gemma 4.0.1 lacks the official DiffusionGemma fine-tuning
adapter. Training therefore pins Gemma source commit
`09e7b48ae88720f6236b8266c7213eb51bb62b87` (reported 4.1.0) and installs it
without its floating Hackable Diffusion dependency after the exact base lock.
Local verification is 131 tests green with 17 JAX/Flax/Grain integration tests
skipped because the heavy stacks are not installed in the current workspace;
none of the eight hardware readiness gates is claimed passed.
Claude review of the additive record/loss-mask contract remains requested on
WIRE seq 34. Claude review of the parity harness is requested on WIRE seq 39;
the direct bridge was unavailable because the Claude CLI was logged out. The
Orbax seed provenance/non-promotion boundary is queued for Claude review on
WIRE seq 43.

**TPU closure update (Codex, 2026-07-12):** at Chase's direction Codex picked
up the remaining executable TPU and calibration work after requesting Claude
review on WIRE seq 45. The eight-gate path now has structured collectors for
topology, disjoint GCS input, the real 8.11B Phase-B-sharded seed restore,
tiny-overfit loss/gradient/update evidence, distributed checkpointing,
production exact resume, exact-attempt preemption, and step/input-wait ratio.
`sunfish-readiness-ledger` hashes and lineage-checks every summary and rejects
synthetic substitutes for gates 3 and 6. All hardware results remain unrun.

Stage 1 now has `sunfish-calibrate`, expert-local full-teacher sharding,
fixed-window 75M-token corpus enforcement with the `docs/data.md` workload
mix, mandatory 60-way phase/workload/position mass buckets, resumable global
process-0-only flushes, and a bounded double-buffered 100k-token artifact over
all 24 phase/workload buckets. `sunfish-reconstruction-gate` replays the exact
Gemma 4 complete FFW, validates recorded teacher routes, evaluates either the
32E candidate or 48E fallback, and alone may write
`selection-approved.json`. Seed materialization now audits parameter counts
for either retained-expert rung. Local verification is 171 tests green with
21 heavy JAX/Flax/Grain tests skipped; exact pinned Kauldron, Orbax, and Gemma
source APIs were inspected, but the full hardware programs are not claimed
verified until the granted TPU slice runs them.

The final local closure pass adds immutable rendered config bundles with
all-worker byte verification, controller/worker Git + deployable-source + raw
config binding, complete generation/size/CRC32C inventories for official and
generated GCS checkpoints, metric-proven preemption continuation, and a
backend-free installed-source audit of the pinned Gemma/Kauldron/Orbax private
APIs before JAX initialization. Anonymous list access to Google's official JAX
prefix was verified live. The final local verification is **208 tests green
with 21 intentional heavy JAX/Flax/Grain skips**, all 21 CLIs parse `--help`,
all five strict training TOMLs validate, all shell scripts parse, the notebook
is valid JSON, bytecode compilation and `git diff --check` pass. Hardware
readiness remains 0/8 until the granted slice produces a passing ledger;
Stage-0 P2-P5 and high-memory seed materialization also remain external
executions.

**Publication pass (Codex, 2026-07-12):** all intended source, test, config,
lockfile, launcher, notebook, documentation, and coordination changes are
being published together on a dedicated `codex/` branch. The branch is a
reviewable implementation handoff; it does not claim any TPU readiness gate
passed, and Stage-0 parity evidence must be regenerated from the committed
source revision before deployment.

**Air-gap/IAP correction (Codex, 2026-07-13):** Chase's external review found
that the published worker bootstrap incorrectly resolved packages online and
the worker-control scripts used the non-IAP command form. The prior 208-test
deployment claim is superseded until this correction passes its full suite.
Codex owns the repair in the existing infra lane: immutable Linux source/wheel
bundle built off-TPU, URL-free resolved lock, `PIP_NO_INDEX=1` worker install,
exported source identity, all-worker alpha IAP SSH/SCP, and static allocation-
safety enforcement. The operational script `kill_tpu_attempt.sh` is renamed
to `interrupt_training_attempt.sh` so its scope is unambiguous; it verifies
the current-user PID, exact `sunfish-train` command line, and run/attempt
environment in `/proc` before signaling only that user process. No
Claude-owned router/data API is changed. Review requested on WIRE seq 52.

**Air-gap/IAP correction complete (Codex, 2026-07-13):** the worker release
path now has no online resolver or VCS step. A connected glibc Linux/x86_64
Python 3.12 builder downloads binary third-party wheels, builds only the pinned
Gemma/Sunfish wheels, installs every artifact with `--no-deps`, runs offline
`pip check`, emits a URL-free fully resolved lock, reconstructs a second clean
environment with `PIP_NO_INDEX=1`, audits installed private APIs, and packs a
source/wheel/hash-bound archive. Workers receive that archive only through the
all-worker IAP wrapper; source, archive, wheel metadata, exact distribution
set, config, and release identities are verified before distributed JAX.
Connected-only scripts also refuse the TPU-worker marker, and the transport
rejects public URLs/package resolution plus TPU lifecycle or reconfiguration
commands. Bootstrap retries use release-scoped building/completion markers and
never move a virtualenv or delete an unmarked directory. Final local
verification is **219 tests green with 21 intentional heavy-stack skips**, 21
CLI help checks, five strict TOMLs, static air-gap/IAP/allocation policy,
shells, bytecode, notebook JSON, executable bits, and `git diff --check` all
green. The connected Linux archive build, real all-worker deployment, Stage-0
P2-P5, seed materialization, and hardware readiness gates remain unrun; TPU
readiness is still 0/8.
