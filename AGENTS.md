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

**Plan/config reconciliation requested from Claude:** the decision log now
selects a 5 TB/full-offline-trace run profile, but
`configs/sunfish-8b-a3b.toml` currently says 1 TB and rolling-window traces,
and `docs/training.md` still contains the lean profile. Since Claude owns those
budgets/docs, please reconcile all three with `PLAN.md` rather than having
Codex guess which storage policy Chase approved. Codex agrees that online
co-residency stays pilot-gated either way.

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
granted slice and real GCS prefix. Stage-0 P2-P5 parity remains queued for a
host with sufficient RAM. JAX calibration hook remains unclaimed.
