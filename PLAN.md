# Sunfish master plan

**The canonical reference.** Deep dives live in `docs/`; if this file and a
deep-dive disagree, fix the disagreement before proceeding.

## North star

Ship **Sunfish-8B-A3B**: an 8B-total / 3.1B-active text diffusion MoE that is
the best local coding/agentic model of its size — 500+ tok/s on an RTX 5080,
usable on a 5070 or a MacBook, excellent at diffs, tool calls, and multi-turn
agent work. SculptorAI's first full training program.

**Release definition** (all must be true):

1. Beats comparable-size open models on the agentic gate evals (below), not
   just HumanEval.
2. Reproducible speed claim: anyone with a 4090/5080 can run `bench/` and see
   the number we published.
3. Honest lineage: prominently documented as a structurally pruned,
   extensively retrained derivative of Google's DiffusionGemma (Apache 2.0).
   We inherited the diffusion wheel on purpose; the pruning science, recovery
   method, data, and post-training are ours. Own it — it is a good story.

## Lineage and identity

- Architecture and initial weights: `google/diffusiongemma-26B-A4B-it`
  (Apache 2.0). Preserve NOTICE, cite in the model card, acknowledge Google's
  TPU Research Cloud in everything published.
- What is genuinely ours: coverage-constrained expert pruning of a diffusion
  MoE (novel, publishable), the recovery recipe, the data mixes, the
  diffusion-RL post-training, the step-efficiency training, and the release
  engineering. That is a real technical contribution at this size.
- Naming: `SculptorAI/sunfish-8b-a3b` (base) and `-it` (instruct+agentic).
  Reserve the Hugging Face org and names early.

## Pipeline

Stages run in order; each has a gate. **Never debug two interventions at
once** — this rule already appears per-stage below and it is the plan's spine.

### Stage 0 — Conversion and audit (local CPU, $0)

Download upstream checkpoint → `checkpoint_audit.py --list` for real tensor
names → dependency-free streaming converter (raw safetensors byte ranges) →
**no-prune 128/8 text-only control**. The converter is implemented and tested
on synthetic sharded checkpoints; validation against the gated upstream shard
set and the model-level parity gate remain.
**Gate:** control reproduces upstream logits and generations exactly.

### Stage 1 — Router calibration (TPU/notebooks, ~$0)

JAX forward hook → `RouterStatsAccumulator`, bucketed by phase
(prefill / high / mid / low noise) × workload (`docs/data.md` calibration set,
~75M tokens). Also bucket by canvas position.
**Gate:** 32 retained experts hold at least 0.9× their size-normalized uniform
mass baseline in *every* bucket (absolute 0.225 for 32/128), and pass the
layer-output reconstruction gate. Router mass alone cannot approve pruning;
if either gate fails, evaluate 48 experts with its own normalized floor.

### Stage 2 — Pruning ablation ladder (TPU, ~$0)

`expert_selection.py` per layer → 128/8 → 64/8 → 32/8 → 32/6 → 32/4,
zero-shot evals at each rung. Storage pruning and top-k reduction are separate
interventions. Top-4 must earn a measured latency win (it saves only ~19% of
active compute); 32/8 is a legitimate landing spot.
**Gate:** zero-shot pruned model coherent and above small dense baselines.

### Stage 3 — Recovery ("our pre-training", allocated TPU topology, 1-3B tokens)

Staged: router-renorm eval → router-only training → teacher distillation
(router + hidden states; noise levels sampled to match the inference
schedule; **full offline trace store per the run profile in
`docs/training.md`**, lifecycle-deleted after this stage's gate; online
co-residency or rolling-window traces only as measured fallbacks) → LoRA on
attention/shared-MLP/experts under the uniform-state diffusion + encoder AR
objective. Full unfreeze only on evidence.
Corpus per `docs/data.md`: 65% permissive code / 35% general, **enriched with
issue→PR-diff pairs and commit sequences** (GLM's mid-training lesson) so
recovery itself teaches repository causality.
**Gates:** tiny-batch overfit + exact checkpoint resume before scaling;
held-out diffusion loss and task evals improving, per bucket, not just
training loss.

The trainer is built with **prefix-amortized multi-noise batching** (agenda
T1.3: encode each long prefix once, train K≈4 canvas noise draws against
it), and this stage's pilot window also runs the approved Tier-2 ablations:
Missing Expert Replay (T2.0) and the pruning-error family (T2.1: route-mass
transport + residual delta compensation). Kill criteria per
`docs/research_agenda.md`; losers exit without relitigation.

### Stage 4 — Context verification (TPU, cheap)

Verify long-context behavior survives pruning+recovery up to the 32K
deployment target (multi-canvas continuation, repo-context retrieval probes).
Extend with long-context annealing only if probes fail.
**Gate:** no retrieval-probe cliff below 32K.

### Stage 5 — SFT (TPU, 200-500M tokens)

Mix per `docs/data.md`: OpenCodeInstruct + Toucan-1.5M backbones, SWE-smith/
SWE-Lego/R2E-Gym trajectories, CommitPackFT + aider-style edit formats,
thinking-mode control (on/off/budgeted), canvas-boundary packing, full
decontamination. Includes **slot-corruption training** for structured
denoising (research agenda T1.2) so grammar-locked decoding has been trained
for, not bolted on.
**Gate:** edit-format validity and tool-call validity clear preset floors
(set from the stage-3 model's baseline; proposed ≥95% before RL).

### Stage 6 — Rejection sampling / expert iteration (TPU + CPU sandboxes)

Two passes: k rollouts per task in real environments (SWE-smith, terminal,
MCP-simulated), keep verified successes, retrain, difficulty-filter between
passes.
**Gate:** monotone improvement on gate evals; harness (sandboxing, verifiers,
trajectory logging) fully debugged here, before RL needs it.

### Stage 7 — Diffusion RL (TPU + CPU sandboxes)

coupled-GRPO objective, StableDRL-style clipping, group size 8-16 on the
difficulty-filtered pool (~20-50k hard tasks; DiffuCoder scale). Reward stack
in trust order: execution → edit-format validity → static analysis →
**step-efficiency bonus** (correct canvases that stabilize in fewer denoising
steps — trains the speed claim directly). No learned reward models in v1.
**Gate:** gate evals up AND mean denoising steps down; kill-switch reverts to
stage 6 checkpoint on reward collapse.

### Stage 8 — Cross-stage distillation polish (TPU, cheap)

On-policy distillation from the best stage 6/7 checkpoints into the release
candidate (GLM's final-pass trick). Consolidates gains, smooths regressions.

### Stage 8b — Commit-aware step distillation (TPU, sampler rollouts)

Research agenda T1.1 (approved 2026-07-10; convergent Claude/Codex
proposal): distill multi-step denoising transitions — tokens, accepted
mask, and self-conditioning state — into single steps, halving 16→8 and,
only if gates hold, 8→4. Data-free; costs TPU sampler time only.
**Gate:** each halving keeps gate-tier evals within 1% and commit-confidence
calibration intact; a failed halving ships the previous step count.

### Stage 9 — Quantization-aware training (TPU, short)

Brief QAT pass targeting the shipped formats (int4/NVFP4 experts, int8 head,
fp16 router) so the *quantized* model is the good model — Gemma-3-QAT
precedent. Entropy-selection stability under quantization is verified here.
**Gate:** quantized gate-eval deltas vs bf16 within 1-2%.

### Stage 10 — Export and deployment

- CUDA: vLLM/SGLang (DiffusionGemma support is upstream; our pruned config
  must load with config changes only). NVFP4 + int4 + bf16 artifacts.
- **Structured-denoising sampler** (agenda T1.2): grammar-locked token
  commitment + verifier-triggered selective re-noising ships as the default
  agentic decoding mode, with the plain entropy-bound sampler as fallback.
- Apple: MLX 4-bit conversion; canvas-length and adaptive-stopping tuning on
  device.
- `bench/`: one-command reproducible latency/throughput benchmark.
**Gate:** the release-definition numbers, measured on real 5080/5070/M-series
hardware, through the agent harness.

## Evaluation (fixed before training, frozen at stage 5)

**Cheap tier** (every checkpoint): per-bucket held-out diffusion loss,
HumanEval+, MBPP+, edit-format validity (does the diff apply), tool-call
schema validity, mean denoising steps per task family.

**Gate tier** (stage transitions + release): BigCodeBench, LiveCodeBench
(post-cutoff window), Aider polyglot, SWE-bench Verified 100-instance subset
through the agent harness, BFCLv3 + MCP-Universe (held out, never trained
on), "formatting under pressure" suite (long diffs, nested JSON, mixed
prose+code canvases), end-to-end 5080 latency.

**Safety tier** (release): standard safety evals on the -it model; verify
pruning/retraining did not strip upstream safety behavior; document in the
model card.

## Release engineering (what makes it a release, not a checkpoint)

1. **Artifacts**: base + -it; bf16, NVFP4, int4; MLX community weights;
   intermediate research checkpoints (post-prune, post-recovery) for
   reproducibility credibility.
2. **The demo**: diffusion's canvas visibly denoising is the single best
   marketing asset we have — a web demo + video of code materializing in
   parallel at 500+ tok/s does more than any benchmark table. Build early,
   use for debugging throughout.
3. **Day-one integrations**: tested configs + recommended sampler presets for
   aider, Cline, OpenHands, Continue; an "agentic preset" (canvas size,
   entropy bound, thinking budget) per tool.
4. **Tech report**: the pruning methodology + ablation ladder is a real
   paper. Publish it (arXiv + blog) with negative results included; TRC
   acknowledgment; this is also what makes TRC renewals easy.
5. **Reproducible claims**: publish the eval harness, decontamination report,
   and `bench/`. Speed claims name the exact hardware, quant, and sampler
   settings.
6. **Model card honesty**: lineage, license, data sources with licenses,
   known failure modes, safety eval results.
7. **v1.1 pipeline ready at launch**: vocabulary-pruned "turbo" variant
   (262K→~80K head, ~20%+ per-step compute cut) and community feedback
   triage — a release plan that includes the *next* release.

## Cost and infrastructure

TPU Research Cloud (access/topology pending confirmation) for all training;
CPU VM + Docker for
rollout sandboxes; RunPod fallback ($350-1,500 worst case) if TRC lapses;
RTX 5080 + M-series for deployment truth. Details: `docs/training.md`.
Standing rules: run `infra/tpu/README.md` preflight on the actual topology;
Orbax→GCS checkpoints every 30-60 min, tested exact-resume, off-device metrics
(W&B), tokenize-once data staging in GCS.

## Top risks (full list: `docs/architecture.md`)

1. Experts less redundant in denoising than prefill traces suggest →
   caught at stage 1 gate; fallback is 48 experts (still fits everywhere).
2. Diffusion RL instability → stage 6 rejection sampling carries most of the
   gain if stage 7 must be abandoned; kill-switch defined.
3. Quantization destabilizes entropy selection → stage 9 QAT + high-precision
   router/head; caught before release, not after.
4. Speed claim shortfall on consumer cards → step-efficiency training +
   turbo variant; publish measured numbers, never projected ones.
5. Lineage criticism → pre-empted by owning it (see Lineage and identity).

## Document map

- `PLAN.md` — this file: the whole program, stages, and gates
- `docs/research_agenda.md` — approved novel approaches, tiers, kill criteria
- `docs/architecture.md` — pruning design, parameter math, alternatives, risks
- `docs/upstream_checkpoint.md` — audited upstream metadata and tensor contract
- `docs/training.md` — infrastructure, phase costs, checkpoint hygiene
- `docs/data.md` — calibration/recovery/SFT mixes, decontamination, evals
- `docs/post_training.md` — RS/RL recipe, rewards, environments
- `configs/sunfish-8b-a3b.toml` — the candidate, machine-readable
- `src/sunfish/` — audit, router-stats, selection, sampler-oracle, budget tools
