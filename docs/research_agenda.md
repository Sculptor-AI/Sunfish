# Sunfish research agenda — novel training & inference approaches

Provenance: dual independent ideation (Claude C1-C7, Codex X13-X18, wire seq
13-18, 2026-07-10), deliberately unanchored, then reconciled. Two proposals
were **invented independently by both models** — treated as the highest-
confidence bets and listed first.

Every item carries a kill criterion. Ideas are adopted into PLAN.md stages
only after their pilot survives it; this file is the agenda, not a promise.

## Tier 1 — adopt (high confidence, large win)

### T1.1 Commit-aware step distillation  *(convergent: C1 ≈ X17)*
Distill multi-step denoising transitions into single steps: the teacher (or
the recovered student as its own teacher) runs 2-4 denoise transitions —
tokens, accepted mask, AND self-conditioning state — and the student learns
to reproduce the endpoint in one step. Iterate 16→8→(4). Uniform-state
diffusion's re-noising makes the transition operator well-defined and
compressible; the joint commit-mask prediction is what makes this different
from image-diffusion step distillation.
- Cost: pure TPU sampler rollouts (our free resource), ~2× trace pilot.
- Expected: 1.45-2× decoding throughput at 16-22 effective steps (Codex,
  conservative); stretch 2-4× if 8→4 survives quality gates (Claude).
- Kill: any halving that drops gate-tier evals >1% or mis-calibrates
  commit confidence.
- Stage: new stage 8b, after distillation polish, before QAT.

### T1.2 Structured denoising: grammar locks + verifier re-noising  *(convergent: C2 ≈ X18)*
Two complementary halves of one decoding system, trained for in SFT/RL:
- **Lock** (Codex): tokens that formal grammar FORCES (JSON punctuation,
  diff markers) are parser-committed rather than entropy-committed;
  semantic slots stay entropy-controlled. Train with slot corruption.
- **Re-noise** (Claude): when a stabilized canvas fails its verifier (schema
  parse, diff apply, shell syntax), selectively re-noise only the offending
  span and continue denoising. On parser conflict from a locked prefix,
  unlock the affected span (Codex's fix to Claude's half).
AR models can do neither; this is diffusion-native error correction pointed
at exactly what an agentic model must never get wrong.
- Cost: CPU-only data generation; sampler engineering.
- Expected: invalid-action rate cut 30-60% before RL (X18), approaching
  ~100% validity with re-noising; 10-25% step cut on structured outputs.
- Kill: if locking measurably constrains semantic quality on held-out
  agentic evals, ship re-noising alone.
- Stage: sampler work anytime; corruption training in stage 5; RL reward
  already aligned (edit-format validity).

### T1.3 Prefix-amortized multi-noise training  *(C3)*
Agentic examples are long-prefix + short-canvas. Encode each prefix once;
train K≈4 noise-level draws of the canvas against the cached prefix states.
- Expected: 2.5-4× effective training throughput on agentic data; also
  applies to trace generation. Compounds with T1.3.
- Kill: none (pure throughput); verify gradient correctness vs unbatched.
- Stage: trainer design, stages 3-7.

## Tier 2 — pilot-gated (cheap pilots in the stage-3 window)

### T2.0 Missing Expert Replay  *(X14 — demoted from Tier 1 by Codex's own
correction, wire seq post-18)*
Teacher and student share a byte-identical frozen backbone at recovery
start: run the shared backbone once per token and evaluate only the full
teacher router plus omitted expert routes on the same hidden states.
Caveats per Codex: the targets are **layer-local approximations**, not an
exact full-teacher trajectory, and the 63%/35% compute-cut figures are
active-parameter proxies, not measured wall-clock. Periodic full-teacher
anchors handle drift; early-recovery use only.
- Kill: pilot shows layer-local targets recover less than full-teacher KD
  at matched wall-clock.

### T2.1 The pruning-error exploitation family  *(X13 + C4 — one ablation)*
Two flavors of the same rare asset (the pruning error is exactly observable
because student experts are literal teacher copies):
- **Route Mass Transport** (X13): map each removed expert to ~2 retained
  experts by output reconstruction; initialize student router logits from
  aggregated teacher mass. Expected 30-60% lower initial reconstruction gap,
  20-40% fewer recovery steps; refresh mapping once as backbone adapts.
- **Residual delta compensation** (C4): regress a shared-MLP LoRA directly
  on the measured 128-vs-32 output delta before generic KD.
(A third flavor, Ghost Residual X15, was **cut by its own author**: it
overlaps delta compensation while adding permanent inference/kernel
complexity.)
Run as a single ablation family on the stage-3 pilot; keep whatever wins.
- Kill (each): no held-out gain surviving phase buckets.

### T2.2 Variable canvas + canvas-size head  *(C5)*
Train 32/64/128/256 canvases with explicit size conditioning; a tiny head
picks the canvas from the prefix. Most agent-loop outputs are short tool
calls → 2-4× latency cut on actions, multiplicative with T1.1.
- Kill: mode confusion on held-out mixed-length evals.

### T2.3 Halt-head adaptive stopping  *(C6)*
Distill the observed stabilization step into a scalar halt head, replacing
entropy-threshold stopping. Trivial parameters; cuts average steps; stacks
with T1.1/T2.2. Kill: no step reduction at matched quality vs tuned
entropy thresholds.

## Tier 3 — opportunistic (only if their precondition appears)

### T3.1 Annealed routing  *(X16)*: noise-conditioned top-k (top1 at high
noise → top6 at final correction), phase-conditioned mass distillation.
Expected ~9% active-compute cut, 5-10% wall win. Precondition: stage-2
ablations show high-noise routing quality is insensitive. Risk: early
routing errors are irreversible — gate quality per phase bucket.

### T3.2 Agentic expert re-specialization  *(C7)*: re-purpose 2-4
low-mass retained experts as agentic experts via auxiliary routing loss in
SFT. Precondition: post-recovery router stats show persistently underused
experts.

## The compounding story (what Chase should expect if Tier 1-2 hold)

Validity: T1.2 → structurally-guaranteed tool calls and diffs.
Speed: T1.1 (½-¼ steps) × T2.2 (¼-½ canvas) × T2.3 (earlier stop) — the
agent-action latency stack multiplies to a plausible 4-10× over the naive
sampler, which is what turns "700+ tok/s on a 5080" from marketing into a
measured number. Training cost: T1.3 (plus T2.0 if its pilot holds)
multiplies into substantially cheaper distillation/recovery, spending the
compute-rich/cash-poor budget exactly where it's shaped to go.

Tech-report angle: T1.1 + T1.2 together are a publishable story —
"structured, step-distilled uniform-state diffusion for agentic decoding" —
independently derived by two frontier models, which is itself worth a
paragraph in the writeup.
