# Sunfish post-training plan

Status: proposed. This stage begins only after recovery training passes its
gates (`docs/architecture.md`); it makes a healed model excellent, it does not
repair a broken one.

## What the strongest open agentic models actually did

The GLM-5 technical report (arXiv:2602.15763) is the clearest public recipe
for "really good at coding/agentic," and most of it transfers down-scale:

1. **SWE data in mid-training, not just SFT** — ~160B tokens from ~10M
   issue-PR pairs before any instruction tuning. Behavior-shaped data arrives
   early, at pretraining-like scale.
2. **Verifiable environments at scale** — 10k+ executable SWE environments
   across 9 languages with Fail-to-Pass test extraction, plus synthesized
   terminal environments, as RL substrates.
3. **Difficulty filtering** — RL compute concentrated on tasks the current
   model solves only rarely; solved and hopeless tasks are dropped.
4. **Hybrid rewards** — rule-based verifiers first, outcome/generative reward
   models where rules cannot reach, human-authored anchors against
   formulaic-output collapse.
5. **On-policy cross-stage distillation** — earlier SFT/RL stages act as
   teachers for a final cheap refinement pass.
6. **Expert iteration / rejection sampling pipelines** that feed verified
   rollouts back into the SFT corpus.

Items 1, 3, 4, 5, 6 transfer to Sunfish nearly verbatim. Item 2 scales down
via open tooling (below). GLM's asynchronous RL infrastructure does not
transfer — at our scale a simple synchronous rollout loop on one v3-8 plus a
CPU box for sandboxes is enough.

## Sunfish post-training stages

### Stage P1 — SFT (`docs/data.md` mix)

Unchanged, with the dataset upgrades noted there. Adopt GLM's practice of
training explicit thinking-mode control (on / off / budgeted) rather than a
single behavior, so the latency story survives contact with hard tasks.

### Stage P2 — SWE mid-training slice (folded into recovery, phase 5)

GLM's lesson 1 applied to our budget: enrich the recovery corpus's repo slice
with issue → PR-diff pairs and commit sequences (The Stack v2 metadata,
CommitPack raw) so that *recovery itself* teaches repository causality. This
is nearly free — it re-weights data we already planned to use.

### Stage P3 — Rejection sampling / expert iteration

Before any policy-gradient RL: sample k rollouts per task from the SFT model
in real environments, keep verified successes, retrain. Two passes.

- Cheap, stable, and immune to the likelihood-estimation problems that make
  policy-gradient RL on diffusion models delicate.
- Uses the same verifiers RL will need, so the harness gets debugged on the
  easy algorithm first.
- Difficulty-filter between passes (GLM lesson 3): drop tasks the model
  always or never solves.

### Stage P4 — Diffusion RL with verifiable rewards

Policy-gradient RL on diffusion LMs is now practiced but has sharp edges:
sequence log-likelihoods are intractable, so importance ratios are estimated,
and naive GRPO can reward-collapse. The literature to build on:

- **coupled-GRPO** (DiffuCoder, arXiv:2506.20639) — complementary mask noise
  halves likelihood-estimate variance; +4.4% EvalPlus on a 7B diffusion coder
  with only ~21k hard samples. The closest proven analog to Sunfish.
- **StableDRL** (arXiv:2603.06743) — unconditional clipping +
  self-normalization against estimation-noise spikes.
- **DACA-GRPO** (arXiv:2605.16342) — credit assignment across denoising
  steps rather than uniform smearing.
- **IGPO** (arXiv:2509.10396) — inpainting-guided rollouts: constrain part of
  the canvas and explore the rest, a natural fit for edit-format training.
- Static-analysis rewards for diffusion coders (arXiv:2605.17174) — see
  reward stack below.

Default: coupled-GRPO objective, StableDRL-style clipping, group size 8-16,
difficulty-filtered task pool from P3. DiffuCoder's scale (21k hard samples)
suggests this fits a TRC window comfortably.

### Stage P5 — On-policy cross-stage distillation (GLM lesson 5)

Final polish: the P3/P4 checkpoints teach the release candidate with
group-size-1 on-policy distillation. Cheap, and it consolidates gains from
stages that may have partially regressed each other.

## Reward stack (in order of trust)

1. **Execution**: Fail-to-Pass tests in sandboxed environments (SWE tasks),
   unit-test pass rate (function-level tasks).
2. **Edit-format validity**: does the diff apply cleanly; does the tool call
   parse as JSON against the schema. Rule-based, cheap, and directly the
   release gate metric.
3. **Static analysis**: linter/parser score on generated code
   (arXiv:2605.17174). Especially apt for diffusion: a malformed canvas is a
   characteristic failure mode, and static rewards grade failures that never
   reach execution.
4. **Step-efficiency shaping**: small bonus for correct canvases that
   *stabilize in fewer denoising steps*. Upstream showed SFT alone cuts step
   counts on structured tasks; RL can optimize the speed story directly.
   This is the reward that turns "700+ tok/s class" from a sampler setting
   into a trained property.

No learned reward models in v1: at this scale, rule-based verifiers cover the
target behaviors, and reward-model exploitation is a failure class we can
simply not buy.

## Environments and rollout infrastructure

- **SWE tasks**: SWE-smith (open NeurIPS-2025 toolkit; 50k task instances,
  128 repos) as the environment factory; R2E-Gym as a secondary pool.
  Python-first, then the 2-3 next languages by target-user share.
- **Terminal tasks**: Harbor/terminal-bench-style containerized tasks,
  synthesized from seed tasks (GLM's recipe) at hundreds-not-thousands scale.
- **Tool calling**: schema-validated simulated MCP servers; BFCLv3 and
  MCP-Universe as held-out evals, never trained on.
- **Topology**: rollout generation on the v3-8 (batched diffusion sampling is
  exactly what the hardware likes), environment execution on a CPU VM with
  Docker; the two talk over a queue. Sunfish's sandbox bill is a small GCE
  instance, not a cluster.

## Evaluation additions (beyond `docs/data.md`)

- BFCLv3 + MCP-Universe for tool calling.
- Denoising-step distribution per task family, tracked across P3/P4 — the
  step-efficiency reward must show up here or it isn't working.
- A "formatting under pressure" suite: long diffs, nested JSON tool calls,
  mixed prose+code canvases — diffusion-specific failure surfaces that
  autoregressive-model evals under-sample.
