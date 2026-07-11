# Sunfish data plan

Status: proposed. Token budgets are targets; the phase-5 pilot fixes them.

Three datasets serve three different jobs, and mixing them up is the classic
failure mode this document exists to prevent:

- **Calibration** data measures router behavior — it must *span* the target
  distribution, not train anything.
- **Recovery** data repairs pruning damage — it must look like *pretraining*
  data (raw code and text under the diffusion objective), not instructions.
- **SFT** data teaches the target behaviors — code editing, tool calls, agent
  trajectories — and comes last, on top of a healed model.

## Calibration set (~50-100M tokens, phase 1)

Small, broad, and bucket-labeled. Every example carries a bucket tag that the
router-stats hook combines with the phase tag (`prefill`, `denoise_high`,
`denoise_mid`, `denoise_low`) at collection time, matching the buckets in
`docs/architecture.md`:

| Workload bucket | Source | Share |
| --- | --- | --- |
| `code_completion` | The Stack v2 (dedup, permissive) sampled across Python, TS/JS, Go, Rust, Java, C/C++, shell, SQL | 35% |
| `repo_edit` | CommitPackFT (commit-message → diff pairs) | 20% |
| `tool_calls` | xlam-function-calling-60k + Glaive function-calling v2 | 15% |
| `agent_trajectory` | SWE-bench-style trajectories (SWE-Gym / SWE-smith rollouts) | 15% |
| `general_control` | SmolTalk sample | 10% |
| `reasoning_control` | Small math/reasoning sample (e.g. GSM8K-style) | 5% |

The control buckets exist to detect experts that pruning would silently
destroy even though the coding buckets never route to them. In
`expert_selection.select_experts` they enter with low `bucket_weights` but a
real `min_coverage` floor — down-weighted, not unprotected.

## Recovery corpus (1-3B tokens, phase 5)

Objective: the upstream uniform-state diffusion loss plus encoder AR loss on
plain sequences. No chat template, no instructions.

| Slice | Source | Share |
| --- | --- | --- |
| Code | The Stack v2 dedup (permissive licenses only), language-weighted toward the SFT distribution | 65% |
| Code-adjacent text | GitHub issues/READMEs, StackExchange (from The Stack v2 / RedPajama slices) | 10% |
| General web/edu | FineWeb-Edu sample | 15% |
| Math | FineMath or OpenWebMath sample | 5% |
| Long-context repo packing | Multi-file repo contexts assembled from The Stack v2 (trains multi-canvas continuation) | 5% |

Rationale for 65/35 rather than 100/0: recovery repairs *general* routing
damage; a code-only recovery mix risks quietly amputating the general experts
that agent scaffolds still need (planning text, error messages, natural
language reasoning between tool calls).

## SFT mix (~200-500M tokens, phase 6)

Shares are measured by **loss-bearing assistant target tokens/canvases, not
raw packed tokens** — otherwise long repository prefixes and tool outputs
silently determine the mix (Codex, channel [4]). Raw tokens/FLOPs are tracked
separately.

| Loss-bearing SFT slice | Source | Share |
| --- | --- | --- |
| Verified repo-agent trajectories | Hierarchy below | 25% |
| Terminal-agent trajectories | Nemotron Terminal Corpus (~366k execution trajectories), deduped by seed/generator; Terminal-Harbor eval templates disjoint | 10% |
| MCP/function/tool calling | Toucan-1.5M backbone (real MCP servers); Nemotron-SFT-Agentic-v2 subset (dedup vs Toucan) + xlam-60k supplements | 15% |
| Standalone edits / diffs | CommitPackFT + aider-style search/replace and unified-diff formats | 15% |
| Code instructions | OpenCodeInstruct backbone; Magicoder OSS-Instruct supplement | 20% |
| General chat | SmolTalk / Tulu-3 sample (floor, not filler — agents need instruction following and sane failure messages) | 10% |
| Bounded reasoning | Short-form math/logic with capped thinking | 5% |

**Repo-trajectory slice composition** (within the 25%): ~35% SWE-Lego (quality
anchor: 18k validated trajectories, 3k+ repos; adopt its step-level error
masking and easy→hard length curriculum, not just its examples), ~30%
SWE-smith proven-success subset (5k Claude/SWE-agent successes weighted above
the expanded ~26k tail until its filters are reproduced), ~20% Open-SWE-Traces
(207k trajectories, 9 languages, both major harnesses — **audit 500 random
traces first** for executable success, patch faithfulness, leakage, teacher
boilerplate; cap regardless of raw size), ~10% SWE-Gym (small real-task gold
anchor), ~5% verified R2E-Gym successes (its failures go to verifier/
preference data, never imitation loss). Cap any single teacher, repository
family, or harness at ~one third of the slice.

**One canonical action grammar.** Normalize all harness formats
(OpenHands, SWE-agent, ReAct) through an ADP-style intermediate representation
(action, observation, tool schema, call ID, exit status, source harness), then
render 85-90% into Sunfish's single MCP/OpenAI-style grammar — the one the
vLLM tool-call parser will see at inference. 10-15% prompt-tagged native
renderings only if direct harness compatibility becomes a release requirement.
Exact tool syntax is a low-entropy contract a 3.1B-active model should
overlearn; diversity belongs in strategies, tools, observations, errors,
repos, and teachers — not in three competing delimiters for the same action.

The instruction and tool-calling backbones each carry execution- or
schema-verifiable structure (OpenCodeInstruct ships tests; Toucan derives from
real MCP servers), which matters twice: quality filtering at SFT time, and
reuse as reward substrates in `docs/post_training.md`.

Edit formats are a first-class slice, not an afterthought: an agentic coding
model lives or dies on emitting well-formed diffs, and diffusion's
whole-canvas denoising should be *good* at format-constrained output — that is
a hypothesis the evals below test explicitly.

### Thinking-token policy

Upstream supports `<|think|>` reasoning. Sunfish targets latency, so the SFT
mix trains **short-bounded thinking** (a few hundred tokens) rather than
stripping it: keep the capability, cap the budget, and let evals decide
whether thinking-on earns its latency on coding tasks.

### Diffusion formatting (agreed with Codex, channel [4])

- One training example per assistant decision:
  `prefix = system + tool schemas + issue + (action, observation)*`,
  `target canvas = next assistant action or final answer`.
- **Tool observations are prefix-only, always** — they are exogenous state.
  Training the policy to denoise/inpaint observations teaches it to
  hallucinate plausible stdout/test output instead of conditioning on the
  real thing. Observation robustness comes from dropping/truncating/
  corrupting prefix spans while still supervising the next action.
  (Canvas inpainting remains an RL-time technique for *code-edit spans*
  only — IGPO in `docs/post_training.md` — never for observations.)
- Bounded thought + one atomic tool call in the **same** 256-token canvas
  with an explicit end-of-turn marker; up-weight call name, arguments,
  delimiters, and end marker relative to unverifiable thought tokens. Long
  diffs may span canvases via causal multi-canvas continuation; a tool call
  never splits across a canvas boundary.
- Error-step masking (diffusion analogue of SWE-Lego's step masking): keep a
  trajectory's bad action + error observation in later prefixes, but mask the
  bad action's own target loss; supervise the recovery decision. Teaches
  repair without behavior-cloning mistakes.
- Split long targets on 256-token canvas boundaries so block-autoregressive
  generation is actually trained, not just hoped for.
- Sample noise levels to match the inference-time temperature/step schedule,
  not uniformly (same principle as distillation in `docs/architecture.md`).

## Decontamination and licensing

- **Lineage-based decontamination is primary; 13-grams are only an auxiliary
  check** (Codex, channel [4] — n-grams miss paraphrased issues, cherry-picked
  commits, semantically equivalent patches, and the same task rendered
  through a different harness). The lineage filter keys on: repository,
  issue/PR identity, base-commit ancestry, patch hash + normalized AST diff,
  test/failure signatures, and semantic issue similarity. Targets:
  HumanEval/HumanEval+, MBPP/MBPP+, BigCodeBench, LiveCodeBench (post-cutoff
  windows preferred anyway), SWE-bench Verified, Aider polyglot.
- Maintain a **repo-and-time-disjoint sentinel eval set** that stays out of
  every training AND selection loop, alongside the held-out BFCLv3 and
  MCP-Universe.
- Permissive-license-only code (The Stack v2's license filter), keeping
  per-example provenance so a takedown can be honored by re-tokenizing a
  slice rather than retraining.
- Everything staged in GCS as tokenized, canvas-packed arrays — tokenize once,
  train many times; free TRC compute must not be spent re-tokenizing.

## Evaluation suite

Cheap tier (every checkpoint): held-out diffusion loss per bucket,
HumanEval+, MBPP+, a 200-case edit-format validity check (does the diff
apply?), tool-call JSON validity rate.

Expensive tier (candidate gates): BigCodeBench, LiveCodeBench, Aider polyglot
edit benchmark, SWE-bench Verified subset (50-100 instances through the agent
harness), end-to-end tokens/second and time-to-first-canvas on the RTX 5080.

The release claim to defend is "fast *and* usable for real coding agents", so
the edit-format validity and SWE-bench-subset numbers gate the release, not
just HumanEval.
