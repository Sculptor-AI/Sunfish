# Sunfish architecture decision

Status: proposed, pending router calibration and M5 benchmarks.

## Objective

Build a coding- and agentic-task model that prioritizes low latency for one
user on consumer hardware. Two deployment targets, in priority order:

1. **RTX 5080 (16 GB VRAM, CUDA)** — the primary latency target. Upstream
   diffusion speedups are demonstrated on Blackwell (700+ tok/s on a 5090),
   so this target validates quality and latency without also betting on
   Apple's compute-to-bandwidth ratio. The pruned 8B at 4-bit (~4.1 GB) fits
   with wide headroom; the unpruned 26B at 4-bit (~13 GB) barely loads with
   no KV room, so pruning is what makes this card viable at all.
2. **Base M5 MacBook Pro (24 GB unified memory, MLX)** — the secondary,
   riskier target, subject to Google's Apple Silicon caveat.

On either device the deployment model must leave enough headroom for the
operating system, an agent harness, repository indexes, and tool processes.

## Upstream mechanism being retained

DiffusionGemma combines several mechanisms that must be treated as one system:

1. A causal encoder prefills the prompt into a KV cache.
2. A 256-token canvas begins as uniform random vocabulary noise.
3. A decoder using the same backbone attends bidirectionally within the canvas
   and reads the cached prefix.
4. Self-conditioning feeds a soft embedding of the prior step's predictions
   into the next denoising step.
5. Entropy-bounded selection commits confident tokens while uncertain tokens
   are re-noised, with adaptive stopping when predictions are confident and
   stable.
6. A completed canvas is causally appended to the prefix cache before the next
   canvas begins.

The released model has 30 layers, hidden size 2816, expert intermediate size
704, 128 routed experts plus one always-active shared expert per layer, 8
active routed experts, a 262,144-token vocabulary, and a 256-token canvas.
The shared expert is not part of the routed bank: in the parameter audit it
lives inside the expert-count-invariant 2.402B figure, and it is excluded from
every pruning intervention.

## Decision: prune the trained diffusion MoE

Each routed expert contains a gated up projection, an up projection, and a down
projection. Across all layers, one expert therefore contributes:

```text
3 * hidden_size * expert_intermediate_size * num_layers
= 3 * 2816 * 704 * 30
= 178,421,760 parameters
```

An audit of the released shard headers gives 2,402,099,772 parameters outside
the sparse expert banks and expert-count-dependent routers. Accounting for the
smaller router as well as the expert banks yields:

| Routed experts | Active experts | Approx. total | Approx. active |
| ---: | ---: | ---: | ---: |
| 128 | 8 | 25.251B | 3.840B |
| 48 | 4 | 10.970B | 3.120B |
| 32 | 4 | 8.114B | 3.119B |
| 24 | 4 | 6.686B | 3.118B |
| 16 | 2 | 5.258B | 2.760B |

The table excludes the separate vision tower. Actual checkpoint parameter
counts remain the source of truth once conversion exists.

The first named candidate is **Sunfish-8B-A3B**: 32 routed experts per layer,
top-4 routing, text-only. Expert subsets may differ by layer.

## Expert selection

Selection must cover both modes of the shared backbone. Router statistics will
be collected independently for:

- causal prompt/incremental-prefill tokens;
- high-noise, middle, and low-noise denoising steps;
- code completion and generation;
- repository editing and tool-call trajectories;
- a smaller general instruction/reasoning control set.

The initial selector will maximize retained router probability mass with a
size-normalized coverage constraint for every phase and workload bucket. A
load-balanced router's baseline retained mass is `retained / source`, so the
initial floor is 0.9× that baseline (0.225 for 32/128; 0.3375 for 48/128), not
an infeasible fixed 0.5. Frequency-only top-N selection is a baseline, not the
final method. Activation-output similarity and representative-expert selection
will be evaluated before any weight averaging, since independently learned MLP
neurons are not safely averageable without alignment. Layer-output
reconstruction error after rerouting is a separate, decisive gate: aggregate
router mass alone cannot approve a candidate.

The safe ablation path is 128/8 (conversion control), 64/8, 32/8, 32/6, then
32/4. Storage pruning and active-compute pruning are separate interventions and
must not be debugged simultaneously.

Top-k reduction has an asymmetric risk/reward profile that the ablation must
price in. At top-8 the routed experts contribute about 1.427B of the 3.840B
active parameters (~37%); dropping to top-4 removes ~0.713B, only ~19% of
active compute, while concentrating each token's computation into half as many
expert pathways. Storage pruning (128 → 32) is what makes the model fit; top-4
must earn its place with a measured latency win on the M5, not an assumed one.

## Recovery training

Recovery is staged to keep cost and causal diagnosis manageable:

1. Re-normalize the retained router and evaluate without training.
2. Train router parameters while experts and shared backbone are frozen.
3. Distill teacher router behavior and selected hidden states, using offline
   teacher traces so the 26B teacher and student need not share one GPU.
4. Add LoRA to attention, shared MLP, and selected experts while optimizing the
   uniform-state diffusion objective and encoder autoregressive loss.
5. Only if required by evidence, unfreeze selected weights with sharded full
   training.

Coding and agentic SFT follows recovery; it is not used to hide a broken model
conversion.

## Deployment experiments

The faithful 262K-vocabulary, 256-token-canvas model comes first. After quality
parity is understood, optimize the two likely Apple bottlenecks independently:

- canvas lengths 64, 128, and 256 with matched adaptive-stopping settings;
- a 64K-96K code/English vocabulary retaining control tokens and byte fallback.

Vocabulary reduction is potentially valuable because every denoising pass
projects every canvas position over the vocabulary. The embedding/output
matrix is 262,144 × 2816 ≈ 738M parameters — roughly 19% of active parameters
at top-8 and 24% at top-4, and unlike the experts its share grows as the MoE
shrinks. Each denoising step also pays the full 256-position × 262K logit
projection plus the softmax needed for entropy-based selection, so a 64K-96K
vocabulary cuts that term by roughly 3-4×. It is deliberately deferred because
it changes tokenization and embedding/head weights at the same time as expert
pruning.

Two macOS-specific constraints bound the deployment experiments. By default
macOS caps GPU-wired memory at roughly two-thirds to three-quarters of unified
memory (about 16-18 GB on a 24 GB machine, adjustable via
`sysctl iogpu.wired_limit_mb`), so the faithful 4-bit 25.25B baseline
(~12.6 GB of weights) fits but leaves little headroom for KV cache — the gate-1
benchmark should cap context length rather than advertise the 256K maximum.
And because entropy-bounded selection and adaptive stopping consume calibrated
probabilities, quantization should keep the router weights and the output head
in higher precision than the expert banks before concluding that 4-bit
destabilizes token selection.

## Alternatives kept alive

### Convert Gemma 4 E2B/E4B from autoregression

This offers a naturally small backbone but requires learning the diffusion
behavior that the pruned route inherits. It remains a fallback if expert
pruning cannot retain quality.

### Train a small dense diffusion model from scratch

Useful as a correctness control and research platform, but not the shortest
path to a capable coding agent under the available compute budget.

### Distill into a new custom student

Potentially the best final architecture, but only after the pruned checkpoint
provides a cheaper teacher and validates the Apple inference thesis.

## Go/no-go gates

- No-prune checkpoint rewrite must reproduce upstream logits and generation.
- Router calibration must show that 32 experts retain enough probability mass
  across every phase, or the candidate expands to 48 experts.
- The zero-shot pruned model must remain coherent and measurably above small
  dense baselines before recovery training.
- Paid training starts only after a tiny batch can overfit and resume exactly
  from a checkpoint.
- Scaling from a pilot to the full recovery run requires an improving held-out
  diffusion loss and target-task evaluation, not training loss alone.
- The final choice is made on the target consumer devices (RTX 5080 first,
  M5 second) using end-to-end agent latency, peak memory, denoising steps, and
  task success—not datacenter-GPU tokens/second alone.

## Principal risks

1. Experts may be less redundant during denoising than causal router traces
   suggest.
2. Top-4 routing can erase quality even if 32-expert storage pruning works.
3. The 262K output projection may dominate M5 compute after MoE pruning.
4. Quantization error can destabilize entropy-based token selection.
5. Free notebook GPUs are suitable for smoke tests and calibration, but their
   variable hardware and session limits make them unsuitable for a dependable
   full recovery run.
6. Google warns that Apple Silicon's compute-to-memory-bandwidth ratio may not
   show the diffusion speedups measured on dedicated NVIDIA GPUs; the M5
   baseline can falsify the deployment thesis before training begins.
