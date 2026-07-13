# Sunfish training and infrastructure plan

Status: infrastructure scaffold implemented; the handed-off TPU pod is
non-preemptible and must not be lifecycle-mutated, while its exact JAX
topology/counts still require live measurement. Dollar figures are planning
envelopes, not quotes; pilot runs exist to replace them with measurements.

## Framework decision: JAX-first training

Google's official DiffusionGemma fine-tuning recipes ship in Hackable
Diffusion, a JAX toolbox, and the reference diffusion code in
google-deepmind/gemma is JAX as well. Sunfish adopts **JAX + Hackable
Diffusion as the training stack**, with Orbax checkpoints in a GCS bucket and
conversion to safetensors only at export time. This is what makes the TPU
Research Cloud lane below nearly frictionless: the upstream recipes were
written for exactly that hardware. PyTorch appears only on the inference side
(vLLM/SGLang on CUDA).

## Primary training lane: TPU Research Cloud (TRC)

TRC (https://sites.research.google/trc/about/) grants free TPU quota —
typically a mix of on-demand and preemptible v2-8/v3-8 devices, with newer
generations (v4/v5e) sometimes available on request — in renewable ~30-60 day
windows. The TPU time is free; the project pays only incidentals (GCS storage,
checkpoint egress, and any fallback compute; the accepted run-profile storage
envelope is detailed below).

**Planning target: 32×v4 (64 cores); the current pod is non-preemptible, but
its exact granted topology is not inferred from that label.**
`infra/tpu/README.md` makes global device, process, and local-device counts
measured inputs. The eight-gate readiness gauntlet runs on the granted slice
before calibration or training. Smaller Colab/Kaggle TPU sessions remain
useful for isolated warm-up tests, not as evidence for the granted topology.

Fit by phase:

- **Teacher traces (phase 2)** — shard the bf16 26B teacher (~50 GB) across the
  confirmed topology. Only run concurrent replicas after a measured one-replica
  memory/throughput pilot; replica count is not assumed in advance.
- **Router training, distillation, LoRA recovery, diffusion SFT (phases
  4-6)** — fit is measured with the actual mesh. Router-only and LoRA pilots
  should fit an eight-device slice; larger allocations primarily add data
  parallel throughput.
- **Full-parameter unfreeze** — remains both evidence- and topology-gated.
  Optimizer, activation, and sharding memory must be measured before enabling
  it; no unconfirmed HBM budget appears in the plan.

Constraints to plan around:

- Quota is best-effort and window-based; renewals expect research feedback and
  an acknowledgment in anything published. Treat each window as a deadline:
  have data, code, and pilot results ready *before* the window opens.
- Other TRC/fallback quota can be preemptible, so checkpoint hygiene still
  matters. The current scarce pod is non-preemptible: recovery is tested by
  interrupting only a recorded `sunfish-train` user process, never a TPU VM.
- TPUs do not help the deployment story: quantized inference benchmarking,
  entropy-selection stability under int4/NVFP4, and the agent-latency gate
  stay on the RTX 5080 and M5.

## Fallback and supporting lanes

- **Chase's laptop** — edits, unit tests, header-only tensor audits, and small
  verifications only. It is not a compute node.
- **High-memory CPU VM / Colab** — checkpoint conversion, seed materialization,
  bulk tokenization, and other multi-GB jobs. Conversion copies bounded raw
  safetensors byte ranges, but its aggregate I/O still stays off the laptop.
- **Kaggle / Colab free tiers** — router calibration forwards, zero-shot
  pruning evals, and smoke tests; also the JAX warm-up lane (Kaggle's TPU
  v3-8 sessions run the same stack as TRC) while a TRC application is pending.
  Free notebooks are still *not* used for the dependable recovery run
  (architecture risk #5): sessions are capped (~9-12 h) and hardware varies.
- **RunPod rented GPUs** — the paid fallback if TRC quota lapses mid-run or a
  phase needs CUDA specifically. Default instance is a single A100 80 GB
  (~$1.2-1.9/hr); an L40S 48 GB (~$0.8-1.1/hr) covers LoRA-only stages.
- **RTX 5080 workstation** — inference benchmarking, quantized-model evals, and
  the end-to-end agent-latency gate. Not a training device: 16 GB does not fit
  the bf16 student's weights (~16.2 GB).

## Phase map

| Phase | Work | Hardware | Envelope (TRC / fallback) |
| --- | --- | --- | --- |
| 0 | Checkpoint conversion, no-prune control, tensor audit | High-memory CPU VM / Colab | $0-40 |
| 1 | Router calibration traces across phases/workloads | Free notebooks or allocated TPU | $0 / $0-40 |
| 2 | Full offline teacher traces: router probs + selected hidden states | Allocated TPU, bf16 26B sharded, forward-only | ~$0 / $30-100 |
| 3 | Zero-shot ablation evals (128/8 → … → 32/4) | Free notebooks + TRC | ~$0 / $20-80 |
| 4 | Router-only training, then router + hidden-state distillation | Allocated TPU | ~$0 / $50-150 |
| 5 | LoRA recovery on attention, shared MLP, selected experts; diffusion objective + encoder AR loss | Allocated TPU, 1-3 B tokens | ~$0 / $150-600 |
| 6 | Coding/agentic diffusion SFT | Allocated TPU | ~$0 / $100-500 |
| 7 | Export + quantize: NVFP4/int4 for the 5080, MLX 4-bit for the M5 | Workstation / RunPod; M5 verifies final artifact | $0-50 |

"~$0" TRC envelopes still carry incidentals: GCS storage (see the storage
budget below; ~$100/month at peak, ~$25/month after trace cleanup) and egress
when pulling checkpoints down for local evaluation (~$0.12/GB, so a 16 GB
student checkpoint costs about $2 to download). The RunPod fallback column is
the worst case if TRC quota lapses entirely — the original $350-1,500
envelope.

## Storage budget (run profile)

Provision a **~5 TB regional GCS bucket in the TPU zone** (intra-region reads
are free), plus ~150 GB of local disk for stage-0 conversion. GCS bills per
GB-month prorated (~$3.30/day at the 5 TB peak), and the dominant object —
the full teacher trace store — lives only for the ~2-3 weeks around stages
2-3 before its lifecycle rule deletes it, so **whole-program storage spend is
~$100-200 one-time**. Decision: prioritize run quality over storage
micro-optimization; full offline traces make distillation sweeps replayable
without regenerating teacher outputs. The lean-profile techniques below
(streaming staging, on-device stat aggregation, delta checkpoints once
exact-resume covers them) still apply — they are good engineering, just no
longer forced by budget.

| Component | Budget | How it stays small |
| --- | --- | --- |
| Upstream checkpoint + no-prune control | ~120 GB | Move to Nearline after stage-0 gate passes (~4x cheaper; rarely re-read) |
| Tokenized, canvas-packed training data | ~20 GB | uint32 tokens (262K vocab exceeds uint16): 3B tokens ≈ 12 GB |
| Raw dataset staging | ≤100 GB transient | Stream from Hugging Face during tokenization; never mirror. The Stack v2 especially is stream-and-sample only (full corpus is tens of TB) |
| Router calibration stats | ~MB + ~1 GB raw sample | Aggregate on-device via `RouterStatsAccumulator`; only the JSON aggregate plus a ~1M-token debug sample leave the TPU |
| Teacher distillation | 1.5-3 TB transient | **Full offline trace store** (4 selected layers, fp8, 100-200M tokens + router targets and top-64 canvas logits for the whole distill set); lifecycle-deleted after the stage-3 gate. Rolling-window (~150 GB) remains the fallback if the bucket must shrink |
| Checkpoints | ~200 GB | Full, self-contained checkpoints first, rolling window of 2-3; stage-gate milestones to colder storage. Base+delta composition is a later optimization that must pass the same exact-resume gate before use. |
| RL trajectories, eval outputs | ~50 GB | Keep verified-success trajectories (expert iteration needs them) zstd-compressed; prune failures after difficulty stats are extracted |

**Fallbacks if the full trace store becomes infeasible.** The approved default
is the replayable full offline store. Neither fallback assumes the 26B teacher
and student fit together with useful batch size on an unknown TPU topology:

- **Rolling window** (storage fallback): generate
  traces for the next ~10M tokens (4 layers, fp8), consume, delete. Bounded
  at ~150 GB regardless of total distill tokens.
- **Online distillation** (optional): teacher forward runs live during student
  training. Enable only if a measured co-residency pilot fits and improves
  end-to-end throughput after accounting for the extra teacher compute.

**What either fallback deliberately preserves** — do not cut these:
checkpoint *frequency* (30-60 min; preemption safety), the tested
exact-resume path, off-device
scalar metrics in GCS (optionally mirrored to W&B by the controller — they are
the forensic record that deleted checkpoints can no longer provide),
decontamination reports and eval
outputs for the tech report (small), and one debug sample per deleted log
class.

**What the rolling-window fallback accepts:** reproducing a distillation run means regenerating
teacher outputs (deterministic given seed + teacher checkpoint) rather than
replaying stored traces; and a divergence noticed later than 2-3 checkpoints
back means restarting the phase from its milestone rather than rewinding
precisely. Both are cheap at our phase lengths.

Memory sanity: the bf16 student is ~16.2 GB. With experts and backbone frozen
and only router/LoRA parameters carrying optimizer state, a v3-8 (128 GB HBM)
or a single A100 80 GB holds weights, activations (with checkpointing), and
optimizer with a comfortable margin. Distillation defaults to the full offline
trace store on every provider. Rolling-window traces are a storage fallback;
online co-residency is enabled only after a topology-specific fit and
end-to-end throughput measurement.

## Throughput and cost model

Assume 2-6k trained tokens/sec for LoRA on one A100 or v3-8 (to be measured in
the phase-5 pilot). One billion tokens is then ~45-140 device-hours — free
within a TRC window, or ~$55-270 on the RunPod fallback. The envelopes assume
1-3 B recovery tokens and a smaller SFT mix; a pilot that overfits a tiny
batch and resumes exactly from checkpoint (existing go/no-go gate) runs before
committing a TRC window or real money to the full run.

## Checkpoint and preemption hygiene

The executable setup, device/GCS preflight, and Orbax write/restore test live
in `infra/tpu/README.md`. Those checks are mandatory on the allocated topology,
not merely documentation.

- Save optimizer + model + sampler-RNG state every 30-60 minutes of device
  time — Orbax to GCS on TRC, network volume plus a private Hugging Face repo
  (or B2/R2 bucket) on RunPod — so a preempted device loses at most one
  interval.
- Exact-resume is a tested code path (the existing gate), not a hope:
  interrupt only the recorded user-space pilot processes mid-step and verify
  loss-curve continuity. Never stop or reboot a TPU VM for this test.
- Write router-mass, held-out diffusion loss, and eval metrics to the run's GCS
  prefix. The controller may mirror them to a hosted tracker later; workers
  have no public internet access.

## RTX 5080 inference notes

- Pruned 8B at 4-bit ≈ 4.1 GB weights; at 8-bit ≈ 8.2 GB. Both leave real KV
  and canvas headroom in 16 GB; prefer 8-bit first to isolate quality effects
  of pruning from quantization, then step down.
- vLLM and SGLang carry native DiffusionGemma support and are the reference
  serving path on CUDA; MLX remains the Mac path.
- Blackwell FP4 tensor cores make NVFP4 the natural final format on this card.
- Keep the router in fp16 and the head at 8-bit here too, mirroring the MLX
  deployment config, until entropy-selection stability under quantization is
  measured.
