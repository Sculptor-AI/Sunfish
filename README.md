# Sunfish

Sunfish is an experimental, coding-first diffusion language model aimed at very
low single-user latency on consumer hardware — an RTX 5080 (16 GB) as the
primary target and a 24 GB Apple Silicon laptop as the secondary one.

**Start with [PLAN.md](PLAN.md)** — the canonical end-to-end program: every
stage from checkpoint conversion through diffusion RL to release, with gates.

The current leading design is a text-only, structurally pruned derivative of
Google's DiffusionGemma architecture:

- preserve uniform-state discrete diffusion, self-conditioning, causal prompt
  prefill, and bidirectional canvas denoising;
- reduce the routed MoE from 128 experts to 32 experts per layer while keeping
  the always-active shared expert untouched;
- initially route 4 experts per token instead of 8;
- remove the vision tower for the coding/agentic target;
- retain the upstream tokenizer and 256-token canvas for the first faithful
  baseline, then measure smaller canvases and a code-focused vocabulary as
  separate optimizations.

An audit of the real released shard headers (2026-07-10, see
`reference/upstream/`) gives **8,114,384,892 total / 3,118,575,612 active**
text parameters for that configuration — the design calculation was verified
against the materialized weights to within 60 parameters and then corrected
to match them exactly.

## Why this route

Starting from a small autoregressive model would require teaching the model the
diffusion objective, bidirectional denoising, self-conditioning, and hybrid
prefill/decoding behavior. Structured expert pruning lets Sunfish inherit those
behaviors from an already-trained diffusion model and makes expert selection
and recovery training the main research problem.

## First gates

1. Reproduce the upstream 4-bit model on the target hardware — vLLM on the
   RTX 5080 (tight: ~13 GB of weights in 16 GB, small context only) and MLX
   on the M5 — recording peak memory, time to first canvas, tokens/second,
   and denoising-step counts.
2. Instrument router use across coding, agentic, and general prompts, separated
   by causal-prefill and denoising phase.
3. Run a controlled 128/8 → 64/8 → 32/8 → 32/6 → 32/4 ablation, measuring
   retained router mass, quality, memory, and latency at every transition.
4. Recover the best candidate with router training, hidden-state distillation,
   and diffusion SFT before attempting broad coding/agentic post-training.
5. Export to MLX, quantize, and tune canvas size and adaptive stopping on the
   actual laptop.

Google explicitly cautions that Apple Silicon may not see the same diffusion
speedup as dedicated GPUs because its compute-to-memory-bandwidth ratio is
different. The first M5 benchmark is therefore a test of the project thesis,
not a ceremonial setup step.

The project will not spend on a large training run until the checkpoint
transformation is lossless on a no-prune control and the zero-shot pruning data
supports a specific candidate.

## Repository map

- `docs/architecture.md` — architecture choice, alternatives, risks, and gates
- `docs/upstream_checkpoint.md` — audited live config/index and tensor-name contract
- `docs/training.md` — training infrastructure, phase map, and cost envelopes
- `docs/data.md` — calibration/recovery/SFT data mixes and evaluation suite
- `docs/post_training.md` — SFT → rejection sampling → diffusion RL recipe
- `configs/sunfish-8b-a3b.toml` — initial candidate configuration
- `src/sunfish/model_budget.py` — reproducible parameter-budget calculator
- `src/sunfish/sampling.py` — dependency-free sampler correctness oracle
- `src/sunfish/checkpoint_audit.py` — header-only safetensors recount tool
- `src/sunfish/checkpoint_convert.py` — dependency-free streaming text/pruning converter
- `src/sunfish_tpu/tpu_preflight.py` — JAX device, package, and GCS readiness checks
- `src/sunfish_tpu/checkpoint_smoke.py` — exact Orbax save/restore probe for local/GCS paths
- `infra/tpu/README.md` — TPU VM bootstrap, access checklist, and launch gates
- `infra/gcp/README.md` — GCP setup and cost guardrails (budget alerts, lifecycle, egress rules)
- `src/sunfish/router_stats.py` — bucketized router-mass accumulation schema
- `src/sunfish/expert_selection.py` — coverage-constrained expert selector
- `tests/` — unit tests for all of the above

## Run the tools

```bash
PYTHONPATH=src python -m sunfish.model_budget --experts 32 --top-k 4
PYTHONPATH=src python -m unittest discover -s tests

# After downloading the upstream checkpoint (headers only are read; fast):
PYTHONPATH=src python -m sunfish.checkpoint_audit --dir /path/to/checkpoint --list
PYTHONPATH=src python -m sunfish.checkpoint_audit --dir /path/to/checkpoint

# Validate the mandatory no-prune text-only control. The output directory must
# not already exist; remove --dry-run only after reviewing the plan.
PYTHONPATH=src python -m sunfish.checkpoint_convert \
  --source /path/to/diffusiongemma-26B-A4B-it \
  --output /path/to/diffusiongemma-text-control \
  --retained-experts 128 --top-k 8 --dry-run

# On the requested v4-64 TPU VM, after: pip install -e '.[tpu]'. Replace 64
# with the granted global JAX device count if the allocation differs.
sunfish-tpu-preflight --require-tpu --expected-devices 64 \
  --gcs-workdir gs://YOUR_BUCKET/sunfish/experiments --require-gcs --probe-gcs-read
```

## Upstream references

- [DiffusionGemma model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [DiffusionGemma overview](https://ai.google.dev/gemma/docs/diffusiongemma)
- [Diffusion in text generation explained](https://ai.google.dev/gemma/docs/diffusiongemma/explained)
- [Google's developer guide](https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/)
- [Google's launch post and Apple Silicon caveat](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)
- [Official Gemma diffusion code](https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion)
- [Hackable Diffusion](https://github.com/google/hackable_diffusion)

DiffusionGemma and the referenced implementation are released under Apache
2.0. Sunfish will preserve required notices and provenance as upstream code or
weights are incorporated.
