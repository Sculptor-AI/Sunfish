# Logit-parity harness — spec (stage-0 gate)

Owner: Claude (spec) + Codex (implementation/execution). The executable is
`sunfish-parity` (`src/sunfish_tpu/parity.py`). Purpose: prove the converted
128/8 text-only control is
functionally identical to upstream on every text path, so that any later
quality change is attributable to pruning, never to conversion.

## Principle

The control's tensors are byte-identical copies of upstream's text tensors.
Therefore the bar is **exactness, not tolerance**: same implementation, same
dtype, same device, same inputs ⇒ bitwise-equal outputs. A tolerance would
only paper over a real conversion or config defect. Any nonzero diff fails
the gate and must be explained.

## Environment (pin everything)

- Same `transformers` version (5.13.0), same torch,
  same device for both models. CPU float32 for the exactness runs
  (deterministic kernels); one additional bf16 run to confirm dtype parity.
- `requirements-parity.lock` and the runtime guard pin every load-bearing
  direct dependency. `scripts/bootstrap_parity.sh` records the full transitive
  `pip freeze`. This environment belongs on an internet-connected high-memory
  CPU compute host, never Chase's laptop and never the network-isolated TPU
  pod. Only its immutable report/traces move into the TPU deployment bundle.
- The report and each trace record the Sunfish Git commit and deterministic
  source-tree SHA-256. Resume metadata includes that identity, so a code or
  configuration edit invalidates an otherwise reusable trace.
- Upstream loaded from the downloaded snapshot; control loaded from the
  converter output. Record both paths, `sunfish_conversion.json`, and the
  upstream revision hash in the report.
- Text-only prompts throughout — the upstream vision path must be untouched
  (this is what makes upstream-with-vision vs control-without comparable).
- Transformers 5.13.0's `DiffusionGemmaConfig` supports `vision_config=null`,
  but its model constructor still instantiates the vision tower. At load time
  only, the harness gives the control the upstream *vision architecture* so
  both use the same official model class; missing control vision weights are
  initialized and never read by these text-only prompts. The saved control
  config and tensors are not changed.

## Checks, in order (stop at first failure)

### P1 — static equivalence (no forward pass)

1. Tokenizer files byte-identical between source and control.
2. Control config equals upstream config except exactly:
   `vision_config: null` (and nothing else — diff the JSON trees).
3. **Weight parity: hash EVERY retained tensor** (SHA-256 over each tensor's
   byte range in both checkpoints, matched via the two indexes + the
   converter manifest); all hashes equal, and the control contains exactly
   the source tensor set minus the manifest's dropped names. A full 52 GB
   read is minutes of I/O — sampling is false economy here (Codex, wire
   seq 2).

### P2 — prefill logits (encoder/causal path)

- **Loader/runtime (pin and record):** both models via the same pinned
  `transformers` release, `DiffusionGemmaForBlockDiffusion.from_pretrained(
  path, dtype=torch.float32, device_map="cpu")`;
  `torch.use_deterministic_algorithms(True)`; `torch.set_num_threads(1)`;
  no compile/attn backend overrides — identical defaults on both.
- 32 prompts committed as `tests/fixtures/parity_prompts.json`: 8 code
  (Python/TS/Rust/shell), 8 English prose, 8 multilingual, 8 structured
  (JSON/diff/markdown), lengths 256-2048 tokens **with at least 8 prompts
  >1024 tokens so the sliding-window boundary and dual-RoPE global/local
  paths are both crossed**.
- Full-vocabulary logits at every position, compared **streamed
  position-by-position** (262K-vocab logit tensors must never be
  materialized whole per prompt): `max |upstream - control| == 0.0`.
  Report max diff and argmax agreement (must be 1.0) regardless.
- The current text fixtures intentionally have tokenization margin. The
  harness records the raw length and deterministically truncates only the
  effective token IDs to the contract maximum of 2048, then records their
  SHA-256. It refuses any prompt below 256 tokens and requires at least eight
  effective prompts above 1024.
- The encoder returns hidden states rather than logits. P2 applies the model's
  tied `lm_head` and the audited 30.0 final-logit softcap one position at a
  time, exactly matching the block-diffusion forward's output projection.

### P3 — denoising-step logits (decoder/bidirectional path)

- For 8 of the P2 prompts (4 of them >1024 tokens): canvas of 256 tokens
  initialized as `torch.randint(0, vocab, (256,), generator=g)` with
  `g = torch.Generator().manual_seed(20260710)`, **reset identically before
  each model's run** so both consume the same stream.
- Exactly 4 denoising steps, `remaining_step = 4,3,2,1` through the upstream
  `linear_temperature` (min 0.4, max 0.8) → temperatures **0.8, 0.7, 0.6,
  0.5**. Acceptance/renoise/self-conditioning semantics are NOT reimplemented:
  both models run the upstream transformers `diffusion_gemma` generation
  utilities verbatim, seeded as above, so mask/update behavior is inherited
  from one shared implementation rather than specified here.
- Full canvas logits compared streamed at every step: exact equality as in
  P2. This exercises bidirectional attention, self-conditioning, and the
  entropy-selection inputs — the paths a text-only strip could plausibly
  disturb.
- Draft logits are captured through the official streamer's logits callback
  after the official temperature processor. A stopping-history length of 5
  makes early adaptive stopping impossible during this four-step check while
  leaving the upstream generation loop, sampler, renoise, and self-
  conditioning code unchanged.

### P4 — end-to-end seeded generation

- 16 prompts × 2 canvases each, full entropy-bound sampler (bound 0.1,
  48 max steps, temp 0.8→0.4, stopping threshold 0.005, stability 1),
  seeded as in P3: token-for-token identical outputs AND identical
  denoising-step counts per canvas.
- EOS stopping uses an empty EOS set for this fixed-length test so every
  prompt exercises exactly two canvases. The official EOS criterion remains
  in the loop and deterministically returns false.

### P5 — bf16 spot check

- Repeat P2 on 4 prompts in bf16 on the same device: exact equality still
  expected (identical weights, identical graph). If bf16 diverges while fp32
  matches, report as a finding — it indicates a nondeterministic kernel, not
  a conversion bug, but it must be understood before the gate closes.

## Report artifact

`gs://.../sunfish/evals/parity/<run_id>/report.json`: environment pins,
per-check pass/fail, max diffs, step counts, prompt hashes, and the
converter manifest. The stage-0 gate in PLAN.md closes only on all-pass.

## Execution

Run off-TPU on a Linux x86_64 CPU host with enough RAM for one 25B float32 model plus
loader/vision/cache overhead (budget at least 160 GB). The harness loads
only one model at a time and writes resumable traces after each model/dtype:

```bash
PYTHON_BIN=python3.12 VENV_DIR=.venv-parity \
  scripts/bootstrap_parity.sh --connected-compute-host

.venv-parity/bin/sunfish-parity run \
  --source /mnt/checkpoints/diffusiongemma-upstream \
  --control /mnt/checkpoints/sunfish-128x8-control \
  --p1-report evals/stage0/parity-p1-report.json \
  --conversion-manifest /mnt/checkpoints/sunfish-128x8-control/sunfish_conversion.json \
  --upstream-revision THE_EXACT_HF_SNAPSHOT_COMMIT \
  --workdir /mnt/parity/sunfish-stage0 \
  --resume
```

`--resume` reuses a trace only when its role, dtype, checks, upstream revision,
prompt file, checkpoint metadata fingerprint, and Sunfish source identity
still match. On success,
copy `/mnt/parity/sunfish-stage0/report.json`, the four trace JSON files, and
`parity-environment.txt` to the canonical GCS prefix. A trace hash mismatch
fails the gate; because full reference logits are deliberately not stored,
the failing report records `max_abs_diff=null` and requires a targeted rerun
for numeric diagnosis. Passing exact hashes always records max diff `0.0`.

The TPU config renderer accepts only an exact 40-hex upstream revision, all
five passing subgates, the audited 691-tensor P1 result, nonempty exact logit
signatures for P2/P3/P5, the exact dependency lock, and the same deployable
source identity. The report is copied into every immutable deployment bundle
and revalidated by the final readiness ledger.

## Non-goals

Cross-framework parity (JAX vs PyTorch) and quantized parity are later
gates (stages 1 and 9); this harness proves conversion correctness only.
