# Sunfish training harness

Status: implemented locally; TPU gauntlet execution pending the allocation and
real GCS paths. This document describes the executable contract, not a claim
that any hardware gate has passed.

## What runs

`sunfish-train` is the only supported training entrypoint. It reads one strict
TOML file, validates it before touching JAX, initializes distributed JAX and
checks the measured topology, and only then imports Kauldron/Gemma and builds
the trainer. Kauldron config overrides are rejected because an override would
not be represented in the immutable run digest.

The production loop is Kauldron's trainer, not a second Sunfish loop. Its
Orbax checkpoint is one transaction containing model/optimizer/step state,
timer state, and the checkpointable Grain iterator. RNG streams are pure
functions of the checkpointed step and pinned run seed. A restart with the
same workdir therefore resumes the complete deterministic state.

The all-host launcher also binds the run to its code: every worker must match
the controller's Git commit and deterministic source-tree SHA-256 before the
trainer starts. `sunfish-run.json` records that identity, and the readiness
ledger rejects evidence produced by a different tree. See
`infra/tpu/README.md` for the clean-checkout workflow. The launcher also
verifies the raw config-file SHA-256 on every host; the run identity records
both that byte hash and the canonical parsed-config digest.

The implementation is under `src/sunfish_tpu/training/`; the stable Kauldron
entry config is `configs/training/sunfish.py`.

## Training phases

| TOML phase | Trainable leaves | Parameter policy |
| --- | --- | --- |
| `smoke` | Sunfish LoRA leaves | Phase A, replicated 8B base; 100-500 updates required |
| `router` | `router_logits`, `router_scale`, `per_expert_scale` | Phase A replicated |
| `lora` | all `lora` leaves | Phase A replicated |
| `full` | all parameters | evidence-gated Phase B FSDP |

Sunfish LoRA covers attention, the dense shared MLP, self-conditioning, and
the fused ragged-MoE expert banks. Google's adapter handles ordinary linear
modules but does not see the expert `_Weight` providers, so Sunfish adds a
batched low-rank pair per expert and a matching fusion function.

The `router` model also contains the LoRA tree, but its optimizer freezes every
LoRA leaf. This small amount of inert state makes router -> LoRA promotion
structurally exact. A LoRA run uses `format = "kauldron-params"` to restore
only model parameters from one explicitly pinned router checkpoint step; it
starts a fresh step counter and fresh LoRA optimizer. An implicit `latest`
source is rejected because it could move underneath a run identity.

Phase B uses a path-aware target sharding. Expert banks and
`per_expert_scale` shard on axis zero; dense arrays prefer row sharding;
small/norm arrays replicate. On 32 devices this is a one-dimensional 32-wide
`data` mesh. If TPU megacore exposes 64 devices, it becomes `(replica, data) =
(2, 32)`: one expert per device within each of two complete replicas. A slice
that cannot divide the 32 experts evenly fails before model initialization.

## Prefix-amortized objective

For each example the model:

1. samples one valid canvas;
2. encodes the prompt and clean prior-canvas history once;
3. samples `K` independently corrupted versions of that selected canvas;
4. performs first-pass/self-conditioned denoising for all `K` draws with a
   shared KV cache;
5. flattens only the `K × batch` output axis for the stock Hackable Diffusion
   loss.

`prefix_stratified` sampling gives every prefix one draw in each of `K` equal
noise strata. The decoder emits logits only for the selected 256-token canvas,
not every canvas in the cache. This matters at a 262,144-token vocabulary: the
reference helper computes and then masks unused canvas logits, which would
erase most of the memory benefit at `K=4`.

The target tree also discards the categorical process's unused one-hot target
logits, and the prediction tree discards its unused argmax. At this vocabulary
size those convenience fields are too large to leave to compiler dead-code
elimination.

The JAX integration test compares gradients from a shared-prefix `K`-draw
calculation with `K` independent encodes. It runs when the pinned JAX stack is
installed; the dependency-free CI environment reports it as skipped.

## Record contract

`sunfish.datashards` remains an opaque immutable uint32 store. The trainer adds
a v1 envelope with:

- prompt and response token arrays;
- integer workload bucket;
- optional bit-packed prompt and response loss masks.

An omitted mask means all tokens are supervised. Agentic SFT should normally
set the prompt mask false for exogenous tool observations and the response
mask true for the next action. To teach recovery, serialize the bad action and
tool result into the conditioning prompt with false supervision, then put the
correct recovery action in the supervised response. If a response is
explicitly all-false, its EOS fill inherits false too, so it contributes no
hidden target loss.

The encoder target mask supervises a next token only when the current and next
positions are valid and the next token's loss mask is true. The diffusion mask
is the selected canvas's response mask. Records longer than configured prompt
or canvas capacity fail; the loader never silently truncates them.

Tokenized JSONL can be packed without JAX:

```json
{"prompt":[2,123,456],"response":[789,1],"bucket_id":3,"prompt_loss_mask":[false,false,false],"response_loss_mask":[true,true]}
```

```bash
PYTHONPATH=src python3 -m sunfish_tpu.training.pack_records \
  --input tiny.jsonl \
  --output /tmp/sunfish-tiny-v1 \
  --records-per-shard 8192 \
  --source stage05-tiny-overfit
```

Copy the reported `manifest_sha256` into the run TOML before uploading the
immutable directory to its versioned GCS prefix. Every worker verifies those
exact manifest bytes. `sunfish-run.json` in the workdir additionally pins the
config digest, dataset digest, initialization source and promotion step, model
dimensions, exact direct runtime versions, Gemma source commit, and measured
topology. A changed value makes resume fail closed.

## Initial checkpoint contract

The initializer accepts a bare Orbax Standard checkpoint containing the exact
non-LoRA JAX parameter tree rooted at `gemma_network.gemma_model`. It restores
directly into the target NamedShardings, preserves newly initialized LoRA
leaves, releases random base arrays before restore, and validates every path,
shape, dtype, and sharding. It never reconstructs a full checkpoint in each
host's CPU RAM.

The current safetensors converter and this initializer intentionally meet at
an explicit boundary: Stage 0 must produce the exact-tree Orbax seed after
model-level parity. A Hugging Face directory is not accepted by pretending it
is an Orbax checkpoint. `sunfish-orbax-seed` closes that boundary from
Google's official JAX/Orbax DiffusionGemma checkpoint: it slices the same
per-layer expert IDs, saves a normalized pruned intermediate, traces the exact
target model tree with `jax.eval_shape`, uses the pinned public Gemma loader to
reconcile checkpoint-vs-Linen path differences, and saves the final bare
exact-tree seed. It validates all 120 prunable JAX leaves, target paths/shapes/
dtypes, saved Orbax metadata, the audited 25,250,986,812→8,114,384,892
parameter counts, source revision, selection hash, and runtime
pins. The selection manifest must also state `purpose`, `selection_method`,
and a boolean `promotion_allowed`; those fields are copied into the seed
sidecar. A 96 GiB physical-RAM guard plus an explicit acknowledgement prevents
running this multi-GB job on Chase's laptop.

Run it on a high-memory Linux CPU VM. For Stage-0.5 only, use the committed
`configs/training/stage05-first32-selection.json`. That deterministic first-32
selection exists solely to exercise the real 8B checkpoint and training path
before Stage-1 calibration is allowed to run. Its `promotion_allowed=false`
marker is binding: it supplies no quality evidence and must never seed a
research, evaluation, recovery, or release run. Once Stage 2 approves the real
selection, rerun this materializer with that same manifest used by the
safetensors converter.

```bash
PYTHON_BIN=python3.12 VENV_DIR=.venv-seed scripts/bootstrap_seed_cpu.sh

# This is the exact public path hard-coded by Google's pinned DiffusionGemma
# fine-tuning source. Inventory it before the high-memory job.
.venv-seed/bin/sunfish-gcs-inventory \
  --uri gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it \
  --anonymous \
  --output /tmp/diffusiongemma-source-inventory.json
export SOURCE_REVISION="gcs-inventory-sha256:$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' \
  /tmp/diffusiongemma-source-inventory.json)"

JAX_PLATFORMS=cpu .venv-seed/bin/sunfish-orbax-seed \
  --source gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it \
  --source-revision "$SOURCE_REVISION" \
  --source-anonymous \
  --selection configs/training/stage05-first32-selection.json \
  --retained-experts 32 --top-k 4 \
  --intermediate gs://YOUR_BUCKET/sunfish/checkpoints/sunfish-stage05-first32-pruned-nested \
  --output gs://YOUR_BUCKET/sunfish/checkpoints/sunfish-stage05-first32-exact-tree \
  --manifest gs://YOUR_BUCKET/sunfish/checkpoints/sunfish-stage05-first32-exact-tree.json \
  --ack-high-memory-cpu
```

The intermediate is deliberately retained until the final exact-tree seed has
passed Stage-0.5 sharded restore. Delete it only through the bucket lifecycle
after that evidence exists.

The seed sidecar embeds complete source and output GCS inventories: relative
object name, generation, size, and CRC32C for every Orbax object. Seed
materialization rejects a source revision that does not equal the live source
inventory. Every TPU training restore and the real seed-load gate re-list the
output prefix and reject any replaced object before model compilation.

Every exact-tree training config pins both the seed sidecar path and its
SHA-256. At runtime every host verifies the sidecar, its declared output path,
target-vs-saved tree signature, source revision, selection hash, and promotion
policy before compilation. The smoke phase accepts only the non-promotable
Stage-0.5 seed; router and later exact-tree phases reject it. Copy the actual
sidecar digest into `checkpoint.init_manifest_sha256` alongside the dataset
manifest digest before launch.

For later phases, `kauldron-params` points at a prior run workdir plus a
finalized numeric checkpoint step. Kauldron restores only
`params.gemma_network.gemma_model` into the new run's target shardings; it does
not inherit the source optimizer, data cursor, timer, or step counter.

## Pinned adapter source

The released `gemma==4.0.1` does not contain `gemma.diffusion`. Google's
official DiffusionGemma fine-tuning path currently lives in Gemma main, so the
bootstrap installs exact commit
`09e7b48ae88720f6236b8266c7213eb51bb62b87` (package version 4.1.0) with
`--no-deps` after the exact base stack. This prevents that source tree's
floating Hackable Diffusion dependency from moving. Runtime validation reads
the installed distribution's `direct_url.json` and rejects any other commit.
`sunfish-runtime-api-audit` then parses the installed sources without importing
JAX or those packages. It fails bootstrap if the private LoRA, ragged-MoE,
mask/cache, Kauldron process-slicing/checkpoint-loop, or Orbax restore/commit
marker contracts differ, and records SHA-256 hashes for all reviewed files.
Sunfish deliberately does not import the upstream `sft_model` module: that
module imports an unused evaluator which globally overrides Orbax path
finalization. The small prefix helper and encoder loss are implemented locally
against the pinned lower-level adapter APIs instead.

## Launch

Bootstrap the accelerator-free controller environment. Once the actual
dataset/seed hashes and measured topology exist, use
`sunfish-render-tpu-configs` and `scripts/upload_tpu_configs.sh` as shown in
`infra/tpu/README.md`; the renderer also requires the all-pass Stage-0 P1-P5
report from the identical deployable source tree. The uploader atomically
publishes that report with the three configs and their manifest. Do not
hand-edit the reviewed templates. Perform static validation against the
rendered controller copy:

```bash
PYTHON_BIN=python3.12 VENV_DIR=.venv-tpu-controller \
  scripts/bootstrap_tpu_controller.sh

.venv-tpu-controller/bin/sunfish-train \
  --config "$SMOKE_LOCAL_CONFIG" \
  --validate-only
```

Bootstrap each TPU VM only after the grant's measured counts are known:

```bash
EXPECTED_TPU_DEVICES=32 \
EXPECTED_TPU_PROCESSES=8 \
EXPECTED_LOCAL_TPU_DEVICES=4 \
SUNFISH_GCS_WORKDIR=gs://BUCKET/sunfish/runs \
scripts/bootstrap_tpu.sh
```

Launch the identical command on every worker:

```bash
scripts/launch_tpu_pod.sh \
  --run-id "$SMOKE_RUN_ID" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-train \
  --config "$SMOKE_REMOTE_CONFIG"
```

Re-running that exact command against the same workdir is the supported normal
recovery path. The destructive gate-7 proof uses
`sunfish-preemption-smoke.toml` and `sunfish-preemption-gate` with its own
fresh workdir, so it cannot mistake a previously completed gate-4 run for a
successful recovery. Gate 6 similarly uses `sunfish-resume-smoke.toml` and an
empty workdir. Changing any other TOML field requires a new run ID/workdir.

## Stage-0.5 mapping

| Ordered gate | Harness mechanism | Pass condition |
| --- | --- | --- |
| 1. topology/collective | `sunfish-topology-smoke` via distributed-init-first all-host launcher | every host and real `psum` pass |
| 2. disjoint GCS input | `sunfish-input-smoke` runs the production Grain/Kauldron process slice, persists every host's record IDs, and merges exact coverage | `summary.json`: no overlap, no missing IDs, one manifest |
| 3. sharded seed load | `sunfish-seed-load-smoke` restores the real 8B seed into Phase-B target shardings | exact tree/shardings; each host/device resident bytes below a full model |
| 4. 100-500 update smoke | `phase = "smoke"` + `sunfish-smoke-evidence` | ≥100 contiguous steps, finite/nonzero norms, ≥10% tiny-set loss reduction |
| 5. save/restore | `sunfish-checkpoint-smoke` distributed composite state | every host observes exact GCS round trip |
| 6. exact resume | `sunfish-real-resume-smoke` on the production model/optimizer/Grain/Orbax path | next batch/loss/trainable grad+update+params/full optimizer+collections+step exact; base frozen |
| 7. preemption | `sunfish-preemption-gate` on a fresh identical-lineage workdir | finalized checkpoint survives exact-attempt kill; unchanged relaunch starts its metric stream at the saved step rather than step 0 and completes without cleanup |
| 8. throughput | real trainer iterator waits divided by accelerator step time | p95 ≤10%, zero local cache |

No row in this table is a hardware pass until its evidence is recorded from
the granted slice. `sunfish-readiness-ledger` hashes and validates all rows;
it also re-runs the embedded host-evidence mergers and binds the rendered
config bundle plus Stage-0 report. Only its `passed: true` unlocks Stage 1.
