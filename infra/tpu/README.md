# TPU readiness and launch runbook

`coordination/external_tpu_review.md` is authoritative. This runbook is the
executable path implementing it. No TPU training starts until the eight tests
in `PLAN.md` Stage 0.5 pass in order on the granted slice.

## Allocation facts required first

Obtain the accelerator/topology, actual visible device count, host count,
local devices per host, VM image/runtime, preemption policy, allocation dates,
project, zone, attached service account, GCS bucket, and egress restrictions.
Do not infer any of those from the phrase "v4-64".

The service account is restricted to the Sunfish GCS prefixes it needs:

- immutable data manifests/shards (read);
- exact-tree Orbax seed checkpoints (read);
- the selected run workdir (read/write).

Conversion and the 51 GB Hugging Face download happen off the TPU VMs. TPU
workers restore the parity-approved Orbax seed directly from GCS; no worker
needs 150 GB of checkpoint scratch or a Hugging Face token.

## Fully ordered bootstrap

The bootstrap uses Python 3.12, the exact TPU stack in
`requirements-tpu-base.lock`, and exact Gemma source commit
`09e7b48ae88720f6236b8266c7213eb51bb62b87`. Gemma is installed with
`--no-deps` because its unreleased 4.1.0 metadata currently points at a
floating Hackable Diffusion branch. The runtime checks `direct_url.json` and
every direct version before training.

Set the measured values on every host:

```bash
export EXPECTED_TPU_DEVICES=32
export EXPECTED_TPU_PROCESSES=8
export EXPECTED_LOCAL_TPU_DEVICES=4
export SUNFISH_GCS_WORKDIR=gs://YOUR_BUCKET/sunfish/runs
export VENV_DIR=.venv-tpu
```

Run `scripts/bootstrap_tpu.sh` concurrently on every worker. Its ordering is
binding: create environment → install exact stack → distributed JAX init →
topology and real collective → GCS read/list probe → record `pip freeze`.
The first distributed checkpoint smoke is the write/read authorization probe.
Do not import Kauldron or ask JAX for devices before distributed init.

For an ordinary Cloud TPU VM, attach the service account explicitly:

```bash
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --version "$TPU_RUNTIME" \
  --service-account "$TPU_SERVICE_ACCOUNT" \
  --scopes cloud-platform
```

Use internal IPs only after Private Google Access and the dependency-install
path have both been tested.

## All-host launcher

Every pod command goes through `scripts/launch_tpu_pod.sh`. It uses
`gcloud ... --worker=all`, gives every host one run ID/config/command, records
the exact remote command, and keeps a separate host log and `pip freeze`.
`scripts/tpu_host_entrypoint.sh` refuses a missing config or topology.

Never invoke `python -m kauldron.main` directly: that module calls
`jax.devices()` at import. `sunfish-train` and `sunfish-kauldron` initialize
and validate distributed JAX first.

## Ordered Stage-0.5 commands

### 1. Topology and collective

The bootstrap preflight is the test. Save every host's JSON report. Device,
process, and local-device counts must equal the grant, device ownership must be
process-disjoint, and the real cross-host `psum` must pass.

### 2. Process-disjoint GCS input

Pack a tiny tokenized fixture and upload its immutable output directory:

```bash
PYTHONPATH=src python3 -m sunfish_tpu.training.pack_records \
  --input tiny.jsonl \
  --output /tmp/sunfish-tiny-v1 \
  --records-per-shard 256 \
  --source stage05-disjoint-input
```

Copy the reported manifest digest into a smoke TOML. The Grain pipeline slices
record IDs by JAX process before shuffle and retains `record_id` in every
batch. Record read evidence per host must show no overlap and exhaustive
coverage.

### 3, 5, and 6. Sharded load, save/restore, exact resume

Run the synthetic sharded-state proof collectively with a unique ID:

```bash
scripts/launch_tpu_pod.sh \
  --run-id stage05-checkpoint-smoke \
  --config configs/training/sunfish-smoke.toml \
  -- \
  .venv-tpu/bin/sunfish-checkpoint-smoke \
  --workdir gs://YOUR_BUCKET/sunfish/readiness \
  --run-id stage05-checkpoint-smoke \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES"
```

Then repeat the interruption/control comparison with the real trainer. Its
initializer releases random base arrays and restores the exact Gemma tree into
target NamedShardings. Kauldron checkpoints model, optimizer, step, timer, and
the Grain iterator together; RNG is reproduced from the saved step and pinned
seed. Compare the next record IDs, loss, gradients, and update after restart
with the uninterrupted control.

### 4. 100-500 update real training smoke

Replace all placeholders in a copied smoke config, including the actual
dataset digest and parity-approved exact-tree Orbax seed, then validate it:

```bash
.venv-tpu/bin/sunfish-train \
  --config configs/training/sunfish-smoke.toml \
  --validate-only
```

Launch the real 100-update job:

```bash
scripts/launch_tpu_pod.sh \
  --run-id sunfish-stage05-smoke \
  --config configs/training/sunfish-smoke.toml \
  -- \
  .venv-tpu/bin/sunfish-train \
  --config configs/training/sunfish-smoke.toml
```

The strict schema rejects a smoke outside 100-500 updates. Tiny-batch overfit,
finite loss, and changing trainable leaves are required; mere process survival
is not a pass.

### 7. Preemption recovery

Interrupt the job after a finalized checkpoint and rerun the identical
all-host command. No checkpoint deletion, cursor editing, or new run config is
allowed. `sunfish-run.json` rejects any changed config, dataset, dependency,
seed checkpoint, model, or topology.

### 8. Input throughput

Measure host read latency/bytes and accelerator step time during the smoke.
Increase Grain workers/prefetch only from evidence. A training step waiting on
small GCS range reads fails the gate even when correctness tests pass.

## Checkpoint policy

- Every save goes to the run's GCS workdir through Orbax/Kauldron.
- Default retention is three full checkpoints; interval comes from the strict
  run TOML and is converted from the approved wall-clock target after the
  throughput smoke.
- Clean shutdown waits for async finalization. Abrupt preemption loses only
  work after the last finalized checkpoint; restart restores automatically.
- No multi-hour job starts before the real exact-resume comparison passes.

See `docs/training_harness.md` for phase masks, prefix-amortized multi-noise
training, record/loss-mask semantics, and the initial checkpoint contract.
