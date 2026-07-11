# TPU readiness and launch runbook

> **SUPERSEDED IN PART (2026-07-11):** `coordination/external_tpu_review.md`
> is now authoritative for multi-host launch. Known corrections to this
> document: (1) every entrypoint must call `jax.distributed.initialize()`
> BEFORE any backend access — the current preflight ordering is wrong on
> multi-host; (2) everything launches on every worker (`--worker=all`), not
> one shell; (3) conversion does NOT happen on TPU nodes (storage-
> constrained) — the converted checkpoint uploads to GCS and TPU hosts
> restore from there; the 150 GB local-scratch instruction below no longer
> applies; (4) `python -m kauldron.main` is unsafe on pods (eager
> `jax.devices()` at import) — a custom launcher is required. Rewrite
> pending in the infra lane.

This is the operational path for TPU Research Cloud and ordinary Cloud TPU VM
access. It prepares the environment and storage contract; the stage-specific
training entry point is added only after the upstream forward pass and router
hook pass their Stage 0/1 correctness gates.

## What to obtain before the allocation starts

- GCP project and TPU quota/grant, TPU type, zone, topology, and whether it is
  a single host or pod slice.
- A service account attached to the TPU VM with read/write access restricted
  to one Sunfish GCS prefix. Checkpoints must never exist only on preemptible
  local storage.
- A GCS work directory such as `gs://bucket/sunfish/experiments`.
- At least 150 GB of persistent scratch space for the 51.7 GB upstream BF16
  checkpoint, a no-prune text control, conversion output, and download cache.
- A Hugging Face account that has accepted the DiffusionGemma license terms
  and a token supplied through the VM environment, never committed to Git.

Ask the TPU contact for these exact facts: accelerator/topology, visible chip
count, host count, VM image/runtime, preemption policy, allocation dates,
project ID, zone, service account, GCS bucket, and egress restrictions.

## Bootstrap on the TPU VM

The project requires Python 3.12+. JAX's official TPU installation is the
`jax[tpu]` extra; the project extra also installs Gemma 4.0.1, Hackable
Diffusion explicitly, Kauldron through Gemma, Orbax, and Google Cloud Storage
support.

```bash
git clone YOUR_SUNFISH_REPOSITORY
cd sunfish-v2

# Requested v4-64 exposes 64 global JAX devices; replace this if the grant differs.
export EXPECTED_TPU_DEVICES=64
export SUNFISH_GCS_WORKDIR=gs://YOUR_BUCKET/sunfish/experiments
bash scripts/bootstrap_tpu.sh > tpu-environment.txt
```

The bootstrap must finish with no failed checks. Preserve
`tpu-environment.txt` beside the experiment manifest so the exact resolved
package versions are recoverable.

For a Cloud TPU VM rather than a pre-provisioned TRC machine, create and SSH to
the resource using the accelerator type, runtime, project, and zone actually
granted to you. Keep these as variables because available TPU generations and
zones change. The custom service account must be passed explicitly; merely
creating it does not attach it to the TPU VM:

```bash
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --version "$TPU_RUNTIME" \
  --service-account "$TPU_SERVICE_ACCOUNT" \
  --scopes cloud-platform

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

Use `--internal-ips` only after Private Google Access and a controlled outbound
path for the Python/Hugging Face bootstrap have both been tested. Otherwise an
internal-only VM can reach GCS but cannot install packages or download the
gated upstream checkpoint.

On multi-host slices, setup and the eventual job command must run on every
worker. Do not assume a one-host v3-8 launch command applies to a pod slice.

## Mandatory storage smoke test

After preflight, explicitly prove that Orbax can finalize and exactly restore
state from the real GCS prefix. Use a unique run ID; the small artifact is
retained as evidence and never overwritten by the tool.

```bash
.venv-tpu/bin/sunfish-checkpoint-smoke \
  --workdir "$SUNFISH_GCS_WORKDIR" \
  --run-id "$(date -u +%Y%m%dT%H%M%SZ)-topology-smoke"
```

This smoke state contains a step, parameters, optimizer state, data cursor,
and RNG state. The training loop must later add an interruption/restart test
that compares the next loss and update against an uninterrupted control.

## Checkpoint policy for real jobs

- Use Orbax `CheckpointManager` with async checkpointing to the GCS workdir.
- Save every 30 minutes during calibration/recovery and at every stage gate.
- At preemption, wait for checkpoint finalization before exiting.
- Store model/optimizer state, global step, RNG streams, data cursor/shuffle
  epoch, sampler schedule state, and resolved configuration.
- Keep metrics and the package freeze off-device as well.
- No multi-hour job starts until exact interruption/resume is demonstrated.

## Stage 0 checkpoint conversion

Download the accepted upstream checkpoint once to persistent scratch. First
validate the no-prune text-only plan, then run it into a new directory:

```bash
.venv-tpu/bin/sunfish-convert \
  --source /mnt/disks/sunfish-cache/diffusiongemma-26B-A4B-it \
  --output /mnt/disks/sunfish-cache/diffusiongemma-text-control \
  --retained-experts 128 \
  --top-k 8 \
  --dry-run

.venv-tpu/bin/sunfish-convert \
  --source /mnt/disks/sunfish-cache/diffusiongemma-26B-A4B-it \
  --output /mnt/disks/sunfish-cache/diffusiongemma-text-control \
  --retained-experts 128 \
  --top-k 8
```

Do not generate the 32-expert candidate until this control has reproduced
upstream text logits and seeded generations. Pruning additionally requires a
per-layer selection JSON produced from calibration traces.
