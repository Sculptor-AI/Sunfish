# Compute-call brief — questions for the TRC contact

One page to read off during the call. Everything else is ready and waiting
on these answers.

## Must-get facts (the plan is parameterized on these)

1. **Zone and slice**: which TPU type/topology, which zone, single-host or
   pod? (We requested 32× v4 / 64 cores; v5e-64 is our stated fallback.)
2. **Dates**: when does the window open and close, and is renewal
   realistic? We treat the window as a deadline and arrive with data and
   code staged.
3. **Allocation protection**: the pod is non-preemptible; confirm the
   maintenance/reclaim policy and the exact allocation-owner escalation path.
   Sunfish will not start, stop, reset, reboot, delete, or reconfigure it.
4. **Project ID** the quota lands in, and whether we get owner/editor on it
   or work inside someone else's project with a service account.
5. **CPU VM path**: can we provision or receive credits for a separate
   high-memory CPU VM with at least 160 GB RAM? Stage-0 same-framework parity
   and exact-tree seed materialization load the 25B teacher off-TPU; TPU workers
   consume only the finished GCS seed and must not become conversion machines.
6. **IAP and Google API access**: confirm `gcloud alpha compute tpus tpu-vm
   ssh/scp --worker=all --tunnel-through-iap`, required IAP/OS Login roles,
   and service-account access to the exact GCS prefixes. Public worker egress
   is not required or expected; dependencies arrive in the offline bundle.

## The one live test to run DURING the call (2 minutes)

With access to the actual project:

```bash
gcloud storage buckets create gs://sunfish-training-<project> \
  --location <the TPU zone's region> --uniform-bucket-level-access
```

If this **succeeds**, our entire ~$100-200 storage budget is confirmed.
If it **fails** (us-central2 may not be a valid bucket region), we need the
nearest-region name and we'll re-price before training — cross-region reads
at 3 TiB scale would otherwise quietly eat the budget. This is the single
biggest open financial question in the project; two minutes on the call
settles it.

## Nice-to-haves (worth one sentence each)

- Does the grant include any **GCP credits** (some TRC grants do)? Covers
  our entire incidental spend if so.
- A second small **CPU VM allowance** in the same project (rollout sandboxes
  for RL later; Spot is fine).
- Whether **multiple slices** or a larger slice could be available in a
  later window (full-parameter unfreeze contingency).

## What you can tell him we've already done (context, if he asks)

Checkpoint downloaded and audited to the parameter; static conversion P1 is
691/691, while the high-memory P2-P5 model-forward parity run remains explicit
pre-TPU work. The multi-host trainer, immutable configs, exact-tree seed path,
data loader, checkpoint/resume/process-interruption proofs, and eight-gate evidence
ledger are implemented and locally tested. Worker deployment is air-gapped:
one offline-validated Linux wheel/source archive is transferred to every host
through IAP, and Sunfish exposes no allocation lifecycle operation. Hardware
passes are not claimed until the granted slice emits the ledger. Storage lifecycle and cost
guardrails are written, so the window can start with the ordered gauntlet.
