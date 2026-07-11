# Compute-call brief — questions for the TRC contact

One page to read off during the call. Everything else is ready and waiting
on these answers.

## Must-get facts (the plan is parameterized on these)

1. **Zone and slice**: which TPU type/topology, which zone, single-host or
   pod? (We requested 32× v4 / 64 cores; v5e-64 is our stated fallback.)
2. **Dates**: when does the window open and close, and is renewal
   realistic? We treat the window as a deadline and arrive with data and
   code staged.
3. **Preemption policy**: on-demand or preemptible quota, and typical
   reclaim behavior? (Our checkpointing assumes preemptible; on-demand is
   a bonus.)
4. **Project ID** the quota lands in, and whether we get owner/editor on it
   or work inside someone else's project with a service account.
5. **TPU VM host specs**: host RAM matters to us specifically — we want to
   run a 26B bf16 parity check on the host CPU day 1 (needs ~60+ GB RAM;
   v4 hosts typically have hundreds, just confirm).
6. **Egress or org-policy restrictions**: can the TPU VM reach Hugging Face
   and PyPI for bootstrap? Any VPC-SC / org policy that blocks external
   pulls?

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

Checkpoint downloaded and audited to the parameter, conversion pipeline
built, cross-reviewed by two independent models, and executed cleanly on the
real 52 GB weights; training/data/eval plans locked with gates; storage
lifecycle and cost guardrails written. The window starts productive on
day 1 — his compute won't idle while we set up.
