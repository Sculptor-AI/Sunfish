# GCP setup and cost guardrails

TPU time is granted (TRC); everything else on GCP costs real money. This
runbook exists because the bill failure modes are quiet: soft-deleted
storage, idle VMs, orphaned disks, casual egress, and — the one that can
sink the budget outright — a storage region that doesn't match the TPU zone.
Revision 2 — incorporates Codex review (channel [6]).

**Budget alerts are notifications, not spending caps.** Nothing on GCP stops
spend automatically; the guardrails below only make mistakes loud and fast.

All commands parameterized — set these first (zone comes from the TRC grant):

```bash
export PROJECT_ID=sculptorai-sunfish
export ZONE=us-central2-b            # from the TRC grant email
export REGION="${ZONE%-*}"           # bucket MUST be co-located (verify below)
export BUCKET="gs://sunfish-training-${PROJECT_ID}"   # bucket names are GLOBALLY unique
```

## One-time setup (in order)

### 0. Pin gcloud to the project — every command, not just some

```bash
gcloud config configurations create sunfish
gcloud config set project "$PROJECT_ID"
gcloud config list                   # VERIFY project before proceeding
```

Resources created against the wrong active project are the classic silent
failure; the named configuration makes the pin explicit. Then enable the
APIs now, not via first-use prompts mid-window:

```bash
gcloud services enable billingbudgets.googleapis.com storage.googleapis.com \
  iam.googleapis.com compute.googleapis.com tpu.googleapis.com
```

### 1. Budget alerts BEFORE anything else

```bash
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
gcloud billing budgets create \
  --billing-account "$(gcloud billing projects describe $PROJECT_ID --format='value(billingAccountName)')" \
  --display-name sunfish-budget \
  --filter-projects "projects/$PROJECT_NUMBER" \
  --budget-amount 200USD \
  --threshold-rule percent=0.25 --threshold-rule percent=0.5 \
  --threshold-rule percent=0.75 --threshold-rule percent=1.0
```

$200 is deliberately **above** the $100-190 expected envelope: the 25% alert
($50) is the early tripwire, and 100% ($200) firing means the plan itself is
wrong — stop and read the billing report by SKU. `--filter-projects` scopes
the budget to this project; without it the budget watches the whole billing
account. Check whether the billing account carries the $300 new-account
credit, and ask TRC support if the grant includes GCP credits.

### 2. Bucket — VERIFY the region exists, then create with soft-delete OFF

**Unverified assumption, resolve before the window (P0):** v4 lives in
`us-central2-b`, but `us-central2` is not in Cloud Storage's published
regional location list. Run the create command in the real TRC project NOW:

```bash
gcloud storage buckets create "$BUCKET" \
  --location "$REGION" --uniform-bucket-level-access
```

- If it succeeds: co-located, intra-region reads free, proceed.
- If it fails: **stop and decide the fallback with Chase before training.**
  Nearest-region storage means cross-region transfer at ~$0.02/GiB — one
  full 3 TiB trace-read pass is **~$61 per direction**, and repeated reads
  would erase the program budget. The fallback decision (rolling-window
  traces vs. paying transfer vs. different storage region strategy) changes
  the storage profile and must be priced first.

```bash
# MONEY TRAP: soft delete retains deleted objects 7 days AND BILLS for them
# (~$14 on a 3 TiB trace deletion). Turn it off before any large writes:
gcloud storage buckets update "$BUCKET" --soft-delete-duration=0

gcloud storage buckets update "$BUCKET" --lifecycle-file=infra/gcp/lifecycle.json

# Verify what actually applied — mutating commands are not confirmation:
gcloud storage buckets describe "$BUCKET" \
  --format="yaml(softDeletePolicy,lifecycle)"
```

Layout convention the lifecycle rules and the calibration hook both depend
on — do not improvise prefixes:

```
gs://.../sunfish/
  upstream/        # 52 GB checkpoint + control; move to Nearline MANUALLY after stage-0 gate
  data/            # tokenized, canvas-packed        -> Standard, keep
  traces/          # teacher traces                  -> DELETE explicitly after stage-3 gate
  calib/<run>/     # router-stats JSON aggregates    -> tiny, keep
  calib/raw/<run>/ # debug sample + reconstruction   -> lifecycle backstop deletes
  ckpt/<run>/      # rolling Orbax                   -> pruned by CheckpointManager max_to_keep
  milestones/      # stage-gate checkpoints          -> Nearline via lifecycle
  evals/           # eval outputs                    -> keep (small)
```

Lifecycle rules are a **backstop against forgetting, not the mechanism**:
explicit deletion after each gate is primary. The trace backstop is 60 days
(not 30) so a delayed recovery run cannot lose shards it still needs;
`upstream/` is moved to Nearline manually after the stage-0 gate rather than
by an age rule that could fire before a delayed gate. Nearline objects have
a 30-day minimum-storage charge and per-GiB retrieval fees — one more reason
transitions follow gates, not clocks.

### 3. Service account — bucket-scoped, nothing project-wide

```bash
gcloud iam service-accounts create sunfish-tpu --project "$PROJECT_ID"
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member "serviceAccount:sunfish-tpu@$PROJECT_ID.iam.gserviceaccount.com" \
  --role roles/storage.objectAdmin
```

The allocation owner attaches this account and the required scopes before
handoff. Sunfish never mutates the allocated TPU VM. No Hugging Face token or
other public-network credential is copied to a worker; model and seed artifacts
arrive through approved GCS prefixes.

### 4. Scratch disk — not part of the TPU path

Stage-0 download, conversion, parity, seed materialization, tokenization, and
offline wheel-bundle construction all run off-TPU. TPU workers restore the
finished seed and read immutable data from GCS. Do not attach/detach scratch
disks or otherwise reconfigure the scarce allocation from Sunfish automation.

### 5. Outbound network — workers are air-gapped

The TPU VMs have no public internet access. Do not add external IPs or Cloud
NAT. A connected Linux packaging host resolves PyPI/GitHub once and emits the
hash-bound offline source/wheel archive. The controller copies it to all
workers with `gcloud alpha ... tpu-vm scp --worker=all
--tunnel-through-iap`; worker bootstrap uses only `--no-index`. Worker access
to GCS uses the attached service account and Google API path, not public
package/model endpoints.

## Standing money rules

1. **Egress to the internet is the expensive irreversible spend.**
   Intra-region TPU↔GCS is free; pulling to the house costs ~$0.12/GiB.
   Quantized artifacts (4-8 GB, ~$1): freely. bf16 checkpoints (16 GB, ~$2):
   deliberately. Traces (3 TiB, **~$369**): NEVER — this is the single
   biggest possible mistake. Download/materialize on the approved off-TPU
   compute host in-region and stage only required artifacts to GCS; never use
   the TPU pod or Chase's laptop as the conversion machine.
2. **The CPU sandbox VM (stages 6-7) runs only during rollout generation.**
   Spot provisioning (~60-90% off), stopped whenever rollouts aren't
   running. A forgotten on-demand e2-standard-16 is ~$390/month.
3. **GCS operations are cheap but not free**: Class A ≈ $0.005 per 1,000
   ($5/million). Batching traces into large shards remains correct — a
   million tiny objects costs more in ops than a day of their storage — but
   the dominant reason is throughput, not the ops bill.
4. **Costs this plan tracks but does not itemize** (kept small by design;
   check them in the SKU report if the budget alert fires): Cloud
   Logging/Monitoring ingestion, VM boot disks, IAP traffic, and Nearline
   retrieval/early-deletion fees.

## End-of-session checklist (every working day)

```bash
gcloud compute instances list --format="table(name,status)"   # all TERMINATED?
gcloud compute tpus tpu-vm list --zone "$ZONE"                 # TRC VMs: per grant terms
gcloud compute disks list --filter="-users:*"                  # orphaned disks = silent $
gcloud compute addresses list                                  # no RESERVED addresses
gcloud storage du -s "$BUCKET"                                  # within budget envelope?
```

## Expected bill, honestly (conditions attached)

| Period | Expected | Dominated by |
| --- | --- | --- |
| Stages 0-2 (~2 wks) | $10-20 | upstream copy in Standard, scratch disk |
| Stage 3 (~2-3 wks) | $50-90 | ~5 TiB peak (≈$3.41/day) with traces alive |
| Stages 5-7 (~3 wks) | $30-60 | checkpoints + Spot sandbox VM hours |
| Stages 8-10 + wind-down | $10-20 | milestones in Nearline, artifact egress |
| **Program total** | **~$100-190** | |

This envelope holds **only if**: the bucket is genuinely co-located with the
TPU zone (verified in step 2), the 5 TiB peak lasts weeks not months, traces
are written as large shards and never egressed, Spot sandbox hours stay
bounded, and the untracked services above stay small. Any drift: stop, run
the end-of-session checklist, read the billing report by SKU.
