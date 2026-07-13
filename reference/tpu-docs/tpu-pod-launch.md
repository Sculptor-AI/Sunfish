# Cloud TPU pod-slice launch procedure (Sunfish-corrected)

The original public example fetched JAX on each worker and used the classic
non-IAP command form. That example is **not operational for this allocation**:
the TPU VMs have no public internet access and are reachable only through IAP.

## Sunfish transport contract

Every remote command uses:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --worker=all \
  --batch-size=all \
  --tunnel-through-iap \
  --command="..."
```

Every controller-to-worker file transfer uses:

```bash
gcloud alpha compute tpus tpu-vm scp LOCAL_FILE \
  "$TPU_NAME":/absolute/remote/path \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --worker=all \
  --tunnel-through-iap
```

Operational scripts do not spell those commands independently. They call
`scripts/tpu_iap.sh`, whose only transport operations are `ssh-all` and
`scp-all`, requires the flags above and blocks TPU VM lifecycle/power
commands. Its local-only `check-cli` operation verifies that this alpha
command surface is installed without contacting a worker.

## Air-gapped dependency and source delivery

No worker runs `pip` against an index and no worker clones a repository. An
internet-connected Linux packaging host builds the immutable Sunfish release
archive. The controller transfers that one archive to every worker through
IAP SCP. Workers verify its build-host SHA-256, source identity, URL-free
resolved lock, and wheel inventory, then install with `PIP_NO_INDEX=1`,
`--no-index`, `--no-deps`, `--only-binary=:all:`, and the local wheelhouse.

## Failure modes to design against

- Launching on one worker can leave distributed JAX silently blocked. All pod
  programs therefore use the same `--worker=all` command and run identity.
- A partial/corrupt transfer is rejected independently on every host before
  package installation or JAX initialization.
- Sunfish never changes the allocation lifecycle. Gate 7 interrupts only
  exact recorded user-space training PIDs; it does not stop or reboot a VM.

The public guide was fetched on 2026-07-11 for general JAX pod semantics. This
file records the allocation-specific corrections supplied by Chase on
2026-07-13; `infra/tpu/README.md` is the executable source of truth.
