# Cloud TPU pod-slice launch procedure (distilled)

Source: https://docs.cloud.google.com/tpu/docs/jax-pods (fetched 2026-07-11)

## Install on every worker

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" \
  --worker=all \
  --command='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
```

`--worker=all` executes on every host of the slice. For Sunfish, replace the
pip line with our pinned bootstrap (`bash scripts/bootstrap_tpu.sh`).

## Ship code + run on every worker

```bash
gcloud compute tpus tpu-vm scp ./program.py "$TPU_NAME": \
  --worker=all --zone="$ZONE" --project="$PROJECT_ID"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" \
  --worker=all \
  --command="python3 ./program.py"
```

(Sunfish will `git clone` + run the module instead of scp-ing single files;
the launch pattern is identical. Same run ID/config/paths on every worker.)

## Failure modes to design against

- **Run on one worker only** → `jax.device_count()` (and any collective)
  BLOCKS until the same code runs on every host. Symptom of a partial
  launch is a silent hang, not an error message.
- Duplicate output: gate prints/logging on `jax.process_index() == 0`;
  per-worker log files should carry the worker index in the name.

## Version caveat

The guide targets the classic Cloud TPU API and explicitly does not support
TPU7x+. Fine for v4/v5e; revisit if the grant lands on newer hardware.
