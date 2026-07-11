# Orbax distributed checkpointing essentials (distilled)

Source: orbax.readthedocs.io checkpoint 101 guide (fetched 2026-07-11)

## CheckpointManager

```python
options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=STEPS)
mngr = ocp.CheckpointManager(path, options=options)   # path may be gs://
```

- `max_to_keep` prunes old checkpoints automatically (our rolling window).
- `item_names` labels composite sub-items (model / optimizer / loader state).

## Async saves — the two rules

- `save()` runs in a background thread BY DEFAULT. Call
  `mngr.wait_until_finished()` before exiting — this is also the
  preemption-safe finalization step (PLAN's "wait for checkpoint
  finalization before exiting at preemption").
- Failure to wait = truncated checkpoints that look present in GCS.

## Sharded arrays across processes

- **All processes call `save()`** on the same manager/step; Orbax
  coordinates the distributed write. Never gate save on process_index.
- Restore specifies the TARGET sharding (via an abstract train state /
  `ocp.args.StandardRestore(abstract_state)`) — not the sharding it was
  saved with. This is what makes load-time resharding (e.g., different
  topology or the sharded-load readiness test) possible.

## Sunfish-specific requirements layered on top

- Checkpoint contents (external review item): model, optimizer, global
  step, RNG streams, FULL loader state (manifest hash, seed, epoch, shard
  sequence, current shard, record offset, packing buffers), sampler
  schedule state, resolved config.
- The distributed smoke test must: init distributed on every host → mesh
  over all global devices → arrays in intended TRAINING shardings → all
  processes in one save → restore with explicit shardings → compare
  addressable shards → resume ONE optimizer update vs uninterrupted control.
