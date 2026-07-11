# Distributed data loading + Grain essentials (distilled)

Sources: https://docs.jax.dev/en/latest/distributed_data_loading.html and
google-grain.readthedocs.io dataset tutorial (fetched 2026-07-11)

## The pattern JAX recommends (and the review demands)

**Consolidated per-process pipeline**: ONE loader per process serving all of
that host's local devices; each process loads `global_batch /
process_count` examples; shard locally, then place onto addressable
devices. Never have every host read the full global batch (option 1 —
wasteful) and never run per-device loaders (option 2 — thrashes).

- Build global arrays from local data with
  `jax.make_array_from_process_local_data(sharding, local_batch)`.
- Data parallelism insight: batch ORDER is interchangeable, so processes
  load independently — no cross-host coordination of batch positions
  needed, only disjointness.
- Static batch shapes; drop or pad ragged tails.
- One replica gets one batch — double-check replication math when the mesh
  has replicated axes.

## Grain API surface we build on

```python
ds = grain.MapDataset.source(source)          # source: __getitem__/__len__
ds = ds[jax.process_index()::jax.process_count()]   # process-disjoint shard
ds = ds.shuffle(seed=SEED)                    # global shuffle on MapDataset
ds = ds.batch(batch_size=LOCAL_BATCH)
it = ds.to_iter_dataset(grain.ReadOptions(num_threads=N, prefetch_buffer_size=B)).__iter__()
```

- **Checkpointable iterators**: `it.get_state()` / `it.set_state(state)` —
  index-based, tiny, assumes the underlying data is immutable (hence our
  immutable-shards + manifest-hash policy). This state object is what goes
  into the Orbax checkpoint as loader state.
- Order matters: shuffle/global ops on MapDataset BEFORE
  `to_iter_dataset()` (IterDataset loses random access).
- Filters returning None are fine but expensive if they reject >90%.

## Sunfish mapping

- Sources: the packed uint32 `.bin` shards from
  `scripts/assemble_calibration.py` (and later the recovery/SFT packers)
  uploaded to GCS as immutable objects + manifest (names, counts,
  checksums, provenance).
- Reader: seekable random-access source over fixed-size token records
  (canvas-packed), so `MapDataset.source()` gets true `__getitem__`.
- Disjointness readiness test (gauntlet #2): one synthetic epoch, assert
  the union of record IDs across processes is exact and disjoint.
- GCS access: Cloud Storage FUSE or a native streaming reader; strict local
  cache bound (storage-constrained nodes — external review item 4).
