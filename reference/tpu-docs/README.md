# TPU reference docs — distilled for offline agent use

Fetched and distilled 2026-07-11 from the authoritative sources (URLs in
each file). These are working notes, not verbatim copies — verify against
the live page if an API detail is load-bearing and the pinned library
version differs.

| File | Covers | Source |
| --- | --- | --- |
| `jax-multihost.md` | distributed.initialize rules, process semantics, meshes, collectives | docs.jax.dev/en/latest/multi_process.html |
| `tpu-pod-launch.md` | gcloud --worker=all install/run procedure, hang modes | docs.cloud.google.com/tpu/docs/jax-pods |
| `orbax-distributed.md` | CheckpointManager, async saves, sharded save/restore | orbax.readthedocs.io checkpoint 101 |
| `data-loading.md` | per-process loading patterns + Grain dataset/checkpoint API | docs.jax.dev distributed_data_loading + google-grain tutorials |

Related in-repo: `coordination/external_tpu_review.md` (the punch list these
docs serve), `docs/sharding_plan.md` (our mesh/partition policy).
