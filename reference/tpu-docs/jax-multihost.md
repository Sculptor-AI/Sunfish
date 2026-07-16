# JAX multi-host essentials (distilled)

Sources: https://docs.jax.dev/en/latest/multi_process.html and
https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html
(verified 2026-07-16)

## jax.distributed.initialize() — the iron rule

- MUST be called before `jax.devices()`, `jax.local_devices()`, or ANY
  device computation — including inside imported modules. It raises if
  devices were already touched; worse, code that touched the backend first
  cannot discover non-local devices and silently runs single-host.
- On Cloud TPU (also Slurm/K8s): call with NO arguments — coordinator
  address, process id, and count are autodetected from the environment.
- Manual form: `jax.distributed.initialize(coordinator_address="host:port",
  num_processes=N, process_id=i)`.
- Reject or unset `HTTP_PROXY`, `HTTPS_PROXY`, and equivalent proxy variables
  before launch. JAX explicitly warns that they can make distributed
  initialization time out; Sunfish fails closed rather than waiting through an
  ambiguous five-minute initialization timeout.

## Process semantics

- Every host runs the SAME script; JAX does not launch processes for you
  (that's the gcloud --worker=all layer).
- `jax.process_index()` ∈ [0, process_count). Use it ONLY to gate I/O
  (logging, metrics upload) — never to gate collectives, array creation, or
  any computation. All processes must apply the same ops in the same order
  or the job deadlocks (one process skipping a collective hangs the rest).
- `jax.devices()` = ALL devices across all hosts (global).
  `jax.local_devices()` = this host's devices only (addressable). Local
  sets are disjoint; a device belongs to exactly one process.

## Meshes

- `jax.make_mesh((sizes...), ('names'...))` spans all global devices with a
  performant device ordering.
- Fetching values of an array spanning non-addressable devices is
  impossible — read `.addressable_shards`, or replicate (spec `P(None)`)
  across ALL processes first (never conditionally on process_index).
- Cross-process `jax.device_put()` must be called on all processes
  participating in source or destination sharding, even those with no
  local data; source/destination must have identical device counts and
  shard shapes.

## Gotchas checklist (for preflight + code review)

1. init before ANY backend access, including transitive imports
   (kauldron.main touches jax.devices() at import — see external review).
2. Same ops, same order, every process.
3. process_index gates I/O only.
4. Verify: expected global device count, expected process count, unique
   process_index values, expected local device count, and one REAL
   collective (psum over a sharded array) — not a local 1+1.
5. Verify proxy variables are absent before importing JAX.
