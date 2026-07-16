# TPU first-run research sweep (C, 2026-07-16)

Six-agent research pass over current official docs (all URLs fetched
2026-07-16), verifying external audit claims and gathering first-pod-run
knowledge. Full sources inline. Companion to
`coordination/third_audit_2026-07-16.md` and `reference/tpu-docs/`.

Post-sweep implementation status: the offline release now carries the pinned
standalone CPython runtime; Gate 4 uses fail-closed JAX `memory_stats()` HBM
evidence; Orbax startup cleanup is enabled; and `requests` is asserted in the
resolved worker lock. `tpu-info` is deliberately not added to the first-run
bundle: introducing another libtpu-coupled dependency this close to launch
would require its own compatibility and air-gap validation. It remains an
optional monitoring follow-up after the pinned readiness path passes.
The detached `setsid`/`nohup` launch advice recorded in section 2 is also
superseded: the implemented contract deliberately keeps every worker attached
to the all-worker SSH launcher so abnormal exit can trigger exact remote
cleanup. Long attempts keep that controller inside an acknowledged durable
tmux session with bounded retries, as specified canonically in
`infra/tpu/README.md`.

## 1. Worker Python — the sharpest fact found

- v4 runtime is `tpu-ubuntu2204-base` (docs.cloud.google.com/tpu/docs/runtimes);
  Ubuntu 22.04 → system CPython 3.10. No Google image for v4–v6e ships 3.12.
- **The pinned `jax==0.10.2` requires Python ≥3.11** (jax 0.7.0 dropped 3.10 in
  July 2025), so the stock image cannot run the pinned stack at all — the
  question is not "3.12 vs 3.10" but "ship an interpreter or don't run."
- **In-contract fix that needs no owner action, no apt, no root, no network:**
  ship astral-sh python-build-standalone `install_only`
  `cpython-3.12.x-x86_64-unknown-linux-gnu` in the offline bundle (glibc floor
  2.17 vs Ubuntu 22.04's 2.35; includes ensurepip/venv). Extract with the
  existing safe-tar path, pin its SHA-256 in the bundle manifest, point
  `SUNFISH_REMOTE_PYTHON_BIN` at it. Build the wheelhouse with the same 3.12
  minor so cp312 tags match.
- `probe_tpu_worker_base.sh` as written asserts 3.12 and **will fail on a
  stock image**. Split it: base checks (disk/tar/proxy) against system
  python3, the 3.12 assertion only against the bundled interpreter after
  deploy. Confirm the actual image with the owner (read-only describe) first.

## 2. Launch discipline — run lifetime vs IAP

- gcloud `--worker=all --command` semantics confirmed verbatim: concurrent on
  all workers (`--batch-size` default "all"); **`--output-directory=DIR` gives
  per-worker `{WORKER_ID}.log`** instead of interleaved stdout; multi-worker
  mode requires `ssh-add` of the Compute Engine key first.
- **IAP idle-drops sessions after 1 hour** and rate-limits bulk TCP forwarding
  (docs.cloud.google.com/iap/docs/using-tcp-forwarding). Two consequences:
  - Long runs must not be children of the SSH session. Verified semantics:
    `--command` runs non-pty → a dropped tunnel sends **no SIGHUP**; the
    orphan dies on its next write (SIGPIPE) **or keeps running silently and
    holds /dev/accel0**. nohup does NOT auto-redirect when stdout is a pipe
    (the gcloud case). Discipline:
    `setsid nohup cmd > host.log 2>&1 < /dev/null &` — fully redirected,
    provably survives disconnects. After any dropped launch, sweep for
    survivors (`sudo lsof -w /dev/accel0`, exact-PID interrupt) before
    relaunching.
  - Consider staging the multi-GB offline bundle via **GCS + Private Google
    Access instead of IAP SCP** (IAP explicitly disclaims bulk transfer).
    The SCP path works but may be slow/rate-limited.
- Never `sudo reboot` via --worker=all (gcloud retries forever, and the
  allocation is not ours to touch).

## 3. First-run failure modes (JAX multi-host)

- Partial launch (subset of workers) → `jax.device_count()` **hangs, no
  error** (docs.cloud.google.com/tpu/docs/jax-pods). `jax.distributed.
  initialize()` (no args on Cloud TPU; 300 s default timeout) is the cheapest
  straggler detector — run a 20-line topology print on all workers before any
  real job (the repo's topology smoke does exactly this — keep it first).
- Stale `/tmp/libtpu_lockfile` from a crashed run → "libtpu.so already in
  use". Diagnose `sudo lsof -w /dev/accel0`; remove lockfile only when no
  holder. Worth adding to the pre-attempt hygiene path.
- Host OOM from Grain input pipelines (workers × threads × prefetch buffers)
  kills one JAX process → **survivors hang forever at the next collective, no
  traceback**. Detect: `dmesg | grep -i oom` per worker + RSS line in the
  host heartbeat. CPU-heavy input → `numactl --cpunodebind=0` (official v4
  guidance).
- Proxy vars cause initialize() timeouts (JAX docs verbatim) — repo already
  fails closed. jax.make_mesh() over hand-reshaped jax.devices() — confirmed
  documented guidance; profile before long runs.
- First-step compile of the 25B teacher / K=4 vmapped trainer can take tens
  of minutes — not a hang; do not kill it.

## 4. Monitoring without Cloud Monitoring agents

- Include **tpu-info** in the offline bundle (pip package; needs matching
  libtpu). `tpu-info --metric duty_cycle_percent hbm_usage --streaming` via a
  side --worker=all command, or `libtpu.sdk.tpumonitoring.get_metric(...)` in
  the heartbeat. Read: HBM near cap → shrink batch; duty_cycle ~0 on ONE host
  → input-bound straggler; **duty_cycle ~0 on ALL hosts → blocked collective,
  usually one dead/OOM'd process**.
- Verbose libtpu logs: `TPU_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0
  TF_CPP_MIN_LOG_LEVEL=0`; harvest `/tmp/tpu_logs/` after any failure.

## 5. Orbax / checkpoint facts (verified at pinned 0.12.1)

- `commit_success.txt` confirmed at 0.12.1 (COMMIT_SUCCESS_FILE in
  atomicity_types.py; auto-selected for gs:// paths). Semi-stable contract —
  fine while pinned; `runtime_api_audit.py` already asserts it. Keep both.
- On GCS an interrupted save leaves a **full-named ckpt_N dir whose only tell
  is the missing marker** (no atomic rename on GCS). All tooling must key on
  the marker — the preemption gate already does.
- **Gap found: `CheckpointManagerOptions.cleanup_tmp_directories` defaults to
  False and neither Kauldron 1.4.4 nor the repo sets it** (verified in the
  actual wheel). A production preemption that kills an in-flight save leaves
  an unfinalized same-name dir; when the resumed run re-saves that step, GCS
  collides (real historical Orbax failure, fixed crash-wise in 0.5.3/0.5.4;
  behavior at 0.12.1 should be validated). Not a gauntlet blocker (gate-7
  kills right after a finalized marker), but resolve before multi-hour runs.
- Async save + exit: manager must be closed / `wait_until_finished()` on every
  exit path or the final save silently loses its marker. Save-interval
  shorter than write time throttles (implicit serialization), never corrupts.
- Grain iterator state is per-process `process_<i>-of-<n>.json` — exact
  resume requires the same process count; never resume onto a different host
  count without a deliberate plan.

## 6. GCS placement (decision-ready)

- **us-central2 has no Cloud Storage presence — confirmed on the official
  locations page.** No allowlist evidence either.
- Multi-region "free same-continent reads" advice is **obsolete** (pricing
  changed 2023-04-01). A **us-central1 regional Standard bucket strictly
  dominates**: same $0.02/GB read to us-central2, no replication charges, no
  higher at-rest rate.
- Enable **hierarchical namespace** (up to 8× initial QPS, atomic folder
  ops — directly helps Orbax and pod-wide step-0 reads). HNS requires uniform
  bucket-level access; incompatible with object versioning/Bucket Lock — cap
  checkpoint spend with max_to_keep + lifecycle rules (already planned).
- Budget: reads out of bucket $0.02/GB (1 TiB epoch re-stream ≈ $20; 16 GB
  checkpoint restore ≈ $0.32); ingress free. TRC FAQ confirmed: **TPUs free,
  storage/VM/egress billed to the project.**
- Air-gap posture confirmed correct: --internal-ips + per-subnet Private
  Google Access + IAP firewall range. GCS via PGA needs no egress exception.

## 7. Offline bundle notes

- `requirements-tpu.lock` is the input pin list (`jax[tpu]==0.10.2`); the
  builder resolves it into `offline-requirements.lock`. jax's `tpu` extra
  pins exact libtpu AND **`requests` (needed by jax.distributed.initialize)**.
  Resolution should capture it, but `--no-deps` install + `pip check` would
  NOT catch a missed extra — add a one-line assertion that `requests` is in
  the resolved lock after every bundle build.
- Add `tpu-info` to the bundle inputs (section 4).

## Repo-docs audit corrections (fix-before-proceeding per PLAN.md's own rule)

1. `evals/stage0/parity-p1-report.json` is **legacy-schema** (no git commit /
   source digest / upstream revision) — `parity_evidence.py` will reject it;
   P1 must be regenerated source-bound along with P2-P5.
2. Recorded calibration corpus is an 11.4M-token pilot with equal bucket caps
   and substitute sources — the repaired runner will reject it (<75M); the
   documented 75M mix has no implementation yet (Stage-1 blocker, not
   gauntlet).
3. PLAN.md "access pending confirmation" vs docs/training.md "handed-off pod"
   — record the actual allocation facts in-repo; nothing does today.
4. docs/upstream_checkpoint.md still shows the stale 25,823,778,864 index
   figure (canonical header sum: 25,823,781,228).
5. Wire seq 22 "TPU VM re-downloads from HF" was superseded by the air-gap
   decision and never marked corrected.
6. PLAN.md requires the cheap-tier eval harness "fixed before training";
   zero eval code exists (needed before Stage-3 recovery, not the gauntlet).
