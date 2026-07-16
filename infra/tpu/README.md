# TPU readiness and launch runbook

`coordination/external_tpu_review.md` is authoritative. This runbook is the
executable path implementing it. No TPU training starts until the eight tests
in `PLAN.md` Stage 0.5 pass in order on the granted slice.

## Allocation facts required first

Obtain the accelerator/topology, actual visible device count, host count,
local devices per host, VM image/runtime, preemption policy, allocation dates,
project, zone, attached service account, GCS bucket, and egress restrictions.
Do not infer any of those from the phrase "v4-64".

The allocated TPU VMs have **no public internet access**. All controller-to-
worker SSH and SCP traffic goes through IAP, targets every worker, and uses the
alpha TPU VM command surface. Sunfish never creates, starts, stops, resets,
reboots, or deletes the allocation. Those lifecycle operations remain solely
with the allocation owner; losing this non-preemptible pod could cost a month.

The service account is restricted to the Sunfish GCS prefixes it needs:

- immutable data manifests/shards (read);
- exact-tree Orbax seed checkpoints (read);
- the selected run workdir (read/write).

Conversion and the 51 GB Hugging Face download happen off the TPU VMs. TPU
workers restore the parity-approved Orbax seed directly from GCS; no worker
needs 150 GB of checkpoint scratch or a Hugging Face token.
The JAX seed is materialized off-TPU with `sunfish-orbax-seed` and
`scripts/bootstrap_seed_cpu.sh`; a safetensors directory is never passed to
Orbax. The seed sidecar pins every source/output GCS object's generation,
size, and CRC32C; every TPU restore revalidates the output inventory. See
`docs/training_harness.md` for the exact-tree command and manifest.

Before any TPU command, Stage-0 P1-P5 must be green. The tensor comparison
behind `evals/stage0/parity-p1-report.json` passed, but that file uses the
legacy schema and has no deployable-source identity, so it is not promotable
evidence. Regenerate P1 and execute P2-P5 together on a high-memory CPU host
with `sunfish-parity` exactly as documented in `docs/parity_harness.md`, then
upload the final report/traces/environment record. Do not substitute TPU/JAX
outputs for this same-framework conversion gate. The report must come from
the exact deployable source identity that will be launched; every TPU launcher
rejects a config bundle without a strictly validated all-pass P1-P5 report.

## Fully ordered air-gapped bootstrap

### 1. Probe the base image, then build away from the TPU pod

From the authenticated controller, first prove IAP reaches every worker and
that the immutable VM image supplies Linux x86_64, glibc, stock CPython 3.10 or
newer for the dependency-free predeploy helpers, and enough free disk. The
stock interpreter is not the training runtime and need not provide `venv` or
`ensurepip`. This probe does not import JAX or contact any public network
endpoint from a worker. Before it contacts a worker, it runs the controller
preflight described below:

```bash
export TPU_NAME=YOUR_TPU_NAME
export PROJECT_ID=YOUR_PROJECT
export ZONE=YOUR_ZONE
scripts/probe_tpu_worker_base.sh
```

The probe defaults to a 20 GiB free-space floor. Override
`SUNFISH_MIN_FREE_BYTES` only from an allocation-owner-approved disk budget;
the probe fails the all-worker command if any host is below it. It also rejects
every nonempty `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` setting (including
lowercase forms) because a worker proxy can make distributed JAX initialization
stall before useful diagnostics exist.

If any worker lacks that base, stop here and ask the allocation owner for a
compatible image; do not apt-install, reboot, reset, or recreate the pod. Do
not assume an Ubuntu 22.04 image label is sufficient: the probe is the
authority. A stock CPython 3.10 image is supported: the exact training
interpreter arrives inside the immutable release. Use a connected glibc Linux
x86_64 packaging host with stock CPython 3.10 or newer. Third-party runtime
packages must arrive as prebuilt wheels; the builder refuses source
distributions rather than compiling them against an accidental host ABI. The
bundle derives its minimum glibc version from every native wheel's versioned
manylinux tags; deployment and bootstrap reject an older worker before
importing JAX.
Unversioned `linux_x86_64` wheels are rejected because they carry no portable
glibc contract.

Use an internet-connected Linux x86_64 packaging host such as Colab, Kaggle,
Cloud Build, or a disposable CPU VM. Do not build this large wheelhouse on
Chase's laptop, and never run the builder on a TPU worker:

```bash
# Clean committed checkout on the connected packaging host.
scripts/build_tpu_offline_bundle.sh \
  --connected-build-host \
  --output /tmp/sunfish-tpu-offline-COMMIT.tar
```

The builder is the only TPU release path allowed to contact PyPI or GitHub. It
first downloads python-build-standalone release `20260623`, exact
`cpython-3.12.13+20260623-x86_64-unknown-linux-gnu-install_only.tar.gz`
(111,146,559 bytes; SHA-256
`9fa869d69be54f6b8eeae64272fbd9bb0646e0e1a8da9d80e51ba5a3bee48930`).
It verifies that asset before execution and uses only its
`python/bin/python3` to resolve/build Linux wheels, build Gemma from the
audited 40-character source commit, build the Sunfish wheel, create the fully
resolved URL-free lock, and reconstruct/audit a fresh environment with
`PIP_NO_INDEX=1`. The exact runtime archive and URL-free metadata are part of
the immutable bundle. Copy the resulting archive and `.sha256` sidecar back
to the controller. Workers receive no credentials and resolve no dependencies.

### 2. Configure the non-compute controller

The controller is Chase's laptop, Cloud Shell, or another non-compute machine
with the repository, CPython 3.12, Google Cloud SDK >=344, and one authenticated
`gcloud` account. It does not install JAX or run model code. Configure the
target before the first worker probe; every operational command still passes
the same project and zone explicitly:

```bash
export TPU_NAME=YOUR_TPU_NAME
export PROJECT_ID=YOUR_PROJECT
export ZONE=YOUR_ZONE
gcloud config set project "$PROJECT_ID"

# Required by gcloud's all-worker TPU VM SSH path. Generate the normal Compute
# Engine key first if the authenticated controller does not already have it.
test -f "$HOME/.ssh/google_compute_engine"
ssh-add "$HOME/.ssh/google_compute_engine"

PYTHON_BIN=python3.12 VENV_DIR=.venv-tpu-controller \
  scripts/bootstrap_tpu_controller.sh

# Set these only after the allocation owner has verified each cloud-side fact:
# roles/iap.tunnelResourceAccessor (directly or inherited), TCP/22 ingress from
# 35.235.240.0/20, Private Google Access on the TPU subnet, and least-privilege
# GCS access for the attached TPU service account on the exact Sunfish prefixes.
export SUNFISH_IAP_TUNNEL_ROLE_CONFIRMED=1
export SUNFISH_IAP_SSH_FIREWALL_CONFIRMED=1
export SUNFISH_PRIVATE_GOOGLE_ACCESS_CONFIRMED=1
export SUNFISH_GCS_IAM_CONFIRMED=1
scripts/preflight_tpu_controller.sh

export SUNFISH_OFFLINE_ARCHIVE=/absolute/path/sunfish-tpu-offline-COMMIT.tar
export SUNFISH_REMOTE_RELEASE_DIR=/home/YOUR_USER/sunfish-releases/COMMIT
```

The preflight reads only local CLI configuration, authentication metadata,
the SSH agent, and the committed safety policy. It checks the exact Python
minor version, minimum SDK version, alpha SSH/SCP surface, one active account,
configured project, target formats, the loaded `google_compute_engine` key,
and all four owner confirmations. It does not SSH, SCP, inspect a TPU, or read
a bucket. `probe_tpu_worker_base.sh` runs it again and is the first permitted
TPU contact; do not bypass that probe.

The controller launches every worker simultaneously and reads immutable GCS
evidence. The TPU virtualenv below exists only on TPU workers.

### 3. Deploy source and wheels to every worker through IAP

```bash
scripts/deploy_tpu_offline_bundle.sh \
  --bundle "$SUNFISH_OFFLINE_ARCHIVE" \
  --remote-dir "$SUNFISH_REMOTE_RELEASE_DIR"

export REMOTE_REPO_DIR="$SUNFISH_REMOTE_RELEASE_DIR/source"
export SUNFISH_OFFLINE_BUNDLE_ROOT="$SUNFISH_REMOTE_RELEASE_DIR"
```

The deployer performs exactly the transport pattern required by the
allocation: `gcloud alpha compute tpus tpu-vm ssh --worker=all
--batch-size=all --tunnel-through-iap` and `gcloud alpha compute tpus tpu-vm
scp --worker=all --tunnel-through-iap`. The single archive is copied to every
worker, SHA-256 checked there, safely unpacked, inventory-verified, and
atomically published. Before the runtime exists, extraction uses only the
stock-Python-compatible embedded upload/runtime helpers: they reject absolute
paths, traversal, devices, FIFOs, unsafe links, duplicate members, and
unbounded archive expansion. They verify the pinned runtime archive and
manifest binding, derive `python/` transactionally, hash-bind the installed
tree, and prove it is exact CPython 3.12.13. A second check runs the full bundle
verifier with that interpreter. The staging path is content-addressed and
carries an exact release marker: rerunning the same command reconciles partial
workers, accepts already-published byte-identical workers, and refuses an
unmarked staging path or divergent final directory. It never contacts a
package index and never invokes a TPU lifecycle operation.

The launcher pins both the 40-character Git commit and a deterministic SHA-256 over the deployable
surface: `src/`, `scripts/`, `configs/`, exact lock files, `pyproject.toml`,
and the audited upstream reference. The hash includes tracked and non-ignored
untracked entries, file contents, symlink targets, and permission bits. Mutable
coordination history and prose docs are deliberately outside it. Each host
recomputes both values before importing JAX and
refuses the launch on any mismatch. The exported source carries a local
`.sunfish-release.json`, so workers prove the identity without a Git checkout
or GitHub access. Generated bytecode, logs, and virtualenvs are ignored and do
not perturb the identity.

The bootstrap and host/config entrypoints default to the deployed
`$SUNFISH_OFFLINE_BUNDLE_ROOT/python/bin/python3` (exact CPython 3.12.13), then
use the exact TPU stack in
the bundle's `offline-requirements.lock` and exact Gemma source commit
`09e7b48ae88720f6236b8266c7213eb51bb62b87`. Worker pip is forced to
`PIP_NO_INDEX=1`, `--no-index`, `--no-deps`, `--only-binary=:all:`, and the
local wheelhouse. `--no-deps` is mandatory because Gemma's wheel metadata
contains an online Git dependency; every transitive distribution is already
enumerated in the generated lock. The runtime checks the source-bound bundle,
its wheel-derived glibc floor, the proxy-free worker environment, and every
resolved installed version before training. Before any backend import,
bootstrap also
runs `sunfish-runtime-api-audit`: it parses the installed source files for the
reviewed Gemma/Kauldron/Orbax private signatures and checkpoint/cursor/metric
ordering, then records their file hashes in `tpu-runtime-api-audit.json`.

Once the tiny-data and seed manifests exist, render a fresh immutable config
bundle from the reviewed templates. Never hand-edit the templates for a real
run. The tag must be new for every gauntlet attempt:

```bash
export SUNFISH_RUN_TAG=grant-001
export SUNFISH_STORAGE_ROOT=gs://YOUR_BUCKET/sunfish
export SUNFISH_PARITY_REPORT=/path/to/current-source-stage0-parity-report.json
: "${SUNFISH_PARITY_REPORT:?set the downloaded all-pass P1-P5 report}"

# Off-TPU preparation; tiny.jsonl is the reviewed, already-tokenized fixture.
PYTHONPATH=src python3 -m sunfish_tpu.training.pack_records \
  --input tiny.jsonl \
  --output "/tmp/sunfish-tiny-$SUNFISH_RUN_TAG" \
  --records-per-shard 256 \
  --source stage05-disjoint-input
if gcloud storage ls \
  "$SUNFISH_STORAGE_ROOT/data/tiny-overfit-$SUNFISH_RUN_TAG/manifest.json" \
  >/dev/null 2>&1; then
  echo "refusing to replace an existing tiny-data prefix" >&2
  exit 2
fi
gcloud storage rsync --recursive \
  "/tmp/sunfish-tiny-$SUNFISH_RUN_TAG" \
  "$SUNFISH_STORAGE_ROOT/data/tiny-overfit-$SUNFISH_RUN_TAG"

export EXPECTED_TPU_DEVICES=CONFIRMED_GLOBAL_JAX_DEVICE_COUNT
export EXPECTED_TPU_PROCESSES=CONFIRMED_TPU_VM_HOST_COUNT
export EXPECTED_LOCAL_TPU_DEVICES=CONFIRMED_LOCAL_JAX_DEVICE_COUNT
export SUNFISH_GCS_WORKDIR="$SUNFISH_STORAGE_ROOT/runs"
export SUNFISH_READINESS="$SUNFISH_STORAGE_ROOT/readiness/$SUNFISH_RUN_TAG"
export SUNFISH_SEED="$SUNFISH_STORAGE_ROOT/checkpoints/sunfish-stage05-first32-exact-tree"
export SUNFISH_SEED_MANIFEST="$SUNFISH_STORAGE_ROOT/checkpoints/sunfish-stage05-first32-exact-tree.json"
export SUNFISH_DATA_MANIFEST_SHA256="$(python3 -c \
  'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "/tmp/sunfish-tiny-$SUNFISH_RUN_TAG/manifest.json")"
export SUNFISH_SEED_MANIFEST_SHA256="$(.venv-tpu-controller/bin/python -c \
  'import hashlib,sys; from etils import epath; print(hashlib.sha256(epath.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "$SUNFISH_SEED_MANIFEST")"
export SUNFISH_LOCAL_CONFIG_DIR="/tmp/sunfish-stage05-$SUNFISH_RUN_TAG"
export SUNFISH_REMOTE_CONFIG_DIR="/home/YOUR_USER/sunfish-deploy/$SUNFISH_RUN_TAG"
export VENV_DIR=.venv-tpu

.venv-tpu-controller/bin/sunfish-render-tpu-configs \
  --storage-root "$SUNFISH_STORAGE_ROOT" \
  --run-tag "$SUNFISH_RUN_TAG" \
  --dataset-manifest-sha256 "$SUNFISH_DATA_MANIFEST_SHA256" \
  --seed-manifest-sha256 "$SUNFISH_SEED_MANIFEST_SHA256" \
  --parity-report "$SUNFISH_PARITY_REPORT" \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES" \
  --output-dir "$SUNFISH_LOCAL_CONFIG_DIR"

scripts/upload_tpu_configs.sh \
  --local-dir "$SUNFISH_LOCAL_CONFIG_DIR" \
  --remote-dir "$SUNFISH_REMOTE_CONFIG_DIR"

export SMOKE_RUN_ID="sunfish-stage05-smoke-$SUNFISH_RUN_TAG"
export RESUME_RUN_ID="sunfish-stage05-real-resume-$SUNFISH_RUN_TAG"
export PREEMPT_RUN_ID="sunfish-stage05-preemption-$SUNFISH_RUN_TAG"
export SMOKE_LOCAL_CONFIG="$SUNFISH_LOCAL_CONFIG_DIR/sunfish-smoke.toml"
export SMOKE_REMOTE_CONFIG="$SUNFISH_REMOTE_CONFIG_DIR/sunfish-smoke.toml"
export RESUME_LOCAL_CONFIG="$SUNFISH_LOCAL_CONFIG_DIR/sunfish-resume-smoke.toml"
export RESUME_REMOTE_CONFIG="$SUNFISH_REMOTE_CONFIG_DIR/sunfish-resume-smoke.toml"
export PREEMPT_LOCAL_CONFIG="$SUNFISH_LOCAL_CONFIG_DIR/sunfish-preemption-smoke.toml"
export PREEMPT_REMOTE_CONFIG="$SUNFISH_REMOTE_CONFIG_DIR/sunfish-preemption-smoke.toml"
export SUNFISH_SMOKE_WORKDIR="$SUNFISH_STORAGE_ROOT/runs/stage05-smoke-$SUNFISH_RUN_TAG"
export SUNFISH_PREEMPT_WORKDIR="$SUNFISH_STORAGE_ROOT/runs/stage05-preemption-$SUNFISH_RUN_TAG"
```

The renderer validates all three strict TOMLs, proves they differ only in
run ID/workdir, validates P1-P5 exactness, and records template,
canonical-config, raw-file, parity-report, and source hashes. The uploader
copies and hash-verifies all five bundle files into a temporary directory on
every worker, then atomically publishes them with the same content-addressed
retry protocol as the release bundle. An exact retry is safe after partial
all-worker SCP or finalization; a divergent final directory is never replaced.
`--config` always names the controller copy; `--remote-config` names
the worker copy. The launcher rejects a rendered config whose bytes,
canonical digest, run identity, or source identity no longer matches its
sibling bundle manifest. It validates every config in the bundle, even when
only one is being launched, and refuses an ordinary tracked template.
The renderer also replaces the template batch with the measured global JAX
device count, keeping every readiness run at one example per device. This is
load-bearing on a smaller grant: merely changing a v4-64 template's topology
to v4-32 while leaving batch 32 would double per-device activations. Router
and recovery templates remain the approved v4-64 profile; if the granted
topology differs, do not launch them until a new source-bound production
config uses one example per device and its HBM pilot passes.

Run `scripts/bootstrap_tpu.sh` concurrently on every worker through the same
launcher used for jobs:

```bash
scripts/launch_tpu_pod.sh \
  --run-id "stage05-bootstrap-$SUNFISH_RUN_TAG" \
  --attempt-id "stage05-bootstrap-$SUNFISH_RUN_TAG-001" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  env \
  "SUNFISH_GCS_WORKDIR=$SUNFISH_GCS_WORKDIR" \
  "SUNFISH_OFFLINE_BUNDLE_ROOT=$SUNFISH_OFFLINE_BUNDLE_ROOT" \
  scripts/bootstrap_tpu.sh
```

Its ordering is
binding: stock-Python verification of the installed standalone runtime →
standalone-Python verification of archive/source/wheel inventory → create
environment → offline no-index wheel install → exact installed-environment comparison → static
runtime API audit → distributed JAX init → topology and real collective → GCS
read/list probe → record `pip freeze`.
The first distributed checkpoint smoke is the write/read authorization probe.
Do not import Kauldron or ask JAX for devices before distributed init.

The allocation owner must attach the service account and scopes before handoff.
Sunfish intentionally contains no allocation creation or mutation command.

## All-host launcher

Every pod command goes through `scripts/launch_tpu_pod.sh`. It uses
the guarded `scripts/tpu_iap.sh` wrapper, which expands to `gcloud alpha
compute tpus tpu-vm ssh ... --worker=all --batch-size=all
--tunnel-through-iap`. SSH centrally sets `ServerAliveInterval=30`,
`ServerAliveCountMax=6`, and `TCPKeepAlive=yes`, so a dead tunnel is detected
instead of hanging indefinitely. It gives every host one run ID/config/command,
records the exact remote command, and keeps a separate host log and `pip
freeze`.
Config and release transfers use the matching all-worker IAP SCP form. The
wrapper has no TPU lifecycle subcommand and rejects remote power/lifecycle
commands before invoking `gcloud`.
`scripts/tpu_host_entrypoint.sh` refuses a missing config or topology and
rejects any worker whose commit or source-tree digest differs from the
controller. It separately hashes the selected config bytes on the controller
and every worker, so rendered configs outside Git receive the same protection.
It also requires `XLA_PYTHON_CLIENT_PREALLOCATE` to be an explicit lowercase
boolean and forwards that value as a validated entrypoint argument; the worker
exports it before any JAX-bearing command starts. The checked-in `job.env`
example selects `false`.
Before spawning any workload, the exported host entrypoint performs a
dependency-free, read-only `/proc` hygiene check. It fails if another
current-user PID holds `/dev/accel*`, or if `/tmp/libtpu_lockfile` exists
without a verified current-run/attempt current-user fd owner. It reports PID,
device, and command hashes but never signals a process or removes a lockfile.
Either condition is an allocation-owner-intervention stop: do not retry JAX
initialization until the owner confirms the holder/lock state is safe.
The verified source identity is exported to the launched process and embedded
in every run and readiness artifact.

The worker command intentionally remains attached to SSH. Do not add `nohup`,
`setsid`, a background ampersand, or a detached supervisor without a separately
reviewed status/log/reattach protocol. On an abnormal launcher or tunnel exit,
the controller runs the exact run/attempt PID interrupter with a 120-second
bound, then terminates the isolated local gcloud process group; a failed remote
cleanup returns the non-retryable status 126 and is a hard stop before any
retry. Gate 7 resumes only when the intentionally interrupted first launch
reports signal-style status 137 or 143; status 126, zero, or any unrelated
nonzero status cannot serve as interruption evidence. If either exact remote
cleanup or controller-local process-group cleanup is unproven on an exception
or controller signal, `sunfish-preemption-gate` itself exits 126.

The host entrypoint starts a stock-Python, no-fork publication waiter and
exclusively records that PID before releasing it to `exec` the real workload.
The recorded PID therefore becomes the actual `sunfish-train` PID, while a
raced PID-file object cannot start a workload or descendant. If the blocked
pre-exec child cannot be stopped exactly, the entrypoint exits 126 and forbids
retry.

Every calibration, reconstruction, router, and recovery attempt expected to
run longer than the operator's terminal session must use the durable controller
contract. Start a named tmux session on a powered, sleep-disabled controller
with a stable network path, make the two acknowledgements only after those
conditions are true, and precommit a finite retry budget:

```bash
tmux new-session -s sunfish-production
export SUNFISH_CONTROLLER_STAYS_AWAKE_ACK=1
export SUNFISH_CONTROLLER_NETWORK_STABLE_ACK=1

scripts/launch_tpu_pod.sh \
  --run-id "$PRODUCTION_RUN_ID" \
  --attempt-id "$PRODUCTION_RUN_ID-001" \
  --config "$PRODUCTION_LOCAL_CONFIG" \
  --remote-config "$PRODUCTION_REMOTE_CONFIG" \
  --require-durable-controller \
  --attempt-number 1 \
  --max-attempts 3 \
  -- \
  .venv-tpu/bin/sunfish-train --config "$PRODUCTION_REMOTE_CONFIG"
```

The launcher verifies the active tmux session, both acknowledgements, and
`attempt_number <= max_attempts`, then records the contract beside the
controller log. A retry increments both the attempt ID and attempt number; do
not silently extend the budget. Tmux preserves the controller shell and logs
when the UI terminal disconnects; it does not make controller power or network
loss harmless, which is why Gate 7 and exact cleanup remain mandatory.

Never invoke `python -m kauldron.main` directly: that module calls
`jax.devices()` at import. `sunfish-train` and `sunfish-kauldron` initialize
and validate distributed JAX first.

## Ordered Stage-0.5 commands

### 1. Topology and collective

Run the structured collector after bootstrap. Device, process, and
local-device counts must equal the grant, device ownership must be
process-disjoint, and the real cross-host `psum` must pass on every host.

```bash
export TOPOLOGY_RUN_ID="stage05-topology-$SUNFISH_RUN_TAG"
scripts/launch_tpu_pod.sh \
  --run-id "$TOPOLOGY_RUN_ID" \
  --attempt-id "$TOPOLOGY_RUN_ID-001" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-topology-smoke \
  --output-dir "$SUNFISH_READINESS/topology" \
  --run-id "$TOPOLOGY_RUN_ID" \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES"
```

### 2. Process-disjoint GCS input

The renderer already pinned the prepared fixture's exact manifest digest. The
Grain pipeline slices record IDs by JAX process before shuffle and retains
`record_id` in every batch. Execute the production-loader proof on every host:

```bash
export INPUT_RUN_ID="stage05-input-smoke-$SUNFISH_RUN_TAG"
scripts/launch_tpu_pod.sh \
  --run-id "$INPUT_RUN_ID" \
  --attempt-id "$INPUT_RUN_ID-001" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-input-smoke \
  --config "$SMOKE_REMOTE_CONFIG" \
  --output-dir "$SUNFISH_READINESS/input" \
  --run-id "$INPUT_RUN_ID"
```

The command initializes distributed JAX before importing Kauldron/Grain,
calls the pinned Kauldron 1.4.4 `SunfishData.ds_for_current_process` path,
reads the entire bounded tiny fixture, writes immutable per-host record-ID and
GCS range-read evidence, and has process 0 prove pairwise disjoint/exhaustive
coverage in `summary.json`. All readiness templates enable complete bin/index
SHA-256 verification because this fixture is deliberately tiny. The command
refuses datasets over 10,000 records,
which prevents an accidental full-corpus scan.

### 3. Real 8B target-sharded seed load

This is not the replicated LoRA training layout. It restores the actual
8.11B exact-tree seed directly into the Phase-B partition policy and records
per-host and per-device resident parameter bytes. Every host must hold less
than one full model, and every restored leaf must match its target sharding.

```bash
export SEED_LOAD_RUN_ID="stage05-seed-load-$SUNFISH_RUN_TAG"
scripts/launch_tpu_pod.sh \
  --run-id "$SEED_LOAD_RUN_ID" \
  --attempt-id "$SEED_LOAD_RUN_ID-001" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-seed-load-smoke \
  --seed-path "$SUNFISH_SEED" \
  --seed-manifest-path "$SUNFISH_SEED_MANIFEST" \
  --seed-manifest-sha256 "$SUNFISH_SEED_MANIFEST_SHA256" \
  --evidence-dir "$SUNFISH_READINESS/seed-load" \
  --run-id "$SEED_LOAD_RUN_ID" \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES"
```

### 4. 100-500 update real training smoke

Materialize the readiness-only exact-tree seed first, following
`docs/training_harness.md`. It uses the committed deterministic first-32
selection because the scientific selection cannot exist until Stage 1. The
manifest forbids promotion; it tests infrastructure, not pruning quality.
Validate the rendered local copy without importing JAX:

```bash
.venv-tpu-controller/bin/sunfish-train \
  --config "$SMOKE_LOCAL_CONFIG" \
  --validate-only
```

Launch one complete attempt with an explicit evidence ID:

```bash
scripts/launch_tpu_pod.sh \
  --run-id "$SMOKE_RUN_ID" \
  --attempt-id "$SMOKE_RUN_ID-full" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-train \
  --config "$SMOKE_REMOTE_CONFIG"
```

Analyze the immutable per-step loss, gradient/update norms, and input waits:

```bash
.venv-tpu-controller/bin/sunfish-smoke-evidence \
  --attempt-root "$SUNFISH_SMOKE_WORKDIR/readiness/$SMOKE_RUN_ID-full" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --min-steps 100 \
  --min-relative-loss-reduction 0.10 \
  --max-p95-input-wait-ratio 0.10
```

Gate 4 requires the tiny-set loss median to fall by at least 10%, finite
nonzero gradients and updates, and at least 100 contiguous metric steps.

### 5. Distributed checkpoint save/restore

Run the synthetic sharded-state proof collectively with a unique ID:

```bash
export CHECKPOINT_RUN_ID="stage05-checkpoint-smoke-$SUNFISH_RUN_TAG"
scripts/launch_tpu_pod.sh \
  --run-id "$CHECKPOINT_RUN_ID" \
  --attempt-id "$CHECKPOINT_RUN_ID-001" \
  --config "$SMOKE_LOCAL_CONFIG" \
  --remote-config "$SMOKE_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-checkpoint-smoke \
  --workdir "$SUNFISH_READINESS/state" \
  --evidence-dir "$SUNFISH_READINESS/checkpoint" \
  --run-id "$CHECKPOINT_RUN_ID" \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES"
```

This proves the full distributed Orbax composite mechanics. Gate 3 is still
the separate real-seed restore above, and gate 6 is still the real trainer
comparison below; a synthetic state cannot substitute for either.

### 6. Real-model exact resume

The rendered `sunfish-resume-smoke.toml` has the identical dataset, seed,
model, and topology but its own empty workdir. The diagnostic orchestrates two
sequential Python processes on every host. The first checkpoints after one real
update, records the next uninterrupted update, closes its Orbax manager, and
exits. Only then does the second process initialize distributed JAX, rebuild
the production trainer/manager and Grain iterator, discover and restore step 1,
then compare the next batch, loss, trainable gradients/updates/parameters, full
optimizer/collections/step, and frozen-base invariance exactly on every
addressable shard. Distinct cryptographic process tokens and the immutable
first-process summary are embedded in the final evidence. The readiness merger
recomputes that summary from its host records, reconstructs its exact hash, and
binds every control digest and launcher attempt across the process boundary.
The evidence prefix must be a shared `gs://bucket/prefix`; worker-local paths
are rejected before distributed JAX initialization. A bounded all-host error
broadcast turns any process-0 merge or immutable-write failure into the same
nonzero exit on every worker instead of leaving peers at a later collective.
The outer orchestrator latches TERM/HUP/INT before spawning each phase,
forwards it to the exact child, and applies bounded TERM/KILL cleanup; a
shutdown request can never advance from prepare into resume.

```bash
scripts/launch_tpu_pod.sh \
  --run-id "$RESUME_RUN_ID" \
  --attempt-id "$RESUME_RUN_ID-001" \
  --config "$RESUME_LOCAL_CONFIG" \
  --remote-config "$RESUME_REMOTE_CONFIG" \
  -- \
  .venv-tpu/bin/sunfish-real-resume-smoke \
  --config "$RESUME_REMOTE_CONFIG" \
  --attempt-id "$RESUME_RUN_ID-proof" \
  --evidence-dir "$SUNFISH_READINESS/real-resume" \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES"
```

### 7. Preemption recovery

The rendered `sunfish-preemption-smoke.toml` has the same
model/data/seed/topology but its own fresh run ID and empty workdir. The
controller waits for Orbax's pinned
`commit_success.txt` marker, snapshots the published `sunfish-train` roots and
every transitive current-user descendant before the first signal, verifies
their exact run/attempt environment, command hash, PID, parent PID, and Linux
start time, then sends SIGKILL to only those individual records through
pidfds. It never signals a process group. It then waits a bounded interval and
fails if any same-run/attempt process remains,
proves the finalized checkpoint survived, and relaunches the unchanged config.
No GCS deletion or cursor editing is permitted. The resumed attempt must emit
its first process-0 metric at the checkpoint step (Kauldron's step-label
convention) and must not emit step 0; that metric's run/config/data/seed/source
lineage is embedded in the gate evidence. This distinguishes a real restore
from a silent full retrain that merely reaches the same final checkpoint.

This gate never interrupts a TPU VM or allocation. If a descendant lacks the
exact identity, or a new process appears after the immutable snapshot, the
helper signals nothing unrecorded, reports owner intervention required, and
forbids automated retry. All transport still passes through the
lifecycle-blocking IAP wrapper. The controller bounds every interrupt call
and owns the launcher in a separate local process group, so abnormal cleanup
cannot orphan its gcloud/SSH pipeline. Worker logging also stops on an explicit
regular-file marker rather than FIFO EOF, preventing inherited Grain stdout
descriptors from hanging the all-host launcher.
The Gate-7 orchestrator installs TERM/HUP/INT handling before either launcher
spawn. A signal inside the `Popen` window is latched until the new process
object exists, then exact remote cleanup and bounded local-group teardown run
with repeated signals masked; the isolated launcher cannot escape as an
unowned child.
Each host publishes its direct-child PID with exclusive no-clobber creation;
if publication loses a file/symlink race, it refuses replacement and performs
bounded TERM/KILL/reap of only that unpublished direct child before failing.

```bash
export SUNFISH_REMOTE_CONFIG="$PREEMPT_REMOTE_CONFIG"
.venv-tpu-controller/bin/sunfish-preemption-gate \
  --config "$PREEMPT_LOCAL_CONFIG" \
  --preempt-attempt "$PREEMPT_RUN_ID-interrupt-001" \
  --resume-attempt "$PREEMPT_RUN_ID-recover-001" \
  --preempt-after-step 25 \
  --evidence-uri "$SUNFISH_READINESS/preemption/summary.json"
```

### 8. Input throughput

Use the gate-4 smoke summary only after gates 5-7 pass. Gate 8 requires the
steady-state p95 ratio of maximum host input wait to accelerator step time to
be at most 10%, with zero persistent/local-disk cache bytes. The production
path still performs direct per-record GCS range reads. Its only overlap is the
explicit bounded Grain multiprocessing queue: exactly two already-batched
items per worker, recorded in Gate-8 evidence. This in-memory buffer is not a
dataset cache. The gate-2 range-read baseline is diagnostic context and cannot
pass gate 8 by itself; if the real slice misses the threshold, stop and measure
before changing the cache or file-handle policy.

Gate 4 also records `bytes_in_use`, `peak_bytes_in_use`, and `bytes_limit` from
`memory_stats()` for every local TPU device on every evidence step. Missing or
inconsistent device snapshots fail the gate, as does any peak HBM fraction
above the precommitted 90% ceiling.

Finally merge every immutable artifact into the only Stage-0.5 go/no-go:

```bash
.venv-tpu-controller/bin/sunfish-readiness-ledger \
  --topology "$SUNFISH_READINESS/topology/$TOPOLOGY_RUN_ID/summary.json" \
  --input "$SUNFISH_READINESS/input/$INPUT_RUN_ID/summary.json" \
  --seed-load "$SUNFISH_READINESS/seed-load/$SEED_LOAD_RUN_ID/summary.json" \
  --smoke "$SUNFISH_SMOKE_WORKDIR/readiness/$SMOKE_RUN_ID-full/smoke-summary.json" \
  --checkpoint "$SUNFISH_READINESS/checkpoint/$CHECKPOINT_RUN_ID/summary.json" \
  --real-resume "$SUNFISH_READINESS/real-resume/$RESUME_RUN_ID-proof/summary.json" \
  --preemption "$SUNFISH_READINESS/preemption/summary.json" \
  --run-identity "$SUNFISH_SMOKE_WORKDIR/sunfish-run.json" \
  --preemption-run-identity "$SUNFISH_PREEMPT_WORKDIR/sunfish-run.json" \
  --stage0-parity "$SUNFISH_LOCAL_CONFIG_DIR/stage0-parity-report.json" \
  --config-bundle "$SUNFISH_LOCAL_CONFIG_DIR/rendered-configs.json" \
  --expected-devices "$EXPECTED_TPU_DEVICES" \
  --expected-processes "$EXPECTED_TPU_PROCESSES" \
  --expected-local-devices "$EXPECTED_LOCAL_TPU_DEVICES" \
  --output "$SUNFISH_READINESS/stage05-readiness-ledger.json"
```

Only `passed: true` in that ledger unlocks Stage 1. The ledger hashes every
input, re-runs the gate-1/2/3/5/6 host-evidence mergers, enforces the fixed
gate-4/8 thresholds, validates the metric-proven gate-7 continuation, binds
all three raw/canonical rendered configs, and revalidates the source-bound
Stage-0 P1-P5 report. It rejects synthetic substitutes for the real seed or
real resume gates.

## Checkpoint policy

- Every save goes to the run's GCS workdir through Orbax/Kauldron.
- Default retention is three full checkpoints; interval comes from the strict
  run TOML and is converted from the approved wall-clock target after the
  throughput smoke.
- Clean training-process exit waits for async finalization. Abrupt process
  interruption loses only
  work after the last finalized checkpoint; restart restores automatically.
- Manager construction removes only Orbax temporary checkpoint directories
  left by an interrupted asynchronous save. Finalized checkpoints are never
  selected by that cleanup path, and operators must not delete GCS state.
- No multi-hour job starts before the real exact-resume comparison passes.

See `docs/training_harness.md` for phase masks, prefix-amortized multi-noise
training, record/loss-mask semantics, and the initial checkpoint contract.
