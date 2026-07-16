#!/usr/bin/env bash
set -euo pipefail

run_id=""
attempt_id=""
config=""
config_sha256=""
expected_devices=""
expected_processes=""
expected_local_devices=""
xla_python_client_preallocate=""
expected_commit=""
source_tree_sha256=""
while (($#)); do
  case "$1" in
    --run-id) run_id="${2:-}"; shift 2 ;;
    --attempt-id) attempt_id="${2:-}"; shift 2 ;;
    --config) config="${2:-}"; shift 2 ;;
    --config-sha256) config_sha256="${2:-}"; shift 2 ;;
    --expected-devices) expected_devices="${2:-}"; shift 2 ;;
    --expected-processes) expected_processes="${2:-}"; shift 2 ;;
    --expected-local-devices) expected_local_devices="${2:-}"; shift 2 ;;
    --xla-python-client-preallocate) xla_python_client_preallocate="${2:-}"; shift 2 ;;
    --expected-commit) expected_commit="${2:-}"; shift 2 ;;
    --source-tree-sha256) source_tree_sha256="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${run_id}" && "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "invalid or missing run ID" >&2
  exit 2
}
[[ -f "${config}" ]] || { echo "config not found: ${config}" >&2; exit 2; }
[[ "${config_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected config SHA-256 is required" >&2
  exit 2
}
derived_bundle_root=""
if [[ -f .sunfish-release.json ]]; then
  derived_bundle_root="$(cd .. && pwd)"
  default_remote_python="${derived_bundle_root}/python/bin/python3"
else
  default_remote_python="python3"
fi
system_python="${SUNFISH_REMOTE_PYTHON_BIN:-${default_remote_python}}"
for proxy_name in \
  HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [[ -n "${!proxy_name:-}" ]]; then
    echo "worker proxy setting ${proxy_name} is forbidden before JAX startup" >&2
    exit 2
  fi
done
if [[ -n "${derived_bundle_root}" ]]; then
  PYTHONPATH=src "${SUNFISH_STOCK_PYTHON_BIN:-python3}" \
    -m sunfish_tpu.standalone_runtime verify-installed \
    --bundle-root "${derived_bundle_root}" \
    --require-bundle-manifest
fi
actual_config_sha256="$("${system_python}" -c \
  'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "${config}")"
[[ "${actual_config_sha256}" == "${config_sha256}" ]] || {
  echo "worker config differs from controller" >&2
  exit 2
}
if [[ -z "${attempt_id}" ]]; then
  attempt_id="${run_id}-direct"
fi
[[ "${attempt_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "invalid attempt ID" >&2
  exit 2
}
[[ -n "${expected_devices}" && -n "${expected_processes}" ]] || {
  echo "expected global devices and processes are required" >&2
  exit 2
}
case "${xla_python_client_preallocate}" in
  true|false) ;;
  *)
    echo "missing or invalid XLA_PYTHON_CLIENT_PREALLOCATE launch value" >&2
    exit 2
    ;;
esac
[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "expected Git commit is required" >&2
  exit 2
}
[[ "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected source-tree SHA-256 is required" >&2
  exit 2
}
read -r actual_commit actual_source_tree_sha256 < <(
  PYTHONPATH=src "${system_python}" -c \
    'import pathlib; from sunfish.source_tree import workspace_source_identity; i=workspace_source_identity(pathlib.Path.cwd()); print(i["git_commit"], i["source_tree_sha256"])'
)
[[ "${actual_commit}" == "${expected_commit}" ]] || {
  echo "worker release commit ${actual_commit} differs from ${expected_commit}" >&2
  exit 2
}
[[ "${actual_source_tree_sha256}" == "${source_tree_sha256}" ]] || {
  echo "worker source tree differs from controller" >&2
  exit 2
}
if [[ -f .sunfish-release.json ]]; then
  [[ -f "${derived_bundle_root}/offline-bundle.json" ]] || {
    echo "exported worker source is not inside an offline bundle" >&2
    exit 2
  }
  if [[ -n "${SUNFISH_OFFLINE_BUNDLE_ROOT:-}" && "${SUNFISH_OFFLINE_BUNDLE_ROOT}" != "${derived_bundle_root}" ]]; then
    echo "worker offline bundle root differs from exported source parent" >&2
    exit 2
  fi
  export SUNFISH_OFFLINE_BUNDLE_ROOT="${derived_bundle_root}"
  export SUNFISH_OFFLINE_BUNDLE_MANIFEST="${derived_bundle_root}/offline-bundle.json"
fi
(($#)) || { echo "missing host command" >&2; exit 2; }

export SUNFISH_RUN_ID="${run_id}"
export SUNFISH_ATTEMPT_ID="${attempt_id}"
export SUNFISH_CONFIG="${config}"
export SUNFISH_CONFIG_FILE_SHA256="${actual_config_sha256}"
export EXPECTED_TPU_DEVICES="${expected_devices}"
export EXPECTED_TPU_PROCESSES="${expected_processes}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${xla_python_client_preallocate}"
export SUNFISH_GIT_COMMIT="${actual_commit}"
export SUNFISH_SOURCE_TREE_SHA256="${actual_source_tree_sha256}"
export SUNFISH_TPU_WORKER=1
if [[ -n "${expected_local_devices}" ]]; then
  export EXPECTED_LOCAL_TPU_DEVICES="${expected_local_devices}"
fi

# A stale libtpu lock or an orphan still holding /dev/accel* makes the next
# distributed initialization hang. This is deliberately read-only: never
# remove the lock or signal an unverified process automatically. Exported TPU
# releases must pass before any workload (including bootstrap) is spawned.
if [[ -n "${derived_bundle_root}" ]]; then
  PYTHONPATH=src "${system_python}" -m sunfish_tpu.worker_hygiene \
    --run-id "${run_id}" \
    --attempt-id "${attempt_id}"
fi

host="$(hostname | tr -c 'A-Za-z0-9._-' '_')"
log_dir="${SUNFISH_HOST_LOG_ROOT:-${HOME}/sunfish-logs}/${run_id}/${attempt_id}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${host}.log"
pid_root="${SUNFISH_PID_ROOT:-${HOME}/.sunfish/pids}"
mkdir -p "${pid_root}"
pid_file="${pid_root}/${run_id}.${attempt_id}.${host}.pid"
[[ ! -e "${pid_file}" && ! -L "${pid_file}" ]] || {
  echo "attempt PID file already exists: ${pid_file}" >&2
  exit 2
}

# Keep this shell as the signal-owning process. A pipeline would run the
# command block in a subshell and leave TERM/HUP/INT delivered to the outer
# SSH command with no deterministic path to the recorded training child. A
# bounded relay preserves the controller stream plus per-host log, but exits
# on an explicit stop marker rather than FIFO EOF: abrupt parent death can
# leave Grain multiprocessing workers holding inherited stdout indefinitely.
log_pipe="${log_file}.pipe.$$"
log_stop="${log_file}.stop.$$"
log_ready="${log_file}.ready.$$"
launch_gate="${log_file}.launch.$$"
for relay_path in "${log_pipe}" "${log_stop}" "${log_ready}" "${launch_gate}"; do
  [[ ! -e "${relay_path}" && ! -L "${relay_path}" ]] || {
    echo "host log relay path already exists: ${relay_path}" >&2
    exit 2
  }
done
mkfifo -m 600 "${log_pipe}"
command_pid=""
forwarded_signal=""
log_relay_pid=""
pid_file_published=""

cleanup_pid_file() {
  local recorded_pid="" extra=""
  [[ -e "${pid_file}" || -L "${pid_file}" ]] || return 0
  if [[ "${pid_file_published}" != 1 ]]; then
    echo "refusing to remove an attempt PID object this entrypoint did not publish: ${pid_file}" >&4
    return 0
  fi
  if [[ ! -f "${pid_file}" || -L "${pid_file}" ]]; then
    echo "refusing to remove unexpected attempt PID object: ${pid_file}" >&4
    return 0
  fi
  if ! read -r recorded_pid extra < "${pid_file}" || \
     [[ ! "${command_pid}" =~ ^[1-9][0-9]*$ || \
        "${recorded_pid}" != "${command_pid}" || -n "${extra}" ]]; then
    echo "refusing to remove attempt PID file whose contents changed: ${pid_file}" >&4
    return 0
  fi
  rm -f -- "${pid_file}"
}

publish_pid_file() {
  local recorded_pid="" extra=""
  # Bash noclobber uses exclusive creation for a new regular file. Combined
  # with the precheck above, this refuses an existing file or symlink that won
  # the race after the child was spawned; it never follows the raced object.
  if ! (umask 077; set -o noclobber; printf '%s\n' "${command_pid}" > "${pid_file}"); then
    return 1
  fi
  if [[ ! -f "${pid_file}" || -L "${pid_file}" ]] || \
     ! read -r recorded_pid extra < "${pid_file}" || \
     [[ "${recorded_pid}" != "${command_pid}" || -n "${extra}" ]]; then
    return 1
  fi
  pid_file_published=1
}

stop_unpublished_child() {
  local unused=""
  [[ "${command_pid}" =~ ^[1-9][0-9]*$ ]] || return 0
  if /bin/kill -0 "${command_pid}" 2>/dev/null; then
    /bin/kill -TERM -- "${command_pid}" 2>/dev/null || true
    for unused in {1..100}; do
      /bin/kill -0 "${command_pid}" 2>/dev/null || break
      sleep 0.05
    done
  fi
  if /bin/kill -0 "${command_pid}" 2>/dev/null; then
    /bin/kill -KILL -- "${command_pid}" 2>/dev/null || true
    for unused in {1..100}; do
      /bin/kill -0 "${command_pid}" 2>/dev/null || break
      sleep 0.05
    done
  fi
  if /bin/kill -0 "${command_pid}" 2>/dev/null; then
    echo "exact unpublished child did not exit after bounded SIGKILL wait: ${command_pid}" >&4
    return 1
  fi
  # The process is already absent, so this wait only consumes Bash's cached
  # status for the exact direct child and cannot block on a live process.
  wait "${command_pid}" 2>/dev/null || true
}

cleanup_host_entrypoint() {
  local exit_status=$?
  local unused=""
  trap '' TERM HUP INT
  cleanup_pid_file
  # The direct child has been reaped before normal exit. Tell the relay to
  # drain bytes already queued, then close this shell's writer. The relay does
  # not wait for descendants that retained the inherited FIFO descriptor.
  if ! (set -o noclobber; : > "${log_stop}"); then
    echo "refusing to replace host log relay stop marker: ${log_stop}" >&4
  elif [[ ! -f "${log_stop}" || -L "${log_stop}" ]]; then
    echo "host log relay stop marker is not a regular file: ${log_stop}" >&4
  fi
  exec 1>&- 2>&-
  if [[ "${log_relay_pid}" =~ ^[1-9][0-9]*$ ]]; then
    for unused in {1..100}; do
      /bin/kill -0 "${log_relay_pid}" 2>/dev/null || break
      sleep 0.05
    done
    if /bin/kill -0 "${log_relay_pid}" 2>/dev/null; then
      /bin/kill -TERM -- "${log_relay_pid}" 2>/dev/null || true
      for unused in {1..100}; do
        /bin/kill -0 "${log_relay_pid}" 2>/dev/null || break
        sleep 0.05
      done
    fi
    if /bin/kill -0 "${log_relay_pid}" 2>/dev/null; then
      /bin/kill -KILL -- "${log_relay_pid}" 2>/dev/null || true
    fi
    wait "${log_relay_pid}" 2>/dev/null || true
  fi
  rm -f -- "${log_pipe}" "${log_stop}" "${log_ready}" "${launch_gate}"
  exec 3>&- 4>&-
  trap - EXIT
  exit "${exit_status}"
}

forward_signal() {
  local signal_name="$1" recorded_pid="" extra=""
  forwarded_signal="${signal_name}"
  [[ "${command_pid}" =~ ^[1-9][0-9]*$ ]] || return 0

  # The in-memory PID is the direct child returned by `$!`. If the published
  # file exists as well, require it to agree before forwarding anything.
  if [[ -e "${pid_file}" || -L "${pid_file}" ]]; then
    if [[ ! -f "${pid_file}" || -L "${pid_file}" ]] || \
       ! read -r recorded_pid extra < "${pid_file}" || \
       [[ "${recorded_pid}" != "${command_pid}" || -n "${extra}" ]]; then
      echo "refusing ${signal_name} forwarding because the attempt PID file changed" >&4
      return 0
    fi
  fi
  if /bin/kill -0 "${command_pid}" 2>/dev/null; then
    /bin/kill "-${signal_name}" -- "${command_pid}" 2>/dev/null || true
  fi
}

trap cleanup_host_entrypoint EXIT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP
trap 'forward_signal INT' INT

# Preserve the original controller descriptors for fail-closed cleanup
# diagnostics. Neither the relay nor the workload may retain these duplicates.
exec 3>&1 4>&2
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env -u SUNFISH_RUN_ID -u SUNFISH_ATTEMPT_ID \
  "${system_python}" "${script_dir}/host_log_relay.py" \
  --pipe "${log_pipe}" \
  --log "${log_file}" \
  --stop "${log_stop}" \
  --ready "${log_ready}" 3>&- 4>&- &
log_relay_pid=$!
relay_started=""
for unused in {1..200}; do
  if [[ -f "${log_ready}" && ! -L "${log_ready}" ]]; then
    read -r relay_ready_pid relay_ready_extra < "${log_ready}" || true
    if [[ "${relay_ready_pid:-}" == "${log_relay_pid}" && -z "${relay_ready_extra:-}" ]]; then
      relay_started=1
      break
    fi
    echo "host log relay ready file is invalid" >&2
    exit 2
  fi
  if ! /bin/kill -0 "${log_relay_pid}" 2>/dev/null; then
    wait "${log_relay_pid}" || true
    echo "host log relay exited before startup" >&2
    exit 2
  fi
  sleep 0.05
done
[[ "${relay_started}" == 1 ]] || {
  echo "timed out waiting for host log relay" >&2
  exit 2
}
exec > "${log_pipe}" 2>&1

echo "sunfish run_id=${run_id} attempt_id=${attempt_id} host=${host} commit=${actual_commit} source_tree=${actual_source_tree_sha256} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=${config} sha256=${actual_config_sha256}"
printf 'command='
printf '%q ' "$@"
printf '\n'

python_bin="${SUNFISH_PYTHON_BIN:-${system_python}}"
if [[ -x "${python_bin}" ]]; then
  "${python_bin}" -m pip freeze > "${log_dir}/${host}.pip-freeze.txt"
fi

# Do not begin a workload if a shutdown signal arrived during setup.
case "${forwarded_signal}" in
  HUP) exit 129 ;;
  INT) exit 130 ;;
  TERM) exit 143 ;;
esac

# The stock-Python waiter never forks: it is replaced in-place by the real
# command only after this shell exclusively publishes its PID. A raced PID
# object therefore cannot create an unrecorded workload descendant.
launch_token="$("${system_python}" -c 'import secrets; print(secrets.token_hex(32))')"
"${system_python}" "${script_dir}/../src/sunfish_tpu/pid_publish_gate.py" \
  --gate "${launch_gate}" \
  --token "${launch_token}" \
  -- "$@" 3>&- 4>&- &
command_pid=$!
if ! publish_pid_file; then
  echo "failed to publish the exact attempt PID file without replacement: ${pid_file}; child_pid=${command_pid}" >&4
  if ! stop_unpublished_child; then
    echo "unpublished pre-exec child cleanup is unproven; owner intervention is required and retry is forbidden" >&4
    exit 126
  fi
  exit 2
fi
if ! "${system_python}" "${script_dir}/../src/sunfish_tpu/pid_publish_gate.py" \
  --gate "${launch_gate}" \
  --token "${launch_token}" \
  --publish; then
  echo "failed to release the exact PID publication gate; child_pid=${command_pid}" >&4
  if ! stop_unpublished_child; then
    echo "published pre-exec child cleanup is unproven; owner intervention is required and retry is forbidden" >&4
    exit 126
  fi
  exit 2
fi
if [[ -n "${forwarded_signal}" ]]; then
  forward_signal "${forwarded_signal}"
fi

set +e
while true; do
  wait "${command_pid}"
  command_status=$?
  if [[ -n "${forwarded_signal}" ]] && \
     /bin/kill -0 "${command_pid}" 2>/dev/null; then
    # A trapped signal interrupts `wait`; wait again until the child has
    # actually observed the forwarded signal and exited.
    continue
  fi
  break
done
set -e
exit "${command_status}"
