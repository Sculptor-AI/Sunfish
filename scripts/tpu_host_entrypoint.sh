#!/usr/bin/env bash
set -euo pipefail

run_id=""
attempt_id=""
config=""
config_sha256=""
expected_devices=""
expected_processes=""
expected_local_devices=""
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
actual_config_sha256="$(python3 -c \
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
[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "expected Git commit is required" >&2
  exit 2
}
[[ "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected source-tree SHA-256 is required" >&2
  exit 2
}
actual_commit="$(git rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] || {
  echo "worker Git commit ${actual_commit} differs from ${expected_commit}" >&2
  exit 2
}
actual_source_tree_sha256="$(python3 scripts/source_tree_digest.py --root .)"
[[ "${actual_source_tree_sha256}" == "${source_tree_sha256}" ]] || {
  echo "worker source tree differs from controller" >&2
  exit 2
}
(($#)) || { echo "missing host command" >&2; exit 2; }

export SUNFISH_RUN_ID="${run_id}"
export SUNFISH_ATTEMPT_ID="${attempt_id}"
export SUNFISH_CONFIG="${config}"
export SUNFISH_CONFIG_FILE_SHA256="${actual_config_sha256}"
export EXPECTED_TPU_DEVICES="${expected_devices}"
export EXPECTED_TPU_PROCESSES="${expected_processes}"
export SUNFISH_GIT_COMMIT="${actual_commit}"
export SUNFISH_SOURCE_TREE_SHA256="${actual_source_tree_sha256}"
if [[ -n "${expected_local_devices}" ]]; then
  export EXPECTED_LOCAL_TPU_DEVICES="${expected_local_devices}"
fi

host="$(hostname | tr -c 'A-Za-z0-9._-' '_')"
log_dir="${SUNFISH_HOST_LOG_ROOT:-${HOME}/sunfish-logs}/${run_id}/${attempt_id}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${host}.log"
pid_root="${SUNFISH_PID_ROOT:-${HOME}/.sunfish/pids}"
mkdir -p "${pid_root}"
pid_file="${pid_root}/${run_id}.${attempt_id}.${host}.pid"
[[ ! -e "${pid_file}" ]] || { echo "attempt PID file already exists: ${pid_file}" >&2; exit 2; }

{
  echo "sunfish run_id=${run_id} attempt_id=${attempt_id} host=${host} commit=${actual_commit} source_tree=${actual_source_tree_sha256} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "config=${config} sha256=${actual_config_sha256}"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'

  python_bin="${SUNFISH_PYTHON_BIN:-.venv-tpu/bin/python}"
  if [[ -x "${python_bin}" ]]; then
    "${python_bin}" -m pip freeze > "${log_dir}/${host}.pip-freeze.txt"
  fi

  "$@" &
  command_pid=$!
  printf '%s\n' "${command_pid}" > "${pid_file}"
  set +e
  wait "${command_pid}"
  command_status=$?
  set -e
  rm -f "${pid_file}"
  exit "${command_status}"
} 2>&1 | tee -a "${log_file}"
