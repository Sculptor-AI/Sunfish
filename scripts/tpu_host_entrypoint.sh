#!/usr/bin/env bash
set -euo pipefail

run_id=""
config=""
expected_devices=""
expected_processes=""
expected_local_devices=""
while (($#)); do
  case "$1" in
    --run-id) run_id="${2:-}"; shift 2 ;;
    --config) config="${2:-}"; shift 2 ;;
    --expected-devices) expected_devices="${2:-}"; shift 2 ;;
    --expected-processes) expected_processes="${2:-}"; shift 2 ;;
    --expected-local-devices) expected_local_devices="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${run_id}" && "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "invalid or missing run ID" >&2
  exit 2
}
[[ -f "${config}" ]] || { echo "config not found: ${config}" >&2; exit 2; }
[[ -n "${expected_devices}" && -n "${expected_processes}" ]] || {
  echo "expected global devices and processes are required" >&2
  exit 2
}
(($#)) || { echo "missing host command" >&2; exit 2; }

export SUNFISH_RUN_ID="${run_id}"
export SUNFISH_CONFIG="${config}"
export EXPECTED_TPU_DEVICES="${expected_devices}"
export EXPECTED_TPU_PROCESSES="${expected_processes}"
if [[ -n "${expected_local_devices}" ]]; then
  export EXPECTED_LOCAL_TPU_DEVICES="${expected_local_devices}"
fi

host="$(hostname | tr -c 'A-Za-z0-9._-' '_')"
log_dir="${SUNFISH_HOST_LOG_ROOT:-${HOME}/sunfish-logs}/${run_id}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${host}.log"

{
  echo "sunfish run_id=${run_id} host=${host} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if command -v sha256sum >/dev/null 2>&1; then
    config_sha256="$(sha256sum "${config}" | awk '{print $1}')"
  else
    config_sha256="$(shasum -a 256 "${config}" | awk '{print $1}')"
  fi
  echo "config=${config} sha256=${config_sha256}"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'

  python_bin="${SUNFISH_PYTHON_BIN:-.venv-tpu/bin/python}"
  if [[ -x "${python_bin}" ]]; then
    "${python_bin}" -m pip freeze > "${log_dir}/${host}.pip-freeze.txt"
  fi

  "$@"
} 2>&1 | tee -a "${log_file}"
