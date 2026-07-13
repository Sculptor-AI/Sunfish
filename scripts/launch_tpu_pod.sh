#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: launch_tpu_pod.sh --run-id ID [--attempt-id ID] --config LOCAL_PATH [--remote-config REMOTE_PATH] -- command [args...]" >&2
}

run_id=""
attempt_id=""
config=""
remote_config=""
while (($#)); do
  case "$1" in
    --run-id)
      run_id="${2:-}"
      shift 2
      ;;
    --attempt-id)
      attempt_id="${2:-}"
      shift 2
      ;;
    --config)
      config="${2:-}"
      shift 2
      ;;
    --remote-config)
      remote_config="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -n "${run_id}" && "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "invalid or missing --run-id" >&2
  exit 2
}
[[ -n "${config}" ]] || { echo "missing --config" >&2; exit 2; }
[[ -f "${config}" ]] || { echo "config not found on controller: ${config}" >&2; exit 2; }
if [[ -z "${remote_config}" ]]; then
  remote_config="${config}"
fi
[[ -n "${remote_config}" ]] || { echo "missing remote config path" >&2; exit 2; }
(($#)) || { echo "missing command after --" >&2; exit 2; }
if [[ -z "${attempt_id}" ]]; then
  attempt_id="${run_id}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${attempt_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "invalid --attempt-id" >&2
  exit 2
}

: "${TPU_NAME:?set TPU_NAME}"
: "${PROJECT_ID:?set PROJECT_ID}"
: "${ZONE:?set ZONE}"
: "${REMOTE_REPO_DIR:?set REMOTE_REPO_DIR to the absolute repo path on every worker}"
: "${EXPECTED_TPU_DEVICES:?set EXPECTED_TPU_DEVICES}"
: "${EXPECTED_TPU_PROCESSES:?set EXPECTED_TPU_PROCESSES}"

git_commit="$(git rev-parse HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "controller is not on a valid Git commit" >&2
  exit 2
}
controller_python="${SUNFISH_CONTROLLER_PYTHON:-python3}"
source_tree_sha256="$("${controller_python}" scripts/source_tree_digest.py --root .)"
[[ "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to compute controller source-tree digest" >&2
  exit 2
}
PYTHONPATH=src "${controller_python}" -c \
  'import pathlib,sys; from sunfish_tpu.deployment_config import validate_rendered_config_file; validate_rendered_config_file(pathlib.Path(sys.argv[1]), source_root=pathlib.Path(sys.argv[2]), require_bundle=True)' \
  "${config}" "$(pwd)"
config_sha256="$("${controller_python}" -c \
  'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "${config}")"
[[ "${config_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "failed to compute controller config SHA-256" >&2
  exit 2
}

quote_arg() {
  printf '%q' "$1"
}

remote_command="cd $(quote_arg "${REMOTE_REPO_DIR}") && scripts/tpu_host_entrypoint.sh"
remote_command+=" --run-id $(quote_arg "${run_id}")"
remote_command+=" --attempt-id $(quote_arg "${attempt_id}")"
remote_command+=" --config $(quote_arg "${remote_config}")"
remote_command+=" --config-sha256 $(quote_arg "${config_sha256}")"
remote_command+=" --expected-devices $(quote_arg "${EXPECTED_TPU_DEVICES}")"
remote_command+=" --expected-processes $(quote_arg "${EXPECTED_TPU_PROCESSES}")"
remote_command+=" --expected-commit $(quote_arg "${git_commit}")"
remote_command+=" --source-tree-sha256 $(quote_arg "${source_tree_sha256}")"
if [[ -n "${EXPECTED_LOCAL_TPU_DEVICES:-}" ]]; then
  remote_command+=" --expected-local-devices $(quote_arg "${EXPECTED_LOCAL_TPU_DEVICES}")"
fi
remote_command+=" --"
for argument in "$@"; do
  remote_command+=" $(quote_arg "${argument}")"
done

controller_log_dir="${SUNFISH_CONTROLLER_LOG_DIR:-tpu-launch-logs}/${run_id}/${attempt_id}"
mkdir -p "${controller_log_dir}"
printf '%s\n' "${remote_command}" > "${controller_log_dir}/remote-command.txt"

gcloud_bin="${SUNFISH_GCLOUD_BIN:-gcloud}"
"${gcloud_bin}" compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --worker=all \
  --command "${remote_command}" \
  2>&1 | tee "${controller_log_dir}/all-workers.log"
