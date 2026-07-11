#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: launch_tpu_pod.sh --run-id ID --config PATH -- command [args...]" >&2
}

run_id=""
config=""
while (($#)); do
  case "$1" in
    --run-id)
      run_id="${2:-}"
      shift 2
      ;;
    --config)
      config="${2:-}"
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
(($#)) || { echo "missing command after --" >&2; exit 2; }

: "${TPU_NAME:?set TPU_NAME}"
: "${PROJECT_ID:?set PROJECT_ID}"
: "${ZONE:?set ZONE}"
: "${REMOTE_REPO_DIR:?set REMOTE_REPO_DIR to the absolute repo path on every worker}"
: "${EXPECTED_TPU_DEVICES:?set EXPECTED_TPU_DEVICES}"
: "${EXPECTED_TPU_PROCESSES:?set EXPECTED_TPU_PROCESSES}"

quote_arg() {
  printf '%q' "$1"
}

remote_command="cd $(quote_arg "${REMOTE_REPO_DIR}") && scripts/tpu_host_entrypoint.sh"
remote_command+=" --run-id $(quote_arg "${run_id}")"
remote_command+=" --config $(quote_arg "${config}")"
remote_command+=" --expected-devices $(quote_arg "${EXPECTED_TPU_DEVICES}")"
remote_command+=" --expected-processes $(quote_arg "${EXPECTED_TPU_PROCESSES}")"
if [[ -n "${EXPECTED_LOCAL_TPU_DEVICES:-}" ]]; then
  remote_command+=" --expected-local-devices $(quote_arg "${EXPECTED_LOCAL_TPU_DEVICES}")"
fi
remote_command+=" --"
for argument in "$@"; do
  remote_command+=" $(quote_arg "${argument}")"
done

controller_log_dir="${SUNFISH_CONTROLLER_LOG_DIR:-tpu-launch-logs}/${run_id}"
mkdir -p "${controller_log_dir}"
printf '%s\n' "${remote_command}" > "${controller_log_dir}/remote-command.txt"

gcloud_bin="${SUNFISH_GCLOUD_BIN:-gcloud}"
"${gcloud_bin}" compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --worker=all \
  --command "${remote_command}" \
  2>&1 | tee "${controller_log_dir}/all-workers.log"
