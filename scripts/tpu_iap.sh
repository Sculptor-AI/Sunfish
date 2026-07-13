#!/usr/bin/env bash
set -euo pipefail

# The only Sunfish transport allowed to TPU workers. This wrapper deliberately
# exposes no create/start/stop/reset/delete lifecycle operation.

usage() {
  echo "usage: tpu_iap.sh check-cli | ssh-all --command COMMAND | scp-all LOCAL_FILE REMOTE_ABSOLUTE_PATH" >&2
}

gcloud_bin="${SUNFISH_GCLOUD_BIN:-gcloud}"
operation="${1:-}"
shift || true

case "${operation}" in
  check-cli)
    (($# == 0)) || { usage; exit 2; }
    "${gcloud_bin}" alpha compute tpus tpu-vm ssh --help >/dev/null
    "${gcloud_bin}" alpha compute tpus tpu-vm scp --help >/dev/null
    echo "gcloud alpha TPU VM IAP command surface is installed"
    ;;
  ssh-all)
    : "${TPU_NAME:?set TPU_NAME}"
    : "${PROJECT_ID:?set PROJECT_ID}"
    : "${ZONE:?set ZONE}"
    [[ "${1:-}" == "--command" && $# == 2 ]] || { usage; exit 2; }
    command="${2}"
    [[ -n "${command}" && "${command}" != *$'\n'* ]] || {
      echo "remote command must be one nonempty line" >&2
      exit 2
    }
    lower_command="$(printf '%s' "${command}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lower_command}" =~ (^|[[:space:];|&])(sudo[[:space:]]+)?(/(usr/)?s?bin/)?(shutdown|poweroff|reboot|halt|init|telinit)([[:space:];|&]|$) ]] || \
       [[ "${lower_command}" =~ tpu-vm[[:space:]]+(create|start|stop|restart|reset|delete|update|suspend|resume) ]] || \
       [[ "${lower_command}" =~ systemctl[^';|&']*(poweroff|reboot|halt|kexec|soft-reboot) ]] || \
       [[ "${lower_command}" =~ (^|[[:space:];|&])(sudo[[:space:]]+)?(/bin/)?kill[[:space:]][^';|&']*[[:space:]]1([[:space:];|&]|$) ]] || \
       [[ "${lower_command}" == *"/proc/sysrq-trigger"* ]]; then
      echo "refusing a command that could alter the TPU VM allocation lifecycle" >&2
      exit 2
    fi
    if [[ "${lower_command}" =~ https?://|git\+https ]] || \
       [[ "${lower_command}" =~ scripts/(build_tpu_offline_bundle|bootstrap_seed_cpu|bootstrap_parity|bootstrap_tpu_controller)\.sh ]] || \
       [[ "${lower_command}" =~ (^|[[:space:];|&])(sudo[[:space:]]+)?(/usr/bin/)?(curl|wget)([[:space:];|&]|$) ]] || \
       [[ "${lower_command}" =~ (^|[[:space:];|&])git[[:space:]]+clone([[:space:];|&]|$) ]] || \
       [[ "${lower_command}" =~ (^|[[:space:];|&])((python[0-9.]*)[[:space:]]+-m[[:space:]]+)?pip[0-9.]*[[:space:]]+install([[:space:];|&]|$) ]] || \
       [[ "${lower_command}" =~ (^|[[:space:];|&])(sudo[[:space:]]+)?(apt|apt-get|dnf|yum|conda|mamba)[[:space:]] ]]; then
      echo "refusing a public-network or package-resolution command on an air-gapped worker" >&2
      exit 2
    fi
    exec "${gcloud_bin}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
      --project "${PROJECT_ID}" \
      --zone "${ZONE}" \
      --worker=all \
      --batch-size=all \
      --tunnel-through-iap \
      --command "${command}"
    ;;
  scp-all)
    : "${TPU_NAME:?set TPU_NAME}"
    : "${PROJECT_ID:?set PROJECT_ID}"
    : "${ZONE:?set ZONE}"
    [[ $# == 2 ]] || { usage; exit 2; }
    local_file="${1}"
    remote_path="${2}"
    [[ -f "${local_file}" ]] || { echo "local transfer file not found" >&2; exit 2; }
    [[ "${remote_path}" == /* && "${remote_path}" != *$'\n'* ]] || {
      echo "remote transfer path must be absolute and one line" >&2
      exit 2
    }
    exec "${gcloud_bin}" alpha compute tpus tpu-vm scp "${local_file}" \
      "${TPU_NAME}:${remote_path}" \
      --project "${PROJECT_ID}" \
      --zone "${ZONE}" \
      --worker=all \
      --tunnel-through-iap
    ;;
  *)
    usage
    exit 2
    ;;
esac
