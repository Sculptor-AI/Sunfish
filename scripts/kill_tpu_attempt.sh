#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: kill_tpu_attempt.sh --run-id ID --attempt-id ID" >&2
}

run_id=""
attempt_id=""
while (($#)); do
  case "$1" in
    --run-id) run_id="${2:-}"; shift 2 ;;
    --attempt-id) attempt_id="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

for value in "${run_id}" "${attempt_id}"; do
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
    echo "invalid run/attempt ID" >&2
    exit 2
  }
done
: "${TPU_NAME:?set TPU_NAME}"
: "${PROJECT_ID:?set PROJECT_ID}"
: "${ZONE:?set ZONE}"

quote_arg() { printf '%q' "$1"; }
pattern_prefix="$(quote_arg "${run_id}.${attempt_id}.")"
remote="pid_root=\"\${SUNFISH_PID_ROOT:-\${HOME}/.sunfish/pids}\"; "
remote+="found=0; for file in \"\${pid_root}\"/${pattern_prefix}*.pid; do "
remote+="[[ -f \"\${file}\" ]] || continue; found=1; pid=\$(cat \"\${file}\"); "
remote+="[[ \"\${pid}\" =~ ^[0-9]+$ ]] || exit 3; kill -KILL \"\${pid}\"; done; "
remote+="[[ \"\${found}\" == 1 ]]"

gcloud_bin="${SUNFISH_GCLOUD_BIN:-gcloud}"
"${gcloud_bin}" compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --worker=all \
  --command "${remote}"
