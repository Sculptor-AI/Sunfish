#!/usr/bin/env bash
set -euo pipefail

# Gate-7 process interruption only. This sends SIGKILL to the exact recorded
# Sunfish training PIDs; it cannot stop, reset, reboot, or delete a TPU VM.

usage() {
  echo "usage: interrupt_training_attempt.sh --run-id ID --attempt-id ID" >&2
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

quote_arg() { printf '%q' "$1"; }
pattern_prefix="$(quote_arg "${run_id}.${attempt_id}.")"
quoted_run="$(quote_arg "SUNFISH_RUN_ID=${run_id}")"
quoted_attempt="$(quote_arg "SUNFISH_ATTEMPT_ID=${attempt_id}")"
remote="pid_root=\"\${SUNFISH_PID_ROOT:-\${HOME}/.sunfish/pids}\"; "
remote+="found=0; for file in \"\${pid_root}\"/${pattern_prefix}*.pid; do "
remote+="[[ -f \"\${file}\" ]] || continue; found=1; pid=\$(cat \"\${file}\"); "
remote+="[[ \"\${pid}\" =~ ^[0-9]+$ && \"\${pid}\" -gt 1 && -O \"/proc/\${pid}\" && -r \"/proc/\${pid}/environ\" && -r \"/proc/\${pid}/cmdline\" ]] || exit 3; "
remote+="tr '\\0' '\\n' < \"/proc/\${pid}/environ\" | grep -Fqx $(quote_arg "${quoted_run}") || exit 4; "
remote+="tr '\\0' '\\n' < \"/proc/\${pid}/environ\" | grep -Fqx $(quote_arg "${quoted_attempt}") || exit 4; "
remote+="tr '\\0' '\\n' < \"/proc/\${pid}/cmdline\" | grep -Eq '(^|/)sunfish-train$' || exit 5; "
remote+="/bin/kill -KILL -- \"\${pid}\"; done; [[ \"\${found}\" == 1 ]]"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/tpu_iap.sh" ssh-all --command "${remote}"
