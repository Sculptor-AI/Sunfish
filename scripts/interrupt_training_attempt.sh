#!/usr/bin/env bash
set -euo pipefail

# Gate-7 process interruption only. This snapshots every current-user
# descendant of the published sunfish-train roots, verifies exact run/attempt
# identity, and sends SIGKILL only through those individual PID records. It
# never signals a process group or stops/resets/reboots/deletes a TPU VM.

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
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${script_dir}/.." && pwd)"
controller_python="${SUNFISH_CONTROLLER_PYTHON:-python3}"
stock_python="${SUNFISH_STOCK_PYTHON_BIN:-python3}"
helper_path="${root}/src/sunfish_tpu/exact_process_interrupt.py"
helper_b64="$("${controller_python}" -c \
  'import base64,pathlib,sys; print(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode("ascii"))' \
  "${helper_path}")"
helper_bootstrap='import base64,sys; source=base64.b64decode(sys.argv.pop(1),validate=True); exec(compile(source,"sunfish-exact-process-interrupt.py","exec"))'
remote="pid_root=\"\${SUNFISH_PID_ROOT:-\${HOME}/.sunfish/pids}\"; "
remote+="$(quote_arg "${stock_python}") -c $(quote_arg "${helper_bootstrap}")"
remote+=" $(quote_arg "${helper_b64}")"
remote+=" --run-id $(quote_arg "${run_id}")"
remote+=" --attempt-id $(quote_arg "${attempt_id}")"
remote+=" --pid-root \"\${pid_root}\""

"${script_dir}/tpu_iap.sh" ssh-all --command "${remote}"
