#!/usr/bin/env bash
set -euo pipefail

# Validate the controller-side lifetime contract for a long TPU attempt.  This
# deliberately does not detach the worker command: the foreground SSH/process
# relationship is what lets tpu_host_entrypoint.sh own signals and cleanup.

usage() {
  echo "usage: require_durable_controller.sh --attempt-number N --max-attempts N" >&2
}

attempt_number=""
max_attempts=""
while (($#)); do
  case "$1" in
    --attempt-number)
      attempt_number="${2:-}"
      shift 2
      ;;
    --max-attempts)
      max_attempts="${2:-}"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ "${attempt_number}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--attempt-number must be a positive integer" >&2
  exit 2
}
[[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-attempts must be a positive integer" >&2
  exit 2
}
((attempt_number <= max_attempts)) || {
  echo "attempt ${attempt_number} exceeds the precommitted ${max_attempts}-attempt budget" >&2
  exit 2
}

[[ -n "${TMUX:-}" ]] || {
  echo "long TPU attempts must be launched from inside tmux" >&2
  exit 2
}
[[ "${SUNFISH_CONTROLLER_STAYS_AWAKE_ACK:-}" == "1" ]] || {
  echo "set SUNFISH_CONTROLLER_STAYS_AWAKE_ACK=1 only after disabling controller sleep and arranging stable power" >&2
  exit 2
}
[[ "${SUNFISH_CONTROLLER_NETWORK_STABLE_ACK:-}" == "1" ]] || {
  echo "set SUNFISH_CONTROLLER_NETWORK_STABLE_ACK=1 only after arranging a stable controller network path" >&2
  exit 2
}

tmux_bin="${SUNFISH_TMUX_BIN:-tmux}"
session="$("${tmux_bin}" display-message -p '#{session_id}:#{session_name}')" || {
  echo "unable to query the active tmux session" >&2
  exit 2
}
[[ "${session}" =~ ^\$[0-9]+:[A-Za-z0-9._-]+$ ]] || {
  echo "tmux returned an invalid active-session identity" >&2
  exit 2
}

printf 'schema_version=1\n'
printf 'purpose=sunfish-durable-controller-contract\n'
printf 'tmux_session=%s\n' "${session}"
printf 'attempt_number=%s\n' "${attempt_number}"
printf 'max_attempts=%s\n' "${max_attempts}"
printf 'controller_stays_awake_ack=1\n'
printf 'controller_network_stable_ack=1\n'
printf 'ssh_server_alive_interval_seconds=30\n'
printf 'ssh_server_alive_count_max=6\n'
printf 'cleanup_hard_stop_exit_status=126\n'
printf 'cleanup_hard_stop_retry_allowed=0\n'
