#!/usr/bin/env bash
# Two-way agent bridge: Claude (Fable 5) <-> Codex (5.6 Sol).
#
# Either agent invokes the other synchronously:
#   coordination/bridge.sh to-codex  "message"   # Claude -> Codex
#   coordination/bridge.sh to-claude "message"   # Codex  -> Claude
#
# The wrapper adds a shared protocol envelope so both sides follow the same
# rules: durable content goes to coordination/channel.md (append-only,
# numbered), decisions go to the AGENTS.md decision log, and exchanges get a
# turn cap so two reasoning models don't loop unattended.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/coordination/bridge.log"
MAX_TURNS=6

direction="${1:-}"
message="${2:-}"
turn="${3:-1}"

if [[ -z "$direction" || -z "$message" ]]; then
  echo "usage: bridge.sh {to-codex|to-claude} \"message\" [turn]" >&2
  exit 2
fi

if (( turn > MAX_TURNS )); then
  echo "bridge: turn cap ($MAX_TURNS) reached — pause for human checkpoint." >&2
  echo "Summarize the exchange in coordination/channel.md and stop." >&2
  exit 3
fi

envelope="[BRIDGE turn ${turn}/${MAX_TURNS}] You are being invoked directly by \
the other Sunfish agent. Protocol: (1) read coordination/channel.md for \
context before acting; (2) if your reply matters beyond this exchange, append \
it to coordination/channel.md with the next message number; (3) durable \
decisions go to the AGENTS.md decision log; (4) to respond interactively, \
call coordination/bridge.sh with the opposite direction and turn $((turn+1)); \
(5) at the turn cap, summarize and stop for human review. Message follows.

"

stamp="$(date '+%Y-%m-%d %H:%M:%S')"

case "$direction" in
  to-codex)
    echo "$stamp claude->codex turn=$turn chars=${#message}" >> "$LOG"
    exec codex exec \
      --cd "$REPO" \
      --sandbox workspace-write \
      "${envelope}${message}"
    ;;
  to-claude)
    echo "$stamp codex->claude turn=$turn chars=${#message}" >> "$LOG"
    cd "$REPO"
    exec claude -p "${envelope}${message}"
    ;;
  *)
    echo "bridge: unknown direction '$direction' (use to-codex or to-claude)" >&2
    exit 2
    ;;
esac
