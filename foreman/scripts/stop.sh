#!/usr/bin/env bash
# [MAC] Stop everything Foreman started. Safe to run at any time.
#
# Every pkill pattern is bracketed ("[f]oreman") so it cannot match the command
# line of the shell running it - an unbracketed `pkill -f foreman_edge.py` sent
# over ssh kills its own session and leaves the target running.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

step "Host processes"
pkill -f "[h]ost.app"         2>/dev/null && ok "console stopped"       || say "console not running"
pkill -f "[s]tream-camera.sh" 2>/dev/null
pkill -f "[a]vfoundation"     2>/dev/null && ok "camera stream stopped" || say "camera not streaming"
pkill -f "[t]ests.fake_edge"  2>/dev/null

step "DevKit processes"
if ping -c 1 -W 2000 "$DEVKIT_IP" >/dev/null 2>&1; then
  # Stop by PID file first. The GenAI server runs as `python3 -` (a heredoc), so
  # no pkill pattern based on the script name matches it - three copies once
  # accumulated and took CmaFree from 1.6 GB to 15 MB. The pattern kills below
  # are a backstop for processes started before PID files existed.
  dev 'for f in /tmp/foreman-edge.pid /tmp/foreman-genai.pid; do
         [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done' 2>/dev/null
  dev 'pkill -f "[f]oreman_edge.py"; pkill -f "[l]lima/models.*9998"' 2>/dev/null
  sleep 3
  left=$(dev 'echo $(( $(pgrep -f "[f]oreman_edge.py" | wc -l) + $(pgrep -f "[l]lima/models" | wc -l) ))' 2>/dev/null)
  if [ "${left:-0}" = "0" ]; then ok "edge agent and GenAI server stopped"
  else warn "$left DevKit process(es) still running"; fi
else
  warn "DevKit unreachable at $DEVKIT_IP; nothing stopped there"
fi

step "Remaining"
say "Insight and the SDK container are left running (they are shared services)."
say "To stop Insight media sources:"
say "  docker exec $SDK_CONTAINER curl -sk -X POST https://127.0.0.1:$INSIGHT_PORT/api/mediasrc/stop-all"
ok "done"
