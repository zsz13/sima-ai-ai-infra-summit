#!/usr/bin/env bash
# [MAC] Start the whole Foreman demo.
#
#   DEVKIT_IP=192.168.2.2 \
#   FOREMAN_MODEL=/media/nvme/foreman/models/yolo_26n_mpk.tar.gz \
#   ./scripts/demo.sh
#
# Preflight-checks everything, starts the camera, brings up the two DevKit
# processes and the host console, then waits. Ctrl-C stops what it started.
#
# Nothing here is simulated. If a piece is missing the script says which one and
# stops, rather than showing an empty or fabricated console.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

SLOT="${SLOT:-src2}"
CAMERA="${CAMERA:-0}"
UI_PORT="${FOREMAN_PORT:-8800}"
EDGE_PORT="${FOREMAN_EDGE_PORT:-8100}"
FOREMAN_MODEL="${FOREMAN_MODEL:-/media/nvme/foreman/models/yolo_26n_mpk.tar.gz}"
SKIP_CAMERA="${SKIP_CAMERA:-0}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.55}"

PIDS=()
cleanup() {
  printf '\n'
  step "Shutting down"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
  pkill -f "[a]vfoundation" 2>/dev/null
  # Bracketed patterns so pkill does not match this ssh command line itself.
  dev 'for f in /tmp/foreman-edge.pid /tmp/foreman-genai.pid; do
         [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
       pkill -f "[f]oreman_edge.py"; pkill -f "[l]lima/models.*9998"' 2>/dev/null
  ok "stopped"
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- preflight
step "Preflight"
require_container; ok "SDK container is running"

sdk "curl -sk --max-time 5 https://127.0.0.1:$INSIGHT_PORT/api/health" | grep -q '"status":"ok"' \
  && ok "Neat Insight is healthy" \
  || die "Neat Insight is not healthy. Try: docker exec $SDK_CONTAINER insight-admin restart"

require_devkit; ok "DevKit answers at $DEVKIT_IP"
dev "echo up" >/dev/null 2>&1 \
  || die "SSH to $DEVKIT_USER@$DEVKIT_IP failed. Run on the Mac: sima-cli sdk setup --devkit $DEVKIT_IP"
ok "SSH to the DevKit works"

HOST_IP="$(host_ip_for_devkit)"
[ -n "$HOST_IP" ] || die "Could not work out this Mac's address on the DevKit's subnet."
ok "this Mac is $HOST_IP on the DevKit's network"

dev "test -f '$FOREMAN_MODEL'" || die "detector model not found on the DevKit: $FOREMAN_MODEL
  Stage it with: DEVKIT_IP=$DEVKIT_IP ./scripts/setup-devkit.sh"
ok "detector model present"

dev "test -d /media/nvme/llima/models/whisper-small-a16w8 && \
     ls -d /media/nvme/llima/models/Qwen3-VL-*" >/dev/null 2>&1 \
  && ok "GenAI models present" \
  || die "GenAI models missing. Run: DEVKIT_IP=$DEVKIT_IP ./scripts/setup-devkit.sh"

# Sync the current source to the board. NFS gives /workspace; the rsync fallback
# gives /workspace-rsync. Ask the board which one it actually has.
step "Syncing source to the DevKit"
sdk 'source ~/.devkit-sync.rc 2>/dev/null; cd /workspace && dk sync foreman' >/dev/null 2>&1
if dev "test -d /workspace/foreman" 2>/dev/null; then WS_REMOTE=/workspace
elif dev "test -d /workspace-rsync/foreman" 2>/dev/null; then WS_REMOTE=/workspace-rsync
else die "foreman/ is not visible on the DevKit under /workspace or /workspace-rsync"; fi
ok "source is at $WS_REMOTE/foreman on the board"

# ---------------------------------------------------------------- camera
if [ "$SKIP_CAMERA" = "1" ]; then
  warn "SKIP_CAMERA=1 - expecting something else to feed rtsp://$HOST_IP:$RTSP_PORT/$SLOT"
else
  step "Camera"
  command -v ffmpeg >/dev/null || die "ffmpeg missing. Run: brew install ffmpeg"
  pkill -f "[a]vfoundation" 2>/dev/null; sleep 1
  ./scripts/stream-camera.sh "$CAMERA" "$SLOT" >/tmp/foreman-camera.log 2>&1 &
  PIDS+=($!)
  sleep 5
fi

# The DevKit image has no ffprobe, so geometry is probed here and passed through.
GEO=$(sdk "ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
      -show_entries stream=width,height,avg_frame_rate -of csv=p=0 \
      -i rtsp://127.0.0.1:$RTSP_PORT/$SLOT" 2>/dev/null)
[ -n "$GEO" ] || die "no RTSP stream on $SLOT. See /tmp/foreman-camera.log"
SRC_W=${GEO%%,*}; REST=${GEO#*,}; SRC_H=${REST%%,*}; SRC_FPS=${GEO##*,}; SRC_FPS=${SRC_FPS%%/*}
ok "source is ${SRC_W}x${SRC_H} @ ${SRC_FPS} fps on $SLOT"

# ---------------------------------------------------------------- devkit
step "GenAI server on the DevKit (vision-language + Whisper, on the MLA)"
dev 'if [ -f /tmp/foreman-genai.pid ]; then kill "$(cat /tmp/foreman-genai.pid)" 2>/dev/null; rm -f /tmp/foreman-genai.pid; fi
     pkill -f "[l]lima/models.*9998"' 2>/dev/null; sleep 3
ssh -n -f -o BatchMode=yes "$DEVKIT_USER@$DEVKIT_IP" \
  "cd $WS_REMOTE/foreman && nohup bash scripts/run-genai.sh > /tmp/foreman-genai.log 2>&1"
say "loading models onto the MLA (about a minute)"
for i in $(seq 1 60); do
  if dev "curl -s --max-time 3 http://127.0.0.1:9998/v1/models" 2>/dev/null | grep -q '"vlm"'; then
    ok "GenAI server is serving vlm + asr"; break
  fi
  [ "$i" = 60 ] && die "GenAI server did not come up. On the DevKit: cat /tmp/foreman-genai.log"
  sleep 3
done

step "Edge agent on the DevKit (detector on the MLA)"
dev 'if [ -f /tmp/foreman-edge.pid ]; then kill "$(cat /tmp/foreman-edge.pid)" 2>/dev/null; rm -f /tmp/foreman-edge.pid; fi
     pkill -f "[f]oreman_edge.py"' 2>/dev/null; sleep 2
ssh -n -f -o BatchMode=yes "$DEVKIT_USER@$DEVKIT_IP" \
  "FOREMAN_SOURCE=rtsp://$HOST_IP:$RTSP_PORT/$SLOT \
   FOREMAN_MODEL='$FOREMAN_MODEL' \
   FOREMAN_INSIGHT_HOST=$HOST_IP \
   FOREMAN_WIDTH=$SRC_W FOREMAN_HEIGHT=$SRC_H FOREMAN_FPS=$SRC_FPS \
   nohup bash $WS_REMOTE/foreman/scripts/run-edge.sh --score-threshold $SCORE_THRESHOLD \
   > /tmp/foreman-edge.log 2>&1"
for i in $(seq 1 30); do
  if curl -s --max-time 3 "http://$DEVKIT_IP:$EDGE_PORT/health" | grep -qE '"ok": *true'; then
    ok "edge agent is running"; break
  fi
  [ "$i" = 30 ] && die "edge agent did not come up. On the DevKit: cat /tmp/foreman-edge.log"
  sleep 2
done

# ---------------------------------------------------------------- host
step "Foreman console"
pkill -f "[h]ost.app" 2>/dev/null; sleep 1
FOREMAN_EDGE_URL="http://$DEVKIT_IP:$EDGE_PORT" \
FOREMAN_INSIGHT_URL="https://127.0.0.1:8081/static/viewer.html?src=0" \
./scripts/run-host.sh >/tmp/foreman-host.log 2>&1 &
PIDS+=($!)
for i in $(seq 1 30); do
  curl -s --max-time 2 "http://127.0.0.1:$UI_PORT/api/state" >/dev/null && break
  [ "$i" = 30 ] && die "host UI did not start. See /tmp/foreman-host.log"
  sleep 1
done
ok "console is up"

sleep 3
FPS=$(curl -s "http://127.0.0.1:$UI_PORT/api/state" | sed -n 's/.*"fps": *\([0-9.]*\).*/\1/p')
[ -n "$FPS" ] && ok "detector is running at ${FPS} fps"

cat <<BANNER

  ${c_grn}Foreman is running.${c_off}

    Console        http://127.0.0.1:$UI_PORT
    Engineering    https://127.0.0.1:8081/static/viewer.html?src=0   (Neat Insight)

    On Modalix     detection, vision-language judgement, Whisper speech recognition
    On this Mac    gating, policy, audit trail, UI

  Press "State the standard", say what a good item looks like, then hold one up.
  Ctrl-C stops everything.

BANNER

command -v open >/dev/null && open "http://127.0.0.1:$UI_PORT" 2>/dev/null
wait
