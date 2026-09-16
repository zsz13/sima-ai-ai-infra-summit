#!/usr/bin/env bash
# Palette Neat environment smoke test.
# Run from the macOS HOST:  ./scripts/smoke-test.sh
#
# Stages 1-5 need no DevKit. Stage 6 needs a paired DevKit and is reported as
# SKIP (not FAIL) while unpaired, so the script stays useful during bring-up.

set -uo pipefail

CONTAINER="${NEAT_CONTAINER:-ghcr.io-sima-neat-sdk-v2.1.3.0}"
# Internet Sharing gives the DevKit a 192.168.2.x DHCP address (Game Day Guide
# Appendix 2). Override with DEVKIT_IP=... for the static 192.168.2.2 setup.
DEVKIT_IP="${DEVKIT_IP:-192.168.2.2}"
VIDEO="${SMOKE_VIDEO:-video03.mp4}"
SRC_INDEX="${SMOKE_SRC_INDEX:-1}"

pass=0; fail=0; skip=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
no()   { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
sk()   { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; skip=$((skip+1)); }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
sdk()  { docker exec -u d "$CONTAINER" bash -lc "$1" 2>/dev/null; }
# `dk` is a bash function sourced from ~/.devkit-sync.rc, so it only exists in a
# login shell for the paired user - `which dk` returning nothing is normal.
sdk_dk() { docker exec -u d "$CONTAINER" bash -lc "source ~/.devkit-sync.rc 2>/dev/null; $1" 2>/dev/null; }

hdr "1. SDK container"
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  ok "container '$CONTAINER' is running"
else
  no "container '$CONTAINER' is NOT running  ->  sima-cli sdk start"
  echo; echo "Cannot continue without the container."; exit 1
fi

hdr "2. Workspace bind mount (host <-> SDK)"
token="smoke-$$-$(date +%s)"
echo "$token" > "$HOME/workspace/.smoke-probe" 2>/dev/null
if [ "$(sdk 'cat /workspace/.smoke-probe')" = "$token" ]; then
  ok "host ~/workspace is visible inside SDK as /workspace"
else
  no "workspace mount is broken"
fi
rm -f "$HOME/workspace/.smoke-probe"

hdr "3. Neat Insight service"
health=$(sdk 'curl -sk --max-time 5 https://127.0.0.1:9900/api/health')
if echo "$health" | grep -q '"status":"ok"'; then
  ok "Insight healthy on https://127.0.0.1:9900"
else
  no "Insight not healthy  ->  docker exec $CONTAINER insight-admin restart"
fi

hdr "4. RTSP media source"
state=$(sdk "curl -sk --max-time 5 https://127.0.0.1:9900/api/mediasrc" \
        | tr ',' '\n' | grep -A2 "\"index\":$SRC_INDEX" | grep -o '"state":"[a-z]*"' | head -1)
if [ -z "$state" ] || ! echo "$state" | grep -q playing; then
  sk "src$SRC_INDEX not playing; attempting to assign '$VIDEO' and start"
  sdk "curl -sk -X POST -H 'Content-Type: application/json' \
       -d '{\"index\":$SRC_INDEX,\"file\":\"$VIDEO\",\"transport\":\"rtsp\"}' \
       https://127.0.0.1:9900/api/mediasrc/assign" >/dev/null
  sdk "curl -sk -X POST -H 'Content-Type: application/json' -d '{\"index\":$SRC_INDEX}' \
       https://127.0.0.1:9900/api/mediasrc/start" >/dev/null
  sleep 3
fi
probe=$(sdk "ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
         -show_entries stream=codec_name,width,height,avg_frame_rate \
         -of default=noprint_wrappers=1 -i rtsp://127.0.0.1:8554/src$SRC_INDEX")
if echo "$probe" | grep -q codec_name; then
  ok "RTSP src$SRC_INDEX decodes: $(echo "$probe" | tr '\n' ' ')"
else
  no "RTSP src$SRC_INDEX not decodable (upload media via Insight first)"
fi

hdr "5. Insight metadata ingest (UDP 9100)"
before=$(sdk "curl -sk 'https://127.0.0.1:9900/api/ingest/stats?all=1'" \
         | python3 -c 'import json,sys;print(json.load(sys.stdin)["channels"][0]["metadata"]["messages_received"])' 2>/dev/null || echo 0)
sdk "timeout 5 /opt/neat-insight/venv/bin/neat-insight-metadata-test \
     --count 1 --types object-detection --fps 30 >/dev/null 2>&1; true"
after=$(sdk "curl -sk 'https://127.0.0.1:9900/api/ingest/stats?all=1'" \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["channels"][0]["metadata"]["messages_received"])' 2>/dev/null || echo 0)
if [ "$after" -gt "$before" ]; then
  ok "metadata ingest works ($((after-before)) messages received)"
else
  no "metadata ingest saw no messages"
fi

hdr "6. Modalix DevKit"
if ping -c 1 -W 2000 "$DEVKIT_IP" >/dev/null 2>&1; then
  ok "DevKit reachable at $DEVKIT_IP"
  if sdk_dk 'type dk' | grep -q function; then
    ok "'dk' helper is present in the SDK container"
    sdk "mkdir -p /workspace/hello_kit && \
         echo 'print(\"Hello from your DevKit!\")' > /workspace/hello_kit/hello.py"
    out=$(sdk_dk 'cd /workspace && dk hello_kit/hello.py')
    if echo "$out" | grep -q 'Hello from your DevKit'; then
      ok "dk executed Python on the DevKit"
    else
      no "dk ran but produced unexpected output: $out"
    fi
  else
    sk "'dk' absent -> DevKit not paired. Run on HOST: sima-cli sdk setup --devkit $DEVKIT_IP"
  fi
else
  sk "DevKit not reachable at $DEVKIT_IP (check Ethernet adapter + cable; see docs/DEMO.md)"
fi

hdr "Summary"
printf '  %d passed, %d failed, %d skipped\n\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
