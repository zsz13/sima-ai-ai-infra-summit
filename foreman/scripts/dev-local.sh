#!/usr/bin/env bash
# [MAC] DEVELOPMENT ONLY - Foreman running entirely on this Mac.
#
#   ./scripts/dev-local.sh
#
# No DevKit, no SDK container, no RTSP. Inference runs natively rather than in
# Docker on purpose: Docker Desktop on macOS has no Metal passthrough, so a
# containerised model would be CPU-only. Host and console could be
# containerised; the models should not be.
#
# Everything produced here is labelled backend=local, in the console header and
# in every audit record, and is written to audit-local/ so it never mixes with
# the Modalix trail. Never quote a latency measured here as a Modalix result.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${FOREMAN_EDGE_PORT:-8100}"
UI_PORT="${FOREMAN_PORT:-8800}"
export FOREMAN_AUDIT_DIR="${FOREMAN_AUDIT_DIR:-$PWD/audit-local}"
export FOREMAN_LOCAL_DETECTOR="${FOREMAN_LOCAL_DETECTOR:-$PWD/models-local/yolo26n.onnx}"
export FOREMAN_LOCAL_VLM="${FOREMAN_LOCAL_VLM:-mlx-community/Qwen3-VL-2B-Instruct-4bit}"
export FOREMAN_LOCAL_ASR="${FOREMAN_LOCAL_ASR:-small}"

STEPS=5
step() { printf '\033[2m[%s/%s]\033[0m %s\n' "$1" "$STEPS" "$2"; }
die()  { printf '\033[31mfail\033[0m %s\n' "$*" >&2; exit 1; }

# --- ports ------------------------------------------------------------------
# Fail loudly rather than racing something already bound: two edges on one port
# is how a stale fake backend silently answered for the real one once before.
for p in "$PORT" "$UI_PORT"; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    printf '\033[31mfail\033[0m port %s is already in use by:\n' "$p" >&2
    lsof -nP -iTCP:"$p" -sTCP:LISTEN | tail -n +2 | sed 's/^/     /' >&2
    echo "     stop it first, or set FOREMAN_EDGE_PORT / FOREMAN_PORT" >&2
    exit 1
  fi
done

# --- dependencies -----------------------------------------------------------
step 1 "Checking local inference dependencies"
if [ ! -x .venv/bin/python ] || \
   ! .venv/bin/python -c 'import onnxruntime, faster_whisper, mlx_vlm' 2>/dev/null; then
  echo "   installing the optional 'local' dependency group (one-off)"
  uv sync --group local --quiet || die "uv sync --group local failed"
fi
.venv/bin/python -c 'import onnxruntime, faster_whisper, mlx_vlm' 2>/dev/null \
  || die "local dependencies still missing. Run: uv sync --group local"
echo "   onnxruntime, faster-whisper, mlx-vlm present"

# --- models -----------------------------------------------------------------
step 2 "Checking local models"
[ -f "$FOREMAN_LOCAL_DETECTOR" ] || die "detector weights missing: $FOREMAN_LOCAL_DETECTOR
     Download once (about 10 MB):
       curl -sL -o models-local/yolo26n.onnx \\
         https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx
     See docs/LOCAL_BACKEND.md"
echo "   detector $(basename "$FOREMAN_LOCAL_DETECTOR")"
echo "   whisper  $FOREMAN_LOCAL_ASR    (downloads once, ~486 MB)"
echo "   vlm      $FOREMAN_LOCAL_VLM    (downloads once, ~1.7 GB)"

# --- processes --------------------------------------------------------------
# Only the PIDs started here are ever signalled. No pkill by name, nothing on the
# SiMa SDK, the DevKit or Docker is touched.
EDGE=""; HOST=""
cleanup() {
  trap - EXIT INT TERM
  printf '\n'
  printf '\033[2m--\033[0m Stopping\n'
  for pid in "$HOST" "$EDGE"; do
    [ -n "$pid" ] || continue
    # Signal the whole group: the edge owns an ffmpeg child and the host is a
    # `uv run` wrapper around the real server. Killing only the parent leaves an
    # orphan holding the camera or the port, which breaks the next start.
    kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 24); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -KILL -- -"$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "   stopped"
}
trap cleanup EXIT INT TERM

step 3 "Starting local edge (detector + Whisper + VLM + camera)"
set -m                       # own process group per job, so children go too
# </dev/null matters: with job control on, a background child that reads the
# terminal takes SIGTTIN and stops its whole process group. ffmpeg does exactly
# that unless told not to, which left the port bound but nothing answering.
.venv/bin/python edge/local_edge.py --port "$PORT" </dev/null &
EDGE=$!
set +m

EDGE_OK=0
for _ in $(seq 1 90); do
  # A bound socket is not readiness: the failure this guards against left the
  # port listening while the process was stopped. Require real JSON back.
  if curl -sS --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"backend"'; then
    EDGE_OK=1; break
  fi
  kill -0 "$EDGE" 2>/dev/null || die "the local edge exited during startup; output above."
  case "$(ps -o stat= -p "$EDGE" 2>/dev/null)" in
    T*) die "the local edge was STOPPED by the terminal (SIGTTIN).
     A child is reading stdin. Start it with </dev/null, and ensure ffmpeg has -nostdin." ;;
  esac
  sleep 1
done
[ "$EDGE_OK" = 1 ] || die "the local edge did not answer /health within 90s on :$PORT"
echo "   edge healthy on :$PORT"

step 4 "Starting Foreman host and console"
# Its own process group as well: `uv run` spawns python as a child, so killing
# only the wrapper leaves the real server orphaned holding the port.
set -m
FOREMAN_EDGE_URL="http://127.0.0.1:$PORT" \
uv run --quiet --with fastapi --with uvicorn --with httpx --with python-multipart \
  python -m host.app </dev/null >/tmp/foreman-local-host.log 2>&1 &
HOST=$!
set +m
HOST_OK=0
for _ in $(seq 1 60); do
  if curl -sS --max-time 3 "http://127.0.0.1:$UI_PORT/api/state" 2>/dev/null | grep -q '"backend"'; then
    HOST_OK=1; break
  fi
  if ! kill -0 "$HOST" 2>/dev/null; then
    echo "--- last lines of /tmp/foreman-local-host.log ---" >&2
    tail -20 /tmp/foreman-local-host.log >&2 || true
    die "the host exited during startup"
  fi
  sleep 1
done
if [ "$HOST_OK" != 1 ]; then
  echo "--- last lines of /tmp/foreman-local-host.log ---" >&2
  tail -20 /tmp/foreman-local-host.log >&2 || true
  die "the host did not answer /api/state within 60s on :$UI_PORT"
fi

step 5 "Verifying both endpoints"
curl -sS --max-time 3 -o /dev/null -w '' "http://127.0.0.1:$UI_PORT/" \
  || die "the console did not serve its index page on :$UI_PORT"
echo "   edge :$PORT and console :$UI_PORT both answering"

BACKEND=$(curl -s --max-time 3 "http://127.0.0.1:$UI_PORT/api/state" \
          | sed -n 's/.*"backend":"\([a-z]*\)".*/\1/p')
[ "$BACKEND" = "local" ] || die "the host reports backend='$BACKEND', expected 'local'.
     Refusing to continue rather than let a different backend look like local."

printf '\n  \033[32mForeman is running on the LOCAL backend.\033[0m  No SiMa hardware in use.\n\n'
echo "    Console    http://127.0.0.1:$UI_PORT"
echo "    Edge API   http://127.0.0.1:$PORT/health"
echo "    Audit      $FOREMAN_AUDIT_DIR   (separate from the Modalix trail)"
echo
echo "  The header shows 'Local inference' in amber, not 'Modalix connected'."
echo "  Every verdict is recorded with backend=local. These are not Modalix results."
echo
echo "  Ctrl-C stops the two processes this script started, and nothing else."
echo
wait
