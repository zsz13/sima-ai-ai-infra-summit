#!/usr/bin/env bash
# [DEVKIT] Serve the vision-language model and Whisper on the MLA.
# Run this ON THE DEVKIT (ssh sima@<ip>), or from the Mac via:
#   ssh sima@<ip> 'bash /workspace/foreman/scripts/run-genai.sh'
#
# Exposes an OpenAI-compatible API on :9998 that the edge agent calls.
set -euo pipefail

CATALOG="${CATALOG:-/media/nvme/llima/models}"
VLM_MODEL="${VLM_MODEL:-Qwen3-VL-2B-Instruct-GPTQ-a16w4}"
ASR_MODEL="${ASR_MODEL:-whisper-small-a16w8}"
PORT="${GENAI_PORT:-9998}"
PYNEAT="${PYNEAT:-$HOME/pyneat/bin/python3}"

PIDFILE="${PIDFILE:-/tmp/foreman-genai.pid}"

# Refuse to start a second server. The process ends up as `python3 -` (heredoc),
# so a pattern like `pkill -f run-genai.sh` never matches it; three servers once
# accumulated unnoticed and took CmaFree from 1.6 GB to 15 MB.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[genai] already running as PID $(cat "$PIDFILE"); not starting another." >&2
  exit 0
fi

[ -x "$PYNEAT" ] || { echo "PyNeat interpreter not found at $PYNEAT" >&2; exit 1; }
[ -d "$CATALOG/$VLM_MODEL" ] || { echo "missing VLM: $CATALOG/$VLM_MODEL" >&2; exit 1; }
[ -d "$CATALOG/$ASR_MODEL" ] || { echo "missing ASR: $CATALOG/$ASR_MODEL" >&2; exit 1; }

# The MLA bulk loader allocates from CMA, but CMA pages held by the page cache
# are not reclaimed on demand: loading a 4B VLM fails with MLA_LOAD_FAILED when
# CmaFree is low even though nothing else is running. Measured on this board:
# CmaFree 407 MB -> 1630 MB after a cache drop. Reclaim before loading.
echo "[genai] CmaFree before: $(grep CmaFree /proc/meminfo | awk '"'"'{print $2" "$3}'"'"')"
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null ||   echo "[genai] warning: could not drop caches; a large model may fail to load"
echo "[genai] CmaFree after:  $(grep CmaFree /proc/meminfo | awk '"'"'{print $2" "$3}'"'"')"

echo "[genai] serving vlm=$VLM_MODEL asr=$ASR_MODEL on :$PORT"

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
exec "$PYNEAT" - "$CATALOG/$VLM_MODEL" "$CATALOG/$ASR_MODEL" "$PORT" <<'PY'
import signal, sys
import pyneat

# pyneat exposes genai as an attribute of the package, not an importable
# submodule: `import pyneat.genai` raises ModuleNotFoundError on pyneat 0.4.0.
genai = pyneat.genai

vlm_dir, asr_dir, port = sys.argv[1], sys.argv[2], int(sys.argv[3])

options = genai.GenAIServerOptions()
options.host = "0.0.0.0"
options.port = port

server = genai.GenAIServer(options)
server.add_model(vlm_dir, "vlm")
server.add_model(asr_dir, "asr")

def shutdown(signum, _frame):
    print(f"\n[genai] signal {signum}, stopping", flush=True)
    try:
        server.stop()
    finally:
        sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, shutdown)

# pyneat 0.4.0 exposes start()/stop(), not a blocking serve().
server.start()
print(f"[genai] models: {server.model_names()} listening on :{port}", flush=True)
signal.pause()
PY
