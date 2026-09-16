#!/usr/bin/env bash
# [DEVKIT] Foreman edge agent: detection on the MLA + the inspection API.
# Run this ON THE DEVKIT, or from the Mac via:
#   ssh sima@<ip> 'bash /workspace/foreman/scripts/run-edge.sh'
set -euo pipefail

: "${FOREMAN_SOURCE:?set FOREMAN_SOURCE, e.g. rtsp://<mac-ip>:8554/src2}"
: "${FOREMAN_MODEL:?set FOREMAN_MODEL to the compiled detector .tar.gz}"

# The workspace reaches the board over NFS (/workspace) or the rsync fallback
# (/workspace-rsync). Pick whichever is actually present.
if [ -d /workspace/foreman ]; then WS=/workspace
elif [ -d /workspace-rsync/foreman ]; then WS=/workspace-rsync
else echo "foreman not found under /workspace or /workspace-rsync on this board" >&2; exit 1; fi

export FOREMAN_LABELS="${FOREMAN_LABELS:-/media/nvme/foreman/coco.txt}"
export FOREMAN_INSIGHT_HOST="${FOREMAN_INSIGHT_HOST:?set FOREMAN_INSIGHT_HOST to this Mac IP address}"
export FOREMAN_GENAI_URL="${FOREMAN_GENAI_URL:-http://127.0.0.1:9998}"
export FOREMAN_EDGE_PORT="${FOREMAN_EDGE_PORT:-8100}"
PYNEAT="${PYNEAT:-$HOME/pyneat/bin/python3}"

[ -x "$PYNEAT" ] || { echo "PyNeat interpreter not found at $PYNEAT" >&2; exit 1; }

echo "[edge] workspace=$WS source=$FOREMAN_SOURCE insight=$FOREMAN_INSIGHT_HOST genai=$FOREMAN_GENAI_URL"
exec "$PYNEAT" "$WS/foreman/edge/foreman_edge.py" "$@"
