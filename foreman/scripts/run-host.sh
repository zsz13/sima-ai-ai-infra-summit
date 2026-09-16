#!/usr/bin/env bash
# [MAC] Foreman orchestrator + product UI.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

EDGE_PORT="${FOREMAN_EDGE_PORT:-8100}"
export FOREMAN_EDGE_URL="${FOREMAN_EDGE_URL:-http://$DEVKIT_IP:$EDGE_PORT}"
export FOREMAN_AUDIT_DIR="${FOREMAN_AUDIT_DIR:-$PWD/audit}"
export FOREMAN_INSIGHT_URL="${FOREMAN_INSIGHT_URL:-https://127.0.0.1:8081/static/viewer.html?src=0}"
export FOREMAN_PORT="${FOREMAN_PORT:-8800}"

say "edge:    $FOREMAN_EDGE_URL"
say "audit:   $FOREMAN_AUDIT_DIR"
say "open:    http://127.0.0.1:$FOREMAN_PORT"
exec uv run python -m host.app
