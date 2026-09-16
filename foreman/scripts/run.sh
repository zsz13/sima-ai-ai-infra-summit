#!/usr/bin/env bash
# [MAC] Start Foreman against a chosen inference backend.
#
#   ./scripts/run.sh                      # modalix (default - the demo path)
#   ./scripts/run.sh --backend local      # local models on this Mac, no DevKit
#   ./scripts/run.sh --backend fake       # synthetic harness, no models at all
#
# The backend is the ONLY thing that changes. The host, the console and the Edge
# API contract are identical in all three, which is the point: a bug reproduced
# against `fake` or `local` is a bug in the real path too.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${FOREMAN_BACKEND:-modalix}"
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend needs a value}"; shift 2 ;;
    --backend=*) BACKEND="${1#*=}"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$BACKEND" in
  modalix)
    # The real path. Unchanged, and still the default.
    exec ./scripts/demo.sh
    ;;
  fake)
    exec ./scripts/dev-ui.sh
    ;;
  local)
    exec ./scripts/dev-local.sh
    ;;
  *)
    echo "unknown backend: $BACKEND (expected modalix, local or fake)" >&2
    exit 2
    ;;
esac
