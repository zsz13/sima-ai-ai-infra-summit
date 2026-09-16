#!/usr/bin/env bash
# DEVELOPMENT ONLY - starts the host UI against the FAKE EDGE test harness.
# Every verdict it shows is synthetic. Never run this during a demo.
# For the real system use scripts/demo.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
DEPS=(--with fastapi --with uvicorn --with httpx --with python-multipart --with pillow)
uv run --quiet "${DEPS[@]}" python -m uvicorn tests.fake_edge:app --host 127.0.0.1 --port 8100 --log-level error &
FAKE=$!
trap 'kill $FAKE 2>/dev/null || true' EXIT
sleep 2
FOREMAN_EDGE_URL=http://127.0.0.1:8100 \
FOREMAN_AUDIT_DIR=/tmp/foreman-dev-audit \
FOREMAN_STABLE_FRAMES=6 \
uv run --quiet "${DEPS[@]}" python -m host.app
