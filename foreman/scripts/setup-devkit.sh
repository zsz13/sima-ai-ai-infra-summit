#!/usr/bin/env bash
# [MAC] One-time DevKit preparation: copy the agent and models, pull GenAI models.
#
#   DEVKIT_IP=192.168.2.2 ./scripts/setup-devkit.sh
#
# Assumes `sima-cli sdk setup --devkit <IP>` has already been run, so /workspace
# is mounted on the board.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

VLM_MODEL="${VLM_MODEL:-Qwen3-VL-2B-Instruct-GPTQ-a16w4}"
ASR_MODEL="${ASR_MODEL:-whisper-small-a16w8}"
CATALOG="${CATALOG:-/media/nvme/llima/models}"

require_devkit
ok "DevKit answers at $DEVKIT_IP"

step "Checking the workspace mount on the DevKit"
dev "test -d /workspace/foreman" \
  && ok "/workspace/foreman is visible on the board" \
  || die "/workspace is not mounted on the DevKit. Run on the Mac: sima-cli sdk setup --devkit $DEVKIT_IP"

step "Speech recognition model"
if dev "test -d $CATALOG/$ASR_MODEL/devkit"; then
  ok "$ASR_MODEL already on the board"
elif dev "test -d /workspace/models/$ASR_MODEL/devkit"; then
  say "Copying the pre-staged $ASR_MODEL from the workspace (no download needed)"
  dev "mkdir -p $CATALOG && cp -r /workspace/models/$ASR_MODEL $CATALOG/"
  ok "$ASR_MODEL copied to $CATALOG"
else
  warn "$ASR_MODEL not found locally; pulling it on the board (needs internet)"
  dev "llima pull $ASR_MODEL" || die "llima pull failed. See docs/DEMO.md for DevKit internet setup."
fi

step "Vision-language model"
if dev "test -d $CATALOG/$VLM_MODEL/devkit"; then
  ok "$VLM_MODEL already on the board"
else
  warn "Pulling $VLM_MODEL (several GB, needs internet on the DevKit)"
  say "Model names drift between docs and the live repos. If this fails, list what exists:"
  say "  ssh $DEVKIT_USER@$DEVKIT_IP 'llima search vl'"
  dev "llima pull $VLM_MODEL" || die "llima pull failed for $VLM_MODEL"
fi

step "Installed models"
dev "ls -1 $CATALOG" || true
ok "DevKit ready. Next: scripts/run-genai.sh then scripts/run-edge.sh (both on the DevKit)."
