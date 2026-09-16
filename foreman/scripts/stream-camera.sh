#!/usr/bin/env bash
# [MAC] Publish the Mac camera into Insight's RTSP server so the DevKit can see it.
#
#   ./scripts/stream-camera.sh            # built-in camera -> src2
#   ./scripts/stream-camera.sh 1 src3     # Desk View camera (points down at the
#                                         # bench - the better inspection angle)
#
# List cameras:  ffmpeg -f avfoundation -list_devices true -i ""
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

DEVICE="${1:-0}"
SLOT="${2:-src2}"
SIZE="${CAM_SIZE:-1280x720}"
# avfoundation only accepts a rate the device advertises exactly (15 or 30 here)
# and only delivers uyvy422; anything else fails with a misleading error.
FPS="${CAM_FPS:-30}"
PIXFMT="${CAM_PIXFMT:-uyvy422}"
OUT_FPS="${CAM_OUT_FPS:-15}"

command -v ffmpeg >/dev/null || die "ffmpeg is not installed. Run: brew install ffmpeg"
require_container

say "camera $DEVICE -> rtsp://127.0.0.1:$RTSP_PORT/$SLOT  (${SIZE}, capture ${FPS}fps -> stream ${OUT_FPS}fps)"
say "The DevKit should consume: rtsp://$(host_ip_for_devkit):$RTSP_PORT/$SLOT"
say "Ctrl-C to stop."

# zerolatency + a short GOP keeps the DevKit's decoder in sync and lets a late
# subscriber get a keyframe quickly.
exec ffmpeg -hide_banner -loglevel warning \
  -f avfoundation -framerate "$FPS" -video_size "$SIZE" -pixel_format "$PIXFMT" -i "$DEVICE" \
  -vf "fps=$OUT_FPS" \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -g "$OUT_FPS" -bf 0 \
  -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:$RTSP_PORT/$SLOT"
