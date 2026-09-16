# Shared helpers. Sourced by the other scripts.
SDK_CONTAINER="${SDK_CONTAINER:-ghcr.io-sima-neat-sdk-v2.1.3.0}"
DEVKIT_IP="${DEVKIT_IP:-192.168.2.2}"
DEVKIT_USER="${DEVKIT_USER:-sima}"
INSIGHT_PORT=9900
RTSP_PORT=8554

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s  ok %s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s warn %s %s\n' "$c_yel" "$c_off" "$*"; }
die()  { printf '%sfail %s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }
step() { printf '\n%s== %s ==%s\n' "$c_dim" "$*" "$c_off"; }

sdk() { docker exec "$SDK_CONTAINER" bash -lc "$1"; }
dev() { ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$DEVKIT_USER@$DEVKIT_IP" "$1"; }

require_container() {
  docker ps --format '{{.Names}}' | grep -qx "$SDK_CONTAINER" \
    || die "SDK container '$SDK_CONTAINER' is not running. Start it with: sima-cli sdk start"
}

require_devkit() {
  ping -c 1 -W 2000 "$DEVKIT_IP" >/dev/null 2>&1 \
    || die "DevKit unreachable at $DEVKIT_IP. See docs/DEMO.md 'If the DevKit is unreachable'."
}

# The Mac's address on the DevKit's subnet - what the board must dial back to.
# `ipconfig getifaddr` returns nothing for a manually configured interface such as
# the Internet Sharing bridge, so read the address off ifconfig instead.
host_ip_for_devkit() {
  local iface
  iface=$(route -n get "$DEVKIT_IP" 2>/dev/null | awk '/interface:/{print $2; exit}')
  [ -n "$iface" ] || return 1
  ifconfig "$iface" 2>/dev/null | awk '/inet /{print $2; exit}'
}
