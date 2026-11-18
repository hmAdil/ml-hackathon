#!/usr/bin/env bash
# Laptop 2 (UI) for the two-laptop run.
# Usage: scripts/laptop2_connect.sh <laptop1-ip>[:port] [ui-port]
#   1. checks the API on Laptop 1 (/health, /trips)
#   2. serves ui/ on this laptop (default port 8090)
#   3. opens Firefox on the UI, already pointed at Laptop 1's API
# Press Ctrl+C here to stop the UI server when finished.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <laptop1-ip>[:port] [ui-port]"; exit 2; }
target="$1"
[[ "$target" == *:* ]] || target="$target:8000"
api="http://$target"
ui_port="${2:-8090}"
root="$(cd "$(dirname "$0")/.." && pwd)"
py="python3"; [ -x "$root/.venv/bin/python" ] && py="$root/.venv/bin/python"

echo "This laptop (Laptop 2): $(hostname -I | awk '{print $1}')"
echo "Checking API at $api ..."
if ! health="$(curl -s -m 5 "$api/health")"; then
  cat <<EOF
Cannot reach $api.
  - Is run_api.ps1 still running on Laptop 1, and is the IP right?
  - Windows firewall rule for the port (setup guide, step 4)?
  - Same network? Some Wi-Fi blocks laptop-to-laptop traffic: use a phone hotspot.
EOF
  exit 1
fi
echo "  health: $health"
echo "  trips:  $(curl -s -m 5 "$api/trips")"

if ss -ltn | grep -q ":$ui_port "; then
  echo "Port $ui_port is busy; pass another, e.g. $0 $1 8091"; exit 1
fi
"$py" -m http.server "$ui_port" --directory "$root/ui" --bind 0.0.0.0 \
  > /tmp/pathsnap_ui_$ui_port.log 2>&1 &
ui_pid=$!
trap 'kill $ui_pid 2>/dev/null; echo; echo "UI server stopped."' EXIT
sleep 1
url="http://localhost:$ui_port/index.html?api=$api"
echo
echo "UI: $url"
echo "In the browser: check the status shows the model and 10 trips, then click 'Run all trips'."
echo "Screenshot when all 10 are done: save as docs/screenshots/two_laptop_ui.png"
echo "Laptop 1 screenshot of the API window: docs/screenshots/two_laptop_api_console.png"
echo "Press Ctrl+C here when finished."
(firefox "$url" >/dev/null 2>&1 &) || true
wait $ui_pid
