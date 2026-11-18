#!/usr/bin/env bash
# Serve the static UI (Laptop 2). Usage: scripts/run_ui.sh [PORT]
# Linux/macOS equivalent of run_ui.ps1.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
port="${1:-8080}"
py="python3"; [ -x "$root/.venv/bin/python" ] && py="$root/.venv/bin/python"
ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "UI URL: http://localhost:${port}  (LAN: http://${ip}:${port})"
echo "Open: http://localhost:${port}/index.html?api=http://<Laptop 1 IP>:8000"
exec "$py" -m http.server "$port" --directory ui --bind 0.0.0.0
