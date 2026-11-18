#!/usr/bin/env bash
# Start the PathSnap API (Laptop 1). Usage: scripts/run_api.sh [EVAL_DIR] [PORT]
# Linux/macOS equivalent of run_api.ps1. EVAL_DIR defaults to data/evaluator.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATHSNAP_EVAL_DIR="${1:-data/evaluator}"
port="${2:-8000}"
export PYTHONPATH="$root/src"
py="python3"; [ -x "$root/.venv/bin/python" ] && py="$root/.venv/bin/python"
ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -z "$ip" ] && ip="$(ipconfig getifaddr en0 2>/dev/null || echo '<this-laptop-ip>')"
echo "PATHSNAP_EVAL_DIR = $PATHSNAP_EVAL_DIR"
echo "API URL for Laptop 2 UI:  http://${ip}:${port}"
echo "If Laptop 2 cannot connect, allow inbound TCP ${port} (e.g. sudo ufw allow ${port}/tcp)."
exec "$py" -m uvicorn pathsnap.api:app --host 0.0.0.0 --port "$port"
