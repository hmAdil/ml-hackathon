# Start the PathSnap API (Laptop 1). Usage: .\scripts\run_api.ps1 [-EvalDir data\evaluator] [-Port 8000]
param([string]$EvalDir = "data/evaluator", [int]$Port = 8000)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PATHSNAP_EVAL_DIR = $EvalDir
$env:PYTHONPATH = Join-Path $root "src"
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127|169\.254)' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress
Write-Host "PATHSNAP_EVAL_DIR = $EvalDir"
Write-Host "API URL for Laptop 2 UI:  http://${ip}:$Port"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port $Port
