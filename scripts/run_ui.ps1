# Serve the static UI (Laptop 2). Usage: .\scripts\run_ui.ps1 [-Port 8080]
param([int]$Port = 8080)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127|169\.254)' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress
Write-Host "UI URL: http://localhost:$Port  (LAN: http://${ip}:$Port)"
Write-Host "In the UI, set the API field to http://<Laptop 1 IP>:8000 and press Connect."
python -m http.server $Port --directory ui --bind 0.0.0.0
