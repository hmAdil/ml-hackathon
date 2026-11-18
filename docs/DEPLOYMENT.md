# Deployment: two-laptop evaluation

Laptop 1 runs the API (pipeline, model, HMM, reference data). Laptop 2 runs the static UI.
Ground-truth routes stay on Laptop 1; the UI only receives GPS points, the predicted route and metrics.

## Setup (Laptop 1)
1. Python 3.10, `pip install -r requirements.txt`.
2. Artifacts in `artifacts/`: `final_model.joblib` (falls back to `logreg.joblib`), optional `hmm_config.json`.
   If `pathsnap.hmm.decode` is missing or fails, the API logs a warning and uses classifier-only decoding.
3. Evaluator trips in a folder (default `data/evaluator`, layout `<id>/<id>.track|.nodes|.arcs|.route`).
   Override with env var `PATHSNAP_EVAL_DIR`.

## Run
Laptop 1: `powershell -File scripts/run_api.ps1 [-EvalDir data/evaluator] [-Port 8000]`
(equivalent: `PYTHONPATH=src uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000`). It prints the LAN IP.
Watch this console: each request logs the trip id, the three pipeline steps with timings, and metrics.

Laptop 2: `powershell -File scripts/run_ui.ps1` (or `python -m http.server 8080 --directory ui`, or just open
`ui/index.html`). Type `http://<Laptop 1 IP>:8000` into the API field and press Connect.
The UI needs internet for Leaflet (cdnjs) and Esri map tiles.

## Firewall (Laptop 1, admin PowerShell)
```
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Domain
```
Both laptops must be on the same network (set the network profile to Private, or drop `-Profile`).

## Verify with curl
```
curl http://<ip>:8000/health
curl http://<ip>:8000/trips
curl -X POST http://<ip>:8000/predict/<trip_id>
```

## Evaluator flow to endpoints
| Step | Endpoint |
|---|---|
| UI opens, lists the ten trips | `GET /trips` (connectivity: `GET /health`) |
| Evaluator picks a trip | `POST /predict/{trip_id}` |
| API: candidates -> model -> HMM, then accuracy vs internal reference | server-side, logged |
| UI draws GPS (blue), predicted route (red), metrics, timings | JSON response |

Response keys: `trip_id, gps, route[{arc_id, polyline}], metrics{top1_accuracy, route_continuity, n_obs, ...},
timings_ms, model, decoder`. Unknown ids return 404. Results are cached in memory per trip (restart to clear).
