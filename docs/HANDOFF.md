# Handoff: two-laptop deployment

Everything below is implemented on `main`. What is left is running it on two real laptops.

## What exists
- **API** (`src/pathsnap/api.py`): `GET /health`, `GET /trips`, `POST /predict/{trip_id}`. Binds
  to `0.0.0.0`, CORS open. Runs the full pipeline (candidates -> final model -> HMM) on every
  request, computes metrics server-side against the `.route` reference, and never returns
  reference data. Per-trip lock, so several laptops can request at once. Caching is off unless
  `PATHSNAP_CACHE=1`.
- **UI** (`ui/`): static HTML/JS. Lists the trips, runs one or all, draws GPS (blue) and the
  predicted route (red), shows metrics and a session table. API address can be set in the box or
  by link: `index.html?api=http://<laptop1-ip>:8000` (add `&run=all` or `&run=<trip_id>` to
  start a run on load).
- **Model**: `artifacts/final_model.joblib` (LightGBM LambdaRank, Optuna-tuned, see
  `artifacts/final_model.json`). **Decoder**: `artifacts/hmm_config.json`.
- **Scripts**: `scripts/run_api.ps1`, `scripts/run_ui.ps1`; full steps in `docs/DEPLOYMENT.md`.
- Evaluator trips are read from `PATHSNAP_EVAL_DIR` (default `data/evaluator`, git-ignored, same
  layout as `data/student_test`). Do not commit them and do not use them for any decision.

## Verified so far
- Student-test run (`PATHSNAP_EVAL_DIR=data/student_test`): all 10 trips return successfully via
  the API from the UI link with `?run=all`; no errors in the API log.
- `pytest` (24 tests) and `ruff check .` pass.

## Verified on one Linux laptop (2026-11-18, `deploy/handoff-checks`)
- `scripts/run_api.sh data/student_test` + `scripts/run_ui.sh 8090` (bash equivalents of the
  `.ps1` scripts). API and UI reached through the LAN IP, not localhost: `/health`, `/trips` and
  all 10 `POST /predict/{trip_id}` returned 200 from the UI link `?api=http://<ip>:8000&run=all`.
- Headless Chromium visual check: map tiles, legend, metric cards and session table render; the
  only console error is a missing `favicon.ico`. Evidence: `docs/screenshots/`.
  Note: this is one laptop playing both roles; the two-laptop run is still needed.
- `scikit-learn` pinned to 1.7.2, the version that pickled `final_model.joblib` (1.9.1 gave an
  InconsistentVersionWarning at API start; scores were identical, but pin to stay safe). The
  LightGBM version is not recorded in the artifact: run `pip freeze` on the training machine and
  pin `lightgbm` too.
- If port 8080 is taken on the UI laptop, pass another port (`run_ui.sh 8090` / `-Port 8090`).
- CI: `.github/workflows/ci.yml` adds a job named `test` (ruff + pytest, CPU torch);
  `tests/test_api.py` skips when `data/student_test` is absent.

## Not verified (this is the work)
1. **Two real laptops**: API on Laptop 1 reachable from Laptop 2 over the LAN (Windows firewall
   inbound rule for TCP 8000, both on the same network, use Laptop 1's LAN IP).
2. **UI visual check** in a normal browser on Laptop 2 (layout, map tiles, legend). It was
   exercised headlessly only; no screenshot was captured.
3. **PowerShell scripts** (`run_api.ps1`, `run_ui.ps1`) were never executed (no Windows here;
   the bash versions were).
4. **Required screenshots** for the report: the working UI, and the API console on the receiving
   laptop showing successful requests.
5. Run the API with the real evaluator trips (`data/evaluator`) once everything is frozen, and
   confirm all ten trips work. Do not change models or parameters based on those results.
6. **CI**: done on `deploy/handoff-checks` (job `test`); confirm the check goes green on the PR.

## Still changing (do not rely on)
The HMM decoder parameters are being re-tuned on the final model's scores. When that finishes,
`artifacts/hmm_config.json` is updated; pull before the final test. Model weights do not change.

## Quick start
```
pip install -r requirements.txt
set PYTHONPATH=src
set PATHSNAP_EVAL_DIR=data/student_test      # use data/evaluator only for the final check
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
python -m http.server 8080 --directory ui    # on Laptop 2, then open index.html?api=http://<ip>:8000
```
Note: `data/` is not in git. Copy the dataset folders (and `data/evaluator`) onto the laptop.
