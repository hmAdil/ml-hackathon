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

## Remaining checklist
1. ~~Two real laptops~~ **Done** (2026-11-18): API on a Windows laptop (Laptop 1, set up with
   `docs/LAPTOP1_WINDOWS_SETUP.pdf`, firewall rule for TCP 8000), UI on a Linux laptop
   (Laptop 2, `scripts/laptop2_connect.sh`), same Wi-Fi, Laptop 1 reached by its LAN IP.
   `/health` and `/trips` answered from Laptop 2, and "Run all trips" returned all ten evaluator
   trips: mean Top-1 95.2%, route continuity 82.8%, 1.35 s per trip
   (`docs/screenshots/two_laptop_run.csv`). The numbers match `results/evaluator_run.csv` to the
   UI's 0.1% rounding, so the deployed system reproduces the frozen evaluator run on another
   machine.
2. ~~UI visual check~~ **Done**: map tiles, legend, metric cards and session table render on
   Laptop 2 (`docs/screenshots/two_laptop_ui.png`); only console error is a missing favicon.
3. **PowerShell scripts**: `run_api.ps1` started the API on Laptop 1 following the setup guide.
   The `run_ui.ps1` start check (guide step 5) is still to be confirmed by Laptop 1's operator.
4. **Required screenshots**: UI on Laptop 2 **done** (`docs/screenshots/two_laptop_ui.png`).
   Still needed: the API console on Laptop 1 showing the ten `POST /predict/... 200 OK` lines
   from Laptop 2 (save as `docs/screenshots/two_laptop_api_console.png`).
5. ~~Evaluator run~~ **Done** (`f083b9c`, `results/evaluator_run.csv`): all ten evaluator trips
   returned successfully through the API with the frozen model and decoder. Do not re-run it to
   change anything; for the live demo, copy `data/evaluator` onto Laptop 1.
6. ~~CI~~ **Done**: job `test` is green on `main`; PR #12 is closed (its content is merged).

## Frozen
Model (`artifacts/final_model.joblib`) and decoder (`artifacts/hmm_config.json`, re-tuned on the
final model's scores in `48f3322`) were frozen before the evaluator run. Do not change them.
`lightgbm==4.7.0`, `xgboost==3.2.0` and `scikit-learn==1.7.2` are pinned in `requirements.txt`.

## Quick start
```
pip install -r requirements.txt
set PYTHONPATH=src
set PATHSNAP_EVAL_DIR=data/student_test      # use data/evaluator only for the final check
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
python -m http.server 8080 --directory ui    # on Laptop 2, then open index.html?api=http://<ip>:8000
```
Note: `data/` is not in git. Copy the dataset folders (and `data/evaluator`) onto the laptop.
