# PathSnap: Distributed Map-Matching & Route Reconstruction

PES UE24CS352A ML hackathon. Noisy GPS tracks are matched to a road network and the driven route
is reconstructed:

```
GPS track + road network
  -> candidate arcs per GPS point (KD-tree, 50 m radius)          candidates.py
  -> 22 features per (point, arc)                                  features.py
  -> candidate scorer: P(arc is the true road)  [5 model families] models/
  -> HMM / Viterbi decoding on the directed road network           hmm.py
  -> predicted route  ->  FastAPI (Laptop 1)  ->  UI (Laptop 2)    api.py, ui/
```

Arc IDs are trip-local, so every (GPS point, candidate arc) pair is a binary example. Labels come
from a monotone alignment of GPS points to the reference route (`labels.py`), because `.route` has
no timestamps and about 5x fewer arcs than GPS points.

## Layout
| Path | Purpose |
|---|---|
| `src/pathsnap/` | package: `io, geo, eda*, labels, candidates, features, dataset, cv, train, compare, tuning, hmm, hmm_eval, pipeline, api, viz*` |
| `src/pathsnap/models/` | `logreg, random_forest, xgboost, mlp, lgbm_rank` (one file per model) |
| `notebooks/01_eda.ipynb` | executed EDA notebook (HTML export in `results/eda/01_eda.html`) |
| `ui/` | evaluator UI (static HTML/JS, talks to the API) |
| `scripts/` | `run_api.ps1`, `run_ui.ps1` |
| `artifacts/` | saved models and `hmm_config.json` |
| `results/` | `eda/`, `models/` (per-model JSON, comparison, plots), `tuning/`, `hmm/`, `viz/` (Folium maps) |
| `docs/` | `PLAN.md` (research plan), `DEPLOYMENT.md` (two-laptop run) |

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Put the dataset in `data/train/<trip_id>/` and `data/student_test/<trip_id>/` (git-ignored). A CUDA
build of PyTorch/XGBoost is used for the GPU models (`xgboost`, `mlp`) when available.

## Reproduce the experiments
```
set PYTHONPATH=src
python -m pathsnap.eda --deep                  # EDA tables and plots -> results/eda/
python -m pathsnap.dataset                     # alignment labels + candidate tables -> data/processed/
python -m pathsnap.train logreg                # also: random_forest xgboost mlp lgbm_rank
python -m pathsnap.compare                     # consolidated comparison -> results/models/
python -m pathsnap.tuning lgbm_rank 20         # Optuna (TPE) on grouped CV -> results/tuning/
python -m pathsnap.train lgbm_rank --params results/tuning/lgbm_rank_best.json
python -m pathsnap.final_scores lgbm_rank_tuned
set PATHSNAP_SCORES=lgbm_rank_tuned
python -m pathsnap.hmm_eval --out-dir final --freeze   # tune and freeze the decoder
python -m pathsnap.viz 00000001                # Folium map for a trip
pytest
```
Every run is seeded (`RANDOM_SEED = 42`). Splits are by trip (`GroupKFold` on `trip_id`).

## Run the system (two laptops)
Laptop 1 (API): `scripts\run_api.ps1`   (or `uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000`)
Laptop 2 (UI): `scripts\run_ui.ps1`, then enter the API address in the UI.
Evaluator trips live in `PATHSNAP_EVAL_DIR` (default `data/evaluator`, same layout as
`data/student_test`). Step-by-step instructions: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Rules followed
Split by trip only; preprocessing and models are fit on the 80 training trips; student-test trips
are for development comparison and decoder tuning; evaluator trips and labels are never used for
training, selection or tuning, and stay server-side (the API never returns reference data).
