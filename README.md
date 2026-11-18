# PathSnap: Distributed Map-Matching & Route Reconstruction

Sem-5 ML hackathon. Noisy GPS points are matched to a road network using a
candidate classifier followed by HMM/Viterbi decoding, served via FastAPI
(Laptop 1) with a separate UI (Laptop 2).

## Layout
| Path | Purpose |
|---|---|
| `src/data` | raw file loading, cleaning, EDA |
| `src/features` | candidate generation, feature engineering |
| `src/models` | model training and evaluation (5+ models) |
| `src/tuning` | Bayesian hyperparameter optimisation |
| `src/hmm` | HMM / Viterbi route reconstruction |
| `src/api` | FastAPI backend |
| `src/viz` | Folium maps and plots |
| `src/common` | shared schema, paths, metrics |
| `ui/` | evaluator interface |
| `artifacts/` | saved model and inference assets |
| `results/` | metrics CSVs, logs, plots |
| `notebooks/` | exploration |
| `data/` | raw dataset (not tracked): `data/train`, `data/student_test` |

## Team workflow
Read [CONTRIBUTING.md](CONTRIBUTING.md) before your first commit. Ownership is in [OWNERS.md](OWNERS.md). `main` is PR-only; branches are named `<area>/<description>`.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```
Place the dataset trip folders in `data/train` and `data/student_test`.

## Run
API (Laptop 1): `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
UI (Laptop 2): see `ui/README.md`.

## Status
Skeleton only; components are being implemented.
