# PathSnap: Distributed Map-Matching & Route Reconstruction

PES UE24CS352A ML hackathon. Noisy GPS tracks are matched to a road network by candidate
generation, a candidate scorer (five model families) and HMM/Viterbi decoding, served via
FastAPI (Laptop 1) with a separate UI (Laptop 2). Plan and status: [docs/PLAN.md](docs/PLAN.md).

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Place the dataset in `data/train/` and `data/student_test/` (git-ignored).

## Run
```
set PYTHONPATH=src
python -m pathsnap.eda            # EDA tables + plots -> results/eda/
python -m pathsnap.viz 00000001   # Folium map -> results/viz/
pytest
uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

## Rules
Split by trip only; fit preprocessing on the 80 training trips; evaluator trips and labels are
never used for training, selection or tuning and stay server-side.
