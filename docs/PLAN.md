# PathSnap research plan

Sources: `PathSnap_ML_Research_Proposal.md` (structure) and the phase-by-phase researcher
brief (process). Status column is updated as work lands; nothing here is a result.

Pipeline: candidate generation -> candidate scoring (binary classification / ranking)
-> HMM/Viterbi decoding -> API + UI.

| Phase | Work | Code | Status |
|---|---|---|---|
| 1 | Problem understanding, ideation | this file | done |
| 2 | Dataset exploration, Folium map | `pathsnap/eda.py`, `pathsnap/viz.py` | implemented, run pending |
| 3 | Preprocessing, label alignment | `pathsnap/labels.py` | implemented |
| 4 | Candidates + feature engineering | `pathsnap/candidates.py`, `features.py`, `dataset.py` | implemented |
| 5 | Augmentation (only if justified) | - | not started |
| 6 | Five models | `pathsnap/models/` | 1 of 5 (logreg) |
| 7 | Grouped CV + Optuna | `pathsnap/cv.py`, `train.py`, `tuning.py` | CV done, Optuna stub |
| 8 | Evaluation + comparison table | `pathsnap/metrics.py` | partial |
| 9 | Ensembling, HMM/Viterbi | `pathsnap/hmm.py` | stub |
| - | API + UI | `pathsnap/api.py`, `ui/` | stub |

## Key design decisions
- Arc IDs are trip-local: model candidate rows `(trip, obs, arc)` as binary targets, never
  a global multiclass over arc IDs.
- `.route` has no timestamps (e.g. 525 arcs vs 1315 GPS rows in trip 1), so per-point labels
  come from a monotone alignment of GPS points to route positions, validated on distance
  and visually.
- Split by trip only (GroupKFold on trip_id). Student-test trips are for development
  comparison; evaluator trips are never touched.
- Primary selection metric: observation-level Top-1 on grouped CV.

## Five models (distinct families)
Logistic Regression (linear), Random Forest (bagging), XGBoost GPU (boosting),
PyTorch MLP GPU (neural), LightGBM LambdaRank (ranking).
Stacking and HMM are kept only if they improve held-out metrics.
