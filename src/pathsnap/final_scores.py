"""Honest candidate scores for decoder tuning.

Usage: PYTHONPATH=src python -m pathsnap.final_scores lgbm_rank_tuned

For the 80 train trips scores are OUT-OF-FOLD (same 5 trip-grouped folds as cv.cross_validate),
so the decoder is never tuned on scores from a model that saw the same trip. For student_test
the scores come from the saved model fit on all train trips. Output:
data/processed/scores_<tag>.parquet with trip_id, obs_idx, candidate_arc_id, p.
"""
import importlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import ARTIFACTS, RESULTS
from .cv import fit_scorer, score
from .dataset import PROCESSED


def run(tag: str):
    name = tag.removesuffix("_tuned")
    mod = importlib.import_module(f"pathsnap.models.{name}")
    params = json.loads((RESULTS / "models" / f"{tag}.json").read_text())["params"]
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "student_test.parquet")
    p_tr = np.zeros(len(train))
    for k, (tri, vai) in enumerate(GroupKFold(n_splits=5).split(train, groups=train.trip_id)):
        model, _ = fit_scorer(mod.build, params, train.iloc[tri])
        p_tr[vai] = score(model, train.iloc[vai])
        print(f"fold {k} scored", flush=True)
    p_te = score(joblib.load(ARTIFACTS / f"{tag}.joblib"), test)
    cols = ["trip_id", "obs_idx", "candidate_arc_id"]
    out = pd.concat([train[cols].assign(p=p_tr), test[cols].assign(p=p_te)], ignore_index=True)
    out.to_parquet(PROCESSED / f"scores_{tag}.parquet", index=False)
    print(f"wrote {len(out):,} scores for {out.trip_id.nunique()} trips")


if __name__ == "__main__":
    run(sys.argv[1])
