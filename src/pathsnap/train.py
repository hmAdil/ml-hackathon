"""Train and evaluate one model: python -m pathsnap.train logreg

1. 5-fold GroupKFold CV over the 80 train trips (split by trip).
2. Refit on all train trips, evaluate on the 10 student-test trips (development comparison).
Writes results/models/<name>.json and artifacts/<name>.joblib. Evaluator trips are never read.
"""
import importlib
import json
import sys
import time

import joblib
import pandas as pd

from .config import ARTIFACTS, RANDOM_SEED, RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .cv import cross_validate, evaluate, fit_scorer, load_arcs, score
from .dataset import PROCESSED
from .features import FEATURES


def run(name: str):
    mod = importlib.import_module(f"pathsnap.models.{name}")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "student_test.parquet")
    print(f"{name}: train {len(train):,} rows / {train.trip_id.nunique()} trips, "
          f"student_test {len(test):,} rows / {test.trip_id.nunique()} trips", flush=True)

    cv = cross_validate(mod.build, mod.DEFAULT_PARAMS, train)
    model, fit_s = fit_scorer(mod.build, mod.DEFAULT_PARAMS, train)
    t0 = time.perf_counter()
    p_test = score(model, test)
    inf_s = time.perf_counter() - t0
    st = evaluate(test, p_test, load_arcs(STUDENT_TEST_DIR))
    p_tr = score(model, train)
    tr = evaluate(train, p_tr, load_arcs(TRAIN_DIR))

    res = {"model": name, "family": mod.FAMILY, "params": mod.DEFAULT_PARAMS,
           "features": FEATURES, "seed": RANDOM_SEED,
           "cv": {c: {"mean": float(cv[c].mean()), "std": float(cv[c].std())}
                  for c in ["top1", "top1_covered", "top1_per_trip", "macro_f1", "precision",
                            "recall", "log_loss"]},
           "cv_folds": cv.to_dict("records"),
           "student_test": {**st, "inference_s": inf_s,
                            "inference_ms_per_obs": 1000 * inf_s / st["n_obs"]},
           "train_resubstitution": tr, "fit_s_full_train": fit_s}
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    (RESULTS / "models" / f"{name}.json").write_text(json.dumps(res, indent=2))
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    print(json.dumps({k: res[k] for k in ("cv", "student_test", "train_resubstitution")},
                     indent=1)[:2500])


if __name__ == "__main__":
    run(sys.argv[1])
