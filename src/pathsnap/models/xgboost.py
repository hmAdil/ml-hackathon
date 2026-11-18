"""Model: gradient-boosted trees on GPU (XGBoost, hist). Handles NaNs natively.

Early stopping uses a validation block made of the contiguous tail of the training rows
(rows are ordered by trip, so the tail is essentially whole trips, no leakage across rows
of the same trip). The model is then used with the best iteration.
"""
import numpy as np
from xgboost import XGBClassifier

NAME = "xgboost"
FAMILY = "boosting"
DEFAULT_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 7,
                  "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_lambda": 1.0, "max_bin": 128, "early_stopping_rounds": 40,
                  "val_frac": 0.1}
SEARCH_SPACE = {"learning_rate": ("log", 0.01, 0.3), "max_depth": ("int", 3, 10),
                "min_child_weight": ("float", 1.0, 20.0), "subsample": ("float", 0.5, 1.0),
                "colsample_bytree": ("float", 0.4, 1.0), "reg_lambda": ("log", 1e-2, 50.0),
                "n_estimators": ("int", 200, 1000)}


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class XGBScorer:
    def __init__(self, val_frac=0.1, early_stopping_rounds=40, **params):
        self.val_frac, self.esr, self.params = val_frac, early_stopping_rounds, params

    def fit(self, X, y):
        X, y = np.asarray(X, np.float32), np.asarray(y)
        cut = int(len(X) * (1 - self.val_frac))
        self.model_ = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            device=_device(), random_state=42, n_jobs=4,
            early_stopping_rounds=self.esr, **self.params)
        self.model_.fit(X[:cut], y[:cut], eval_set=[(X[cut:], y[cut:])], verbose=False)
        self.best_iteration_ = int(self.model_.best_iteration)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, np.float32))


def build(**params):
    return XGBScorer(**{**DEFAULT_PARAMS, **params})
