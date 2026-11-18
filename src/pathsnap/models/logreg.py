"""Model 1: logistic regression (linear / probabilistic baseline).

Scores each (GPS point, candidate arc) pair. Standardised features + L2 logistic loss, so the
output is a calibrated-ish probability that can feed the HMM emission later. Preprocessing is
inside the pipeline, so it is fit on training folds only.
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "logreg"
FAMILY = "linear"
DEFAULT_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 500}
SEARCH_SPACE = {"C": ("log", 1e-3, 1e2)}


def build(**params):
    p = {**DEFAULT_PARAMS, **params}
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(random_state=42, **p))])
