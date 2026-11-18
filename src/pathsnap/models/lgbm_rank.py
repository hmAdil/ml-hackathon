"""Model: LightGBM LambdaRank (ranking family).

Trained with objective 'lambdarank', one query group per GPS observation (cv.fit_scorer passes
`groups=` = per-observation group sizes in row order). LightGBM handles NaNs natively.

The raw ranker score is only meaningful for ordering within an observation, so log loss on it
is meaningless. predict_proba therefore returns a Platt-style calibration: a 1-D
LogisticRegression (C=1e6, i.e. effectively unregularised) fit inside fit() on the
training-set raw scores vs is_true. Column 1 = sigmoid(a*score + b), in (0,1). The ranking
(argmax per observation) is unchanged because the map is monotone for a > 0. Log loss is only
meaningful through this calibrated probability.
"""
import numpy as np
from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression

NAME = "lgbm_rank"
FAMILY = "ranking"
DEFAULT_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63,
                  "min_child_samples": 50, "subsample": 0.5, "subsample_freq": 1,
                  "colsample_bytree": 0.8, "n_jobs": 6}
SEARCH_SPACE = {"num_leaves": ("int", 15, 127), "learning_rate": ("log", 0.01, 0.2),
                "n_estimators": ("int", 100, 600)}


class CalibratedRanker:
    needs_groups = True

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("groups (per-observation sizes) required")
        self.ranker_ = LGBMRanker(objective="lambdarank", random_state=42, verbose=-1,
                                  **self.params)
        self.ranker_.fit(X, y, group=np.asarray(groups))
        s = self.ranker_.predict(X).reshape(-1, 1)
        self.platt_ = LogisticRegression(C=1e6, max_iter=200).fit(s, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.ranker_.predict(X)

    def predict_proba(self, X):
        return self.platt_.predict_proba(self.decision_function(X).reshape(-1, 1))


def build(**params):
    return CalibratedRanker(**{**DEFAULT_PARAMS, **params})
