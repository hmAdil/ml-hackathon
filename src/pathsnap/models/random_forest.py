"""Model: random forest (bagging family).

Median imputation (speed_ms has NaNs) is inside the pipeline, so it is fit on training folds
only. Trees are depth/leaf limited and each tree sees a row subsample (max_samples) to keep
training affordable on ~1.8M rows. predict_proba col 1 = P(true arc); leaf-averaged votes.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NAME = "random_forest"
FAMILY = "bagging"
DEFAULT_PARAMS = {"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 20,
                  "max_features": "sqrt", "max_samples": 0.3, "n_jobs": 6}
SEARCH_SPACE = {"n_estimators": ("int", 100, 300), "max_depth": ("int", 8, 30),
                "min_samples_leaf": ("int", 5, 100)}


def build(**params):
    p = {**DEFAULT_PARAMS, **params}
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(random_state=42, **p))])
