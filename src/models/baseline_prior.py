"""Reference model file: predicts the class prior. Copy this shape for every new model."""
from sklearn.dummy import DummyClassifier

NAME = "baseline_prior"
FAMILY = "baseline"
DEFAULT_PARAMS = {"strategy": "prior"}
SEARCH_SPACE = {}


def build(**params):
    return DummyClassifier(**{**DEFAULT_PARAMS, **params})
