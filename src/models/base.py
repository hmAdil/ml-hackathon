"""Every experimented model must expose predict_proba on the candidate features."""
from typing import Protocol

import numpy as np


class CandidateScorer(Protocol):
    def fit(self, X, y): ...
    def predict_proba(self, X) -> np.ndarray: ...  # shape (n, 2); col 1 = P(is_true)
