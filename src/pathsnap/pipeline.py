"""Inference pipeline shared by evaluation, the HMM experiments and the API.

trip dict (io.load_trip, route optional) -> candidates + features -> model scores ->
(optional) HMM/Viterbi -> one arc per GPS observation. Never reads labels.

predict_trip returns DataFrame[obs_idx, arc_id]; arc_id is the canonical arc id
(candidates.canonical_arc_ids), which is a valid original arc id.
"""
import pandas as pd

from .features import FEATURES, build_features


def score_candidates(trip: dict, model) -> pd.DataFrame:
    """Feature table for the trip plus column `p` = model P(candidate is the true arc)."""
    f = build_features(trip)
    f["p"] = model.predict_proba(f[FEATURES].to_numpy("float32"))[:, 1]
    return f


def predict_trip(trip: dict, model, use_hmm: bool = True, hmm_params: dict | None = None):
    f = score_candidates(trip, model)
    if use_hmm:
        from .hmm import decode  # decode(trip, scored_candidates, params) -> [obs_idx, arc_id]

        return decode(trip, f, hmm_params or {})
    top = f.loc[f.groupby("obs_idx").p.idxmax()]
    return top[["obs_idx", "arc_id"]].reset_index(drop=True)
