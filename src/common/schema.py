"""Shared interface contract between data/features, models and HMM/API.

Candidates table (pandas DataFrame / parquet), one row per
(trip_id, obs_idx, candidate_arc_id):
    trip_id, obs_idx, candidate_arc_id, <feature columns...>, is_true (train/eval only)
"""
KEY_COLS = ["trip_id", "obs_idx", "candidate_arc_id"]
LABEL_COL = "is_true"
RANDOM_SEED = 42
