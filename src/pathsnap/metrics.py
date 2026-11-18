"""Shared metrics (candidate-table level and route level)."""
import numpy as np
import pandas as pd


def top1_accuracy(trip_ids, obs_idx, scores, is_true) -> float:
    """Per (trip, obs) pick the highest-scoring candidate; observations with no true
    candidate count as wrong."""
    df = pd.DataFrame({"t": trip_ids, "o": obs_idx, "p": scores, "y": is_true})
    best = df.loc[df.groupby(["t", "o"])["p"].idxmax()]
    return float(best["y"].mean())


def candidate_miss_rate(trip_ids, obs_idx, is_true) -> float:
    df = pd.DataFrame({"t": trip_ids, "o": obs_idx, "y": is_true})
    return float(1.0 - df.groupby(["t", "o"])["y"].max().mean())


def route_continuity(arc_seq, arcs_from, arcs_to) -> float:
    """Fraction of consecutive predicted arcs that are equal or head-to-tail connected."""
    arc_seq = np.asarray(arc_seq)
    if len(arc_seq) < 2:
        return 1.0
    a, b = arc_seq[:-1], arc_seq[1:]
    arcs_from, arcs_to = np.asarray(arcs_from), np.asarray(arcs_to)
    return float(np.mean((a == b) | (arcs_to[a] == arcs_from[b])))
