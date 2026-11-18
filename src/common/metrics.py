"""Shared metrics used by every model and by the HMM comparison."""
import numpy as np


def top1_accuracy(trip_ids, obs_idx, probs, is_true) -> float:
    """Observation-level Top-1: per (trip, obs) pick the highest-scoring candidate."""
    import pandas as pd
    df = pd.DataFrame({"t": trip_ids, "o": obs_idx, "p": probs, "y": is_true})
    best = df.loc[df.groupby(["t", "o"])["p"].idxmax()]
    return float(best["y"].mean())


def candidate_miss_rate(trip_ids, obs_idx, is_true) -> float:
    """Share of observations whose reference arc is absent from the candidate set."""
    import pandas as pd
    df = pd.DataFrame({"t": trip_ids, "o": obs_idx, "y": is_true})
    return float(1.0 - df.groupby(["t", "o"])["y"].max().mean())


def route_continuity(arc_seq, arcs_from, arcs_to) -> float:
    """Fraction of consecutive predicted arcs that are equal or directly connected."""
    arc_seq = np.asarray(arc_seq)
    if len(arc_seq) < 2:
        return 1.0
    ok = sum(
        a == b or arcs_to[a] == arcs_from[b]
        for a, b in zip(arc_seq[:-1], arc_seq[1:])
    )
    return ok / (len(arc_seq) - 1)
