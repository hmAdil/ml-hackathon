"""Phase 4: candidate features. Uses only the GPS track and the road network, never labels,
so the same function runs at inference time.

Groups: spatial (distance, projection), directional (heading vs arc bearing), movement
(speed, sampling interval), network (arc length, degrees, two-way) and relative-to-the-other-
candidates features (ranks, gap to the closest candidate), which carry most of the signal.
"""
import numpy as np
import pandas as pd

from .candidates import build_candidates
from .geo import bearing_deg, make_projector

SEG_GAP_S = 60.0
MIN_MOVE_M = 3.0

FEATURES = [
    "dist_m", "log_dist", "dist_minus_min", "dist_over_min", "dist_rank", "n_cands",
    "frac", "along_m", "to_end_m", "arc_len_m", "log_arc_len",
    "ang_diff", "cos_diff", "heading_valid", "ang_rank", "ang_minus_min",
    "speed_ms", "dt_prev", "dt_next", "is_two_way", "out_deg_to", "in_deg_from",
]


def _heading_and_speed(tr: pd.DataFrame, px, py):
    """Central-difference heading (deg) and speed (m/s) per GPS point, NaN across gaps/stops."""
    n = len(tr)
    t = tr.t.to_numpy(float)
    i0, i1 = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    dtt = t[i1] - t[i0]
    dist = np.hypot(dx, dy)
    heading = bearing_deg(px[i0], py[i0], px[i1], py[i1])
    ok = (dist >= MIN_MOVE_M) & (dtt > 0) & (dtt <= 2 * SEG_GAP_S)
    heading = np.where(ok, heading, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where((dtt > 0) & (dtt <= 2 * SEG_GAP_S), dist / dtt, np.nan)
    dt_prev = np.diff(t, prepend=t[0])
    dt_next = np.diff(t, append=t[-1])
    return heading, speed, dt_prev, dt_next


def build_features(trip: dict, cands: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate table for one trip: obs_idx, arc_id and the FEATURES columns."""
    cands = build_candidates(trip) if cands is None else cands
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    px, py = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()

    heading, speed, dt_prev, dt_next = _heading_and_speed(tr, px, py)
    out_deg = np.bincount(fn, minlength=len(nodes))
    in_deg = np.bincount(tn, minlength=len(nodes))
    n_nodes = len(nodes)
    pair_set = np.sort(fn.astype(np.int64) * n_nodes + tn)
    rev_key = tn.astype(np.int64) * n_nodes + fn
    two_way = np.isin(rev_key, pair_set)

    o, a = cands.obs_idx.to_numpy(), cands.arc_id.to_numpy()
    f = cands.copy()
    f["arc_len_m"] = np.hypot(nx[tn[a]] - nx[fn[a]], ny[tn[a]] - ny[fn[a]]).astype(np.float32)
    f["along_m"] = f.frac * f.arc_len_m
    f["to_end_m"] = (1 - f.frac) * f.arc_len_m
    f["log_arc_len"] = np.log1p(f.arc_len_m)
    f["log_dist"] = np.log1p(f.dist_m)

    g = f.groupby("obs_idx")
    dmin = g.dist_m.transform("min")
    f["dist_minus_min"] = f.dist_m - dmin
    f["dist_over_min"] = f.dist_m / (dmin + 1.0)
    f["dist_rank"] = g.dist_m.rank(method="first")
    f["n_cands"] = g.dist_m.transform("size")

    ab = bearing_deg(nx[fn[a]], ny[fn[a]], nx[tn[a]], ny[tn[a]])
    h = heading[o]
    valid = ~np.isnan(h)
    diff = np.abs((np.nan_to_num(h) - ab + 180.0) % 360.0 - 180.0)
    f["heading_valid"] = valid.astype(np.float32)
    f["ang_diff"] = np.where(valid, diff, 90.0).astype(np.float32)   # 90 = uninformative
    f["cos_diff"] = np.where(valid, np.cos(np.radians(diff)), 0.0).astype(np.float32)
    g = f.groupby("obs_idx")
    f["ang_rank"] = g.ang_diff.rank(method="first")
    f["ang_minus_min"] = f.ang_diff - g.ang_diff.transform("min")

    f["speed_ms"] = speed[o]
    f["dt_prev"] = dt_prev[o]
    f["dt_next"] = dt_next[o]
    f["is_two_way"] = two_way[a].astype(np.float32)
    f["out_deg_to"] = out_deg[tn[a]].astype(np.float32)
    f["in_deg_from"] = in_deg[fn[a]].astype(np.float32)
    return f[["obs_idx", "arc_id", *FEATURES]]
