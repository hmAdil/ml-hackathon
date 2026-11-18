"""Phase 3: per-GPS-point reference labels by monotone alignment to the reference route.

`.route` is an ordered arc list with no timestamps (about 5x fewer arcs than GPS points), so a
point's label cannot come from its row number or from the plain nearest route arc (wrong at
junctions and revisits). Instead we run a Viterbi-style DP whose states are route positions:

    dp[i, j] = cost(point i, route arc j) + min_{0<=k<=K} dp[i-1, j-k] + JUMP_PEN * k

Route position never decreases, may stay, and may skip up to K arcs per step (short arcs are
often skipped at speed). The trip is cut at gaps > SEG_GAP_S; each segment starts at or after
the previous segment's last position, so order is preserved across gaps without forcing a path.

Only the track, the road network and the route are used, so this runs on any partition that has
a route. It is a labelling tool and is never part of inference.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .geo import make_projector, point_segment_dist

K_SKIP = 30         # max route arcs skipped between consecutive samples
JUMP_PEN = 0.05     # cost per arc advanced; makes "stay" and "advance by 1" cheapest
SIGMA_M = 10.0      # emission scale; cost = 0.5 * min(d / SIGMA, CAP)^2 (robust to outliers)
CAP = 8.0
SEG_GAP_S = 60.0
BIG = 1e9


def _cost_matrix(px, py, ax, ay, bx, by):
    n, r = len(px), len(ax)
    d = np.empty((n, r), dtype=np.float32)
    step = max(1, 4_000_000 // max(r, 1))
    for s in range(0, n, step):
        e = min(n, s + step)
        dd, _ = point_segment_dist(px[s:e, None], py[s:e, None], ax[None], ay[None],
                                   bx[None], by[None])
        d[s:e] = dd
    return d


def _viterbi_segment(cost: np.ndarray, start_min: int) -> np.ndarray:
    """Best non-decreasing route position per row of `cost` (n x R), starting at >= start_min."""
    n, r = cost.shape
    pen = (JUMP_PEN * (K_SKIP - np.arange(K_SKIP + 1))).astype(np.float32)
    dp = cost[0].copy()
    dp[:start_min] = BIG
    bp = np.zeros((n, r), dtype=np.uint8)
    pad = np.full(K_SKIP, BIG, dtype=np.float32)
    ar = np.arange(r)
    for i in range(1, n):
        win = sliding_window_view(np.concatenate([pad, dp]), K_SKIP + 1)  # (r, K+1)
        tot = win + pen
        m = tot.argmin(axis=1)
        dp = cost[i] + tot[ar, m]
        bp[i] = K_SKIP - m
    pos = np.empty(n, dtype=np.int64)
    j = int(dp.argmin())
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j -= int(bp[i, j])
    return pos


def align(trip: dict) -> pd.DataFrame:
    """Per GPS row: route_pos, arc_id (reference arc) and align_dist_m (point-to-arc distance)."""
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    route = trip["route"].arc_id.to_numpy()
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    a, b = arcs.from_node.to_numpy()[route], arcs.to_node.to_numpy()[route]
    ax, ay, bx, by = nx[a], ny[a], nx[b], ny[b]
    px, py = proj(tr.lon.to_numpy(), tr.lat.to_numpy())

    dt = np.diff(tr.t.to_numpy(float), prepend=tr.t.iloc[0])
    seg = np.cumsum(dt > SEG_GAP_S)
    pos = np.empty(len(tr), dtype=np.int64)
    dist = np.empty(len(tr), dtype=np.float32)
    start = 0
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        d = _cost_matrix(px[idx], py[idx], ax, ay, bx, by)
        cost = (0.5 * np.minimum(d / SIGMA_M, CAP) ** 2).astype(np.float32)
        p = _viterbi_segment(cost, start)
        pos[idx] = p
        dist[idx] = d[np.arange(len(idx)), p]
        start = int(p[-1])
    return pd.DataFrame({"obs_idx": np.arange(len(tr)), "route_pos": pos,
                         "arc_id": route[pos], "align_dist_m": dist})
