"""Phase 4: candidate road arcs for every GPS observation.

Per trip: project to local metres, keep arcs near the track, densify them into sample points
every SAMPLE_M, and query a KD-tree around each GPS point. Exact point-to-segment distance is
then computed for the (obs, arc) pairs found. Duplicate (from, to) arcs are collapsed to their
first arc id (`canon`), so a candidate and the reference label are compared on the same id.

Fallback: observations with fewer than MIN_CANDS arcs inside the radius take their nearest
MIN_CANDS arcs regardless of distance, so every observation has candidates.
"""
from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import make_projector, point_segment_dist

RADIUS_M = 50.0
SAMPLE_M = 10.0
MIN_CANDS = 3
MAX_CANDS = 25
MARGIN_M = 300.0


def canonical_arc_ids(arcs: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """canon[arc_id] = first arc id sharing the same (from, to)."""
    key = arcs.from_node.to_numpy(np.int64) * n_nodes + arcs.to_node.to_numpy(np.int64)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return first[inv]


def build_candidates(trip: dict, radius_m: float = RADIUS_M) -> pd.DataFrame:
    """Rows (obs_idx, arc_id, dist_m, frac); arc_id is canonical. No labels used."""
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    px, py = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    canon = canonical_arc_ids(arcs, len(nodes))

    ax, ay, bx, by = nx[fn], ny[fn], nx[tn], ny[tn]
    keep = ((np.minimum(ax, bx) < px.max() + MARGIN_M) & (np.maximum(ax, bx) > px.min() - MARGIN_M)
            & (np.minimum(ay, by) < py.max() + MARGIN_M)
            & (np.maximum(ay, by) > py.min() - MARGIN_M) & (canon == np.arange(len(arcs))))
    ids = np.where(keep)[0]
    ax, ay, bx, by = ax[ids], ay[ids], bx[ids], by[ids]

    # densify kept arcs into sample points
    length = np.hypot(bx - ax, by - ay)
    n_s = np.maximum(2, np.ceil(length / SAMPLE_M).astype(int) + 1)
    owner = np.repeat(np.arange(len(ids)), n_s)
    t = np.concatenate([np.linspace(0, 1, k) for k in n_s])
    sx, sy = ax[owner] + t * (bx - ax)[owner], ay[owner] + t * (by - ay)[owner]
    tree = cKDTree(np.c_[sx, sy])

    pts = np.c_[px, py]
    hits = tree.query_ball_point(pts, radius_m + SAMPLE_M / 2, return_sorted=False)
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    obs = np.repeat(np.arange(len(pts)), lens)
    loc = owner[np.fromiter(chain.from_iterable(hits), dtype=np.int64, count=lens.sum())]
    pair = np.unique(obs.astype(np.int64) * len(ids) + loc)
    obs, loc = pair // len(ids), pair % len(ids)

    # fallback for observations with too few candidates
    cnt = np.bincount(obs, minlength=len(pts))
    low = np.where(cnt < MIN_CANDS)[0]
    if len(low):
        _, nn = tree.query(pts[low], k=min(len(sx), 60))
        nn = np.atleast_2d(nn)
        o2 = np.repeat(low, nn.shape[1])
        l2 = owner[nn.ravel()]
        pair2 = np.unique(o2.astype(np.int64) * len(ids) + l2)
        obs = np.concatenate([obs[~np.isin(obs, low)], pair2 // len(ids)])
        loc = np.concatenate([loc[~np.isin(pair // len(ids), low)], pair2 % len(ids)])

    d, f = point_segment_dist(px[obs], py[obs], ax[loc], ay[loc], bx[loc], by[loc])
    df = pd.DataFrame({"obs_idx": obs, "arc_id": ids[loc], "dist_m": d.astype(np.float32),
                       "frac": f.astype(np.float32)})
    df = df.sort_values(["obs_idx", "dist_m"], kind="stable")
    rank = df.groupby("obs_idx").cumcount().to_numpy()
    in_radius = df.dist_m.to_numpy() <= radius_m
    df = df[(rank < MAX_CANDS) & (in_radius | (rank < MIN_CANDS))]
    return df.reset_index(drop=True)
