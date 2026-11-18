"""Phase 2.3: road-network exploration (per trip). Arc/node IDs are zero-based row indices
and are local to each trip's network."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geo import make_projector, point_segment_dist


def arc_lengths_m(nodes: pd.DataFrame, arcs: pd.DataFrame, lon0: float, lat0: float):
    proj = make_projector(lon0, lat0)
    x, y = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    return np.hypot(x[tn] - x[fn], y[tn] - y[fn])


def densify(ax, ay, bx, by, step_m: float = 5.0):
    """Sample points every ~step_m along each segment; returns (points, segment_index)."""
    n = np.maximum(2, (np.hypot(bx - ax, by - ay) / step_m).astype(int) + 1)
    seg = np.repeat(np.arange(len(ax)), n)
    f = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(n) else np.empty(0)
    pts = np.c_[ax[seg] + f * (bx - ax)[seg], ay[seg] + f * (by - ay)[seg]]
    return pts, seg


def nearest_arc_dist(track, nodes, arcs, pad_m: float = 300.0):
    """Exact distance (m) from each GPS point to its nearest network arc, plus the arc id.
    Only arcs near the track bbox are indexed, so this stays cheap on 300k-arc networks."""
    lon0, lat0 = track.lon.mean(), track.lat.mean()
    proj = make_projector(lon0, lat0)
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    px, py = proj(track.lon.to_numpy(), track.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    x0, x1, y0, y1 = px.min() - pad_m, px.max() + pad_m, py.min() - pad_m, py.max() + pad_m
    keep = np.where(((nx[fn] >= x0) & (nx[fn] <= x1) & (ny[fn] >= y0) & (ny[fn] <= y1))
                    | ((nx[tn] >= x0) & (nx[tn] <= x1) & (ny[tn] >= y0) & (ny[tn] <= y1)))[0]
    if len(keep) == 0:
        return np.full(len(px), np.inf), np.full(len(px), -1)
    ax, ay, bx, by = nx[fn[keep]], ny[fn[keep]], nx[tn[keep]], ny[tn[keep]]
    pts, seg = densify(ax, ay, bx, by)
    # k nearest densified samples, then refine to exact point-segment distance
    k = min(8, len(pts))
    _, ii = cKDTree(pts).query(np.c_[px, py], k=k)
    ii = ii.reshape(len(px), k)
    cand = seg[ii]
    d, _ = point_segment_dist(px[:, None], py[:, None], ax[cand], ay[cand], bx[cand], by[cand])
    j = d.argmin(1)
    r = np.arange(len(px))
    return d[r, j], keep[cand[r, j]]


def network_analysis(nodes: pd.DataFrame, arcs: pd.DataFrame, track: pd.DataFrame | None = None):
    """Returns (summary dict, aux dict). aux holds per-node/arc arrays used by the route EDA
    and by plots: in/out degree, strong-component labels, arc lengths."""
    n_nodes, n_arcs = len(nodes), len(arcs)
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    s = {"n_nodes": n_nodes, "n_arcs": n_arcs}
    s["node_nan"] = int(nodes.isna().sum().sum())
    s["arc_ref_invalid"] = int(((fn < 0) | (fn >= n_nodes) | (tn < 0) | (tn >= n_nodes)).sum())
    if s["arc_ref_invalid"]:
        return s, {}
    s["self_loops"] = int((fn == tn).sum())
    key = fn.astype(np.int64) * n_nodes + tn
    s["dup_arcs"] = int(n_arcs - len(np.unique(key)))
    s["dup_node_coords"] = int(nodes.duplicated().sum())
    rev = tn.astype(np.int64) * n_nodes + fn
    s["two_way_frac"] = float(np.isin(rev, key).mean())

    indeg = np.bincount(tn, minlength=n_nodes)
    outdeg = np.bincount(fn, minlength=n_nodes)
    deg = indeg + outdeg
    s["isolated_nodes"] = int((deg == 0).sum())
    s["dead_end_nodes"] = int(((outdeg == 0) & (indeg > 0)).sum())   # can enter, not leave
    s["source_only_nodes"] = int(((indeg == 0) & (outdeg > 0)).sum())  # can leave, not enter
    s["outdeg_mean"] = float(outdeg.mean())
    s["deg_max"] = int(deg.max()) if n_nodes else 0
    s["deg2_frac"] = float((deg == 2).mean())  # mid-road shape points (in1/out1 one-way)

    g = coo_matrix((np.ones(n_arcs), (fn, tn)), shape=(n_nodes, n_nodes)).tocsr()
    n_w, w_lab = connected_components(g, directed=True, connection="weak")
    n_s, s_lab = connected_components(g, directed=True, connection="strong")
    used = deg > 0
    s["weak_components"] = int(n_w - s["isolated_nodes"])
    s["strong_components"] = int(n_s - s["isolated_nodes"])
    s["largest_wcc_frac"] = float(np.bincount(w_lab[used]).max() / max(used.sum(), 1))
    s["largest_scc_frac"] = float(np.bincount(s_lab[used]).max() / max(used.sum(), 1))

    lon0 = track.lon.mean() if track is not None else nodes.lon.mean()
    lat0 = track.lat.mean() if track is not None else nodes.lat.mean()
    L = arc_lengths_m(nodes, arcs, lon0, lat0)
    s["zero_len_arcs"] = int((L < 0.01).sum())
    for q in (5, 50, 95):
        s[f"arc_len_p{q}_m"] = float(np.percentile(L, q))
    s["arc_len_max_m"] = float(L.max())
    s["net_lon_min"], s["net_lon_max"] = float(nodes.lon.min()), float(nodes.lon.max())
    s["net_lat_min"], s["net_lat_max"] = float(nodes.lat.min()), float(nodes.lat.max())

    aux = {"indeg": indeg, "outdeg": outdeg, "scc": s_lab, "wcc": w_lab, "arc_len": L}
    if track is not None and len(track):
        # how far the network is from the observations: drives the candidate radius
        d, _ = nearest_arc_dist(track, nodes, arcs)
        aux["gps_nearest_arc_d"] = d
        for q in (50, 95, 99):
            s[f"gps_near_arc_p{q}_m"] = float(np.percentile(d, q))
        s["gps_near_arc_max_m"] = float(d.max())
        s["gps_no_arc_within_50m"] = int((d > 50).sum())
        scc_sizes = np.bincount(s_lab)
        s["largest_scc_nodes"] = int(scc_sizes.max())
    return s, aux
