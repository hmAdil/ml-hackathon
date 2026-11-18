"""Phase 2.4: reference-route exploration (per trip).

The .route file is an ordered arc-id sequence with no timestamps, so these checks answer:
is the route valid on its own network, is it continuous, does it follow the GPS, and is a
monotone GPS -> route-position alignment plausible (needed for Phase 3 labels)?
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .eda_network import nearest_arc_dist
from .geo import haversine_m

NEAR_M = 30.0        # "GPS point is on this road" tolerance used throughout the EDA
MISMATCH_M = 20.0    # route is this much further than the nearest road -> route disagrees


def route_analysis(trip: dict, aux: dict | None = None, net_d: np.ndarray | None = None):
    """Returns (summary dict, per-GPS-point alignment DataFrame or None).
    aux/net_d come from eda_network.network_analysis and are optional."""
    nodes, arcs, tr = trip["nodes"], trip["arcs"], trip["track"]
    r = trip["route"].arc_id.to_numpy()
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    s = {"route_len": len(r), "route_invalid_ids": int(((r < 0) | (r >= len(arcs))).sum())}
    if s["route_invalid_ids"] or len(r) == 0:
        return s, None
    a, b = r[:-1], r[1:]
    same = a == b
    conn = tn[a] == fn[b]
    s["route_consecutive_dup"] = int(same.sum())
    s["route_uturns"] = int((conn & (fn[a] == tn[b]) & ~same).sum())
    brk = ~(conn | same)
    s["route_breaks"] = int(brk.sum())
    s["route_connected_frac"] = float(1 - brk.mean()) if len(a) else 1.0
    s["route_repeat_arcs"] = int(len(r) - len(np.unique(r)))
    if brk.any():
        gap = haversine_m(nodes.lon.values[tn[a[brk]]], nodes.lat.values[tn[a[brk]]],
                          nodes.lon.values[fn[b[brk]]], nodes.lat.values[fn[b[brk]]])
        s["route_break_gap_median_m"] = float(np.median(gap))
        s["route_break_gap_max_m"] = float(gap.max())
    if aux:
        L = aux["arc_len"]
        s["route_len_m"] = float(L[r].sum())
        scc = aux["scc"]
        main = np.bincount(scc).argmax()
        s["route_arcs_in_main_scc_frac"] = float(((scc[fn[r]] == main) & (scc[tn[r]] == main))
                                                 .mean())
        s["route_dead_end_arcs"] = int((aux["outdeg"][tn[r[:-1]]] == 0).sum())

    # GPS <-> route geometry: nearest route arc for every GPS point
    sub = arcs.iloc[r].reset_index(drop=True)          # row k == route position k
    d, pos = nearest_arc_dist(tr, nodes, sub, pad_m=1e7)
    s["gps_route_d_median"] = float(np.median(d))
    s["gps_route_d_p95"] = float(np.percentile(d, 95))
    s["gps_route_d_max"] = float(d.max())
    s["gps_within_30m"] = float((d <= NEAR_M).mean())
    covered = np.zeros(len(r), bool)
    covered[pos[d <= NEAR_M]] = True
    s["route_arcs_near_gps_frac"] = float(covered.mean())  # rest is unobserved/interpolated
    # monotonicity of nearest route position along time: ~1 means a monotone alignment
    # (Phase 3) is plausible; backward jumps flag loops, repeats or route/GPS mismatch
    ok = d <= NEAR_M
    if ok.sum() > 2:
        p = pos[ok]
        s["align_spearman"] = float(spearmanr(np.arange(len(p)), p).statistic)
        s["align_backward_frac"] = float((np.diff(p) < 0).mean())
        s["align_big_back_jumps"] = int((np.diff(p) < -5).sum())
    s["start_gps_to_route0_m"] = float(_pt_arc(tr.iloc[0], nodes, arcs, r[0]))
    s["end_gps_to_routeN_m"] = float(_pt_arc(tr.iloc[-1], nodes, arcs, r[-1]))
    s["first_gps_nearest_pos"] = int(pos[0])
    s["last_gps_nearest_pos"] = int(pos[-1])
    if net_d is not None:
        s["route_mismatch_frac"] = float(((d - net_d) > MISMATCH_M).mean())
    align = pd.DataFrame({"route_d": d, "route_pos": pos})
    if net_d is not None:
        align["net_d"] = net_d
    return s, align


def _pt_arc(p, nodes, arcs, arc_id):
    one = pd.DataFrame({"lon": [p.lon], "lat": [p.lat]})
    d, _ = nearest_arc_dist(one, nodes, arcs.iloc[[arc_id]].reset_index(drop=True), pad_m=1e7)
    return d[0]
