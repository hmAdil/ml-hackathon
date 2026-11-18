"""Phase 9: HMM/Viterbi map-matching decoder on top of the candidate classifier.

States    : candidate arcs of each GPS observation (top `max_cands` by classifier p).
Emission  : w_emis * log(p) - w_dist * dist_m^2 / (2 * sigma_m^2)   (w_dist defaults to 0 because
            the classifier already sees distance; avoids double counting).
Transition: from arc a (obs i, projection frac fa) to arc b (obs i+1, frac fb), on the DIRECTED
            network, route distance
                same arc : |fb - fa| * len
                otherwise: (1 - fa) * len_a + sp(head(a) -> tail(b)) + fb * len_b
            with sp a bounded Dijkstra (cached per source node). Log transition is the
            Newson-Krumm term  -w_trans * min(|route - gps_disp| / beta, max_pen)
            minus an implied-speed penalty  speed_w * max(0, route/dt - vmax)
            minus `uturn_pen` when b is the reverse of a.
Fallback  : if b is unreachable within the Dijkstra limit the cost is the finite constant
            w_trans * max_pen (same as the cap on any other mismatch), so the sequence is
            never impossible; the HMM then simply prefers the better-explained pair.
Segments  : GPS gaps > gap_s start a new Viterbi chain (no transition across the gap).
Uses only the track, the network and the scored candidates; never labels.
"""
import heapq

import numpy as np
import pandas as pd

from .geo import make_projector

DEFAULT_PARAMS = {
    "gap_s": 60.0,        # segment break threshold (s)
    "max_cands": 10,      # keep top-k candidates per observation by p
    "w_emis": 1.0,        # weight on log p
    "w_dist": 0.0,        # weight on gaussian distance term (0 = off)
    "sigma_m": 20.0,
    "p_floor": 1e-6,
    "beta": 30.0,         # metres; scale of |route - gps| mismatch
    "w_trans": 1.0,
    "max_pen": 8.0,       # cap on mismatch penalty = fallback penalty for unreachable pairs
    "vmax": 70.0,         # m/s above which implied speed is penalised
    "speed_w": 0.1,       # nats per m/s above vmax
    "uturn_pen": 2.0,
    "limit_mult": 1.5,    # Dijkstra limit = limit_mult * gps_disp + limit_add (capped)
    "limit_add": 250.0,
    "limit_max": 3000.0,
}


class _Net:
    """Directed adjacency in CSR lists plus bounded cached Dijkstra."""

    def __init__(self, n_nodes, fn, tn, length):
        order = np.argsort(fn, kind="stable")
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(fn, minlength=n_nodes))]).tolist()
        self.to = tn[order].tolist()
        self.w = length[order].tolist()
        self.cache: dict[int, tuple[float, dict]] = {}

    def _dijkstra(self, src, limit):
        dist = {src: 0.0}
        done = {}
        heap = [(0.0, src)]
        ip, to, w = self.indptr, self.to, self.w
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            if d > limit:
                break
            done[u] = d
            for k in range(ip[u], ip[u + 1]):
                v = to[k]
                nd = d + w[k]
                if nd <= limit and nd < dist.get(v, 1e30):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return done

    def reach(self, src, limit):
        c = self.cache.get(src)
        if c is None or c[0] < limit:
            lim = float(np.ceil(limit / 250.0) * 250.0)
            c = (lim, self._dijkstra(src, lim))
            self.cache[src] = c
        return c[1]


def _prepare(trip):
    """Per-trip geometry and Dijkstra cache, memoised on the trip dict (params independent)."""
    c = trip.get("_hmm_prep")
    if c is None:
        tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
        proj = make_projector(tr.lon.mean(), tr.lat.mean())
        nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
        px, py = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
        fn, tn = arcs.from_node.to_numpy(np.int64), arcs.to_node.to_numpy(np.int64)
        length = np.hypot(nx[tn] - nx[fn], ny[tn] - ny[fn])
        c = (_Net(len(nodes), fn, tn, length), length, px, py, fn, tn)
        trip["_hmm_prep"] = c
    return c


def _segments(t, gap_s):
    return np.cumsum(np.diff(t, prepend=t[0]) > gap_s)


def decode(trip: dict, scored: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Viterbi decode. scored needs obs_idx, arc_id, p, dist_m, frac, arc_len_m.
    Returns DataFrame[obs_idx, arc_id] with one row per observation that has candidates."""
    P = {**DEFAULT_PARAMS, **(params or {})}
    tr = trip["track"]
    net, length, px, py, fn, tn = _prepare(trip)
    t = tr.t.to_numpy(float)

    s = scored.sort_values(["obs_idx", "p"], ascending=[True, False], kind="stable")
    s = s[s.groupby("obs_idx").cumcount() < P["max_cands"]]
    obs_ids = s.obs_idx.to_numpy()
    starts = np.flatnonzero(np.r_[True, obs_ids[1:] != obs_ids[:-1]])
    ends = np.r_[starts[1:], len(s)]
    obs_list = obs_ids[starts]
    arc_all = s.arc_id.to_numpy(np.int64)
    frac_all = s.frac.to_numpy(float)
    p_all = np.maximum(s.p.to_numpy(float), P["p_floor"])
    d_all = s.dist_m.to_numpy(float)
    emis_all = P["w_emis"] * np.log(p_all) - P["w_dist"] * d_all**2 / (2 * P["sigma_m"] ** 2)

    seg = _segments(t[obs_list], P["gap_s"])
    n = len(obs_list)
    back = [None] * n
    prev = None
    out_arc = np.empty(n, dtype=np.int64)
    out_idx = np.empty(n, dtype=np.int64)

    def finish(first, last, score):  # backtrack segment [first, last]
        j = int(np.argmax(score))
        for k in range(last, first - 1, -1):
            sl = slice(starts[k], ends[k])
            out_arc[k] = arc_all[sl][j]
            out_idx[k] = obs_list[k]
            if k > first:
                j = int(back[k][j])

    first = 0
    for k in range(n):
        sl = slice(starts[k], ends[k])
        em = emis_all[sl]
        if k == 0 or seg[k] != seg[k - 1]:
            if k > 0:
                finish(first, k - 1, prev)
            first, prev = k, em.copy()
            continue
        a, fa = arc_all[starts[k - 1]:ends[k - 1]], frac_all[starts[k - 1]:ends[k - 1]]
        b, fb = arc_all[sl], frac_all[sl]
        o0, o1 = obs_list[k - 1], obs_list[k]
        g = float(np.hypot(px[o1] - px[o0], py[o1] - py[o0]))
        dt = max(t[o1] - t[o0], 1e-3)
        limit = min(P["limit_mult"] * g + P["limit_add"], P["limit_max"])
        heads, tails = tn[a], fn[b]
        uh, ih = np.unique(heads, return_inverse=True)
        ut, it = np.unique(tails, return_inverse=True)
        sp = np.full((len(uh), len(ut)), np.inf)
        for r, h in enumerate(uh.tolist()):
            dd = net.reach(h, limit)
            sp[r] = [dd.get(x, np.inf) for x in ut.tolist()]
        sp = sp[ih][:, it]
        la, lb = length[a][:, None], length[b][None, :]
        route = (1 - fa)[:, None] * la + sp + fb[None, :] * lb
        same = a[:, None] == b[None, :]
        route = np.where(same, np.abs(fb[None, :] - fa[:, None]) * la, route)
        reachable = np.isfinite(route)
        with np.errstate(invalid="ignore"):
            pen = np.minimum(np.abs(route - g) / P["beta"], P["max_pen"])
            pen = np.where(reachable, pen, P["max_pen"])
            vpen = np.where(reachable, P["speed_w"] * np.maximum(0.0, route / dt - P["vmax"]), 0.0)
        pen = P["w_trans"] * pen + vpen
        rev = (tn[a][:, None] == fn[b][None, :]) & (fn[a][:, None] == tn[b][None, :]) & ~same
        pen = pen + P["uturn_pen"] * rev
        tot = prev[:, None] - pen
        arg = tot.argmax(axis=0)
        prev = tot[arg, np.arange(len(b))] + em
        back[k] = arg
    if n:
        finish(first, n - 1, prev)
    return pd.DataFrame({"obs_idx": out_idx, "arc_id": out_arc})
