"""Phase 2 EDA. Usage (repo root): set PYTHONPATH=src && python -m pathsnap.eda

Writes per-trip tables and plots to results/eda/. Every row carries its partition.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .config import RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .geo import haversine_m, make_projector, point_segment_dist
from .io import list_trips, load_trip

OUT = RESULTS / "eda"


def gps_to_route(trip: dict, r: np.ndarray) -> dict:
    """Distance (m) from each GPS point to the nearest reference-route arc (unordered check)."""
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.values, nodes.lat.values)
    a, b = arcs.from_node.values[r], arcs.to_node.values[r]
    ax, ay, bx, by = nx[a], ny[a], nx[b], ny[b]
    # densify route arcs at ~5 m for a KD-tree lookup, then refine to exact segment distance
    n_pts = np.maximum(2, (np.hypot(bx - ax, by - ay) / 5).astype(int) + 1)
    seg = np.repeat(np.arange(len(r)), n_pts)
    f = np.concatenate([np.linspace(0, 1, k) for k in n_pts])
    pts = np.c_[ax[seg] + f * (bx - ax)[seg], ay[seg] + f * (by - ay)[seg]]
    px, py = proj(tr.lon.values, tr.lat.values)
    _, idx = cKDTree(pts).query(np.c_[px, py])
    s = seg[idx]
    d, _ = point_segment_dist(px, py, ax[s], ay[s], bx[s], by[s])
    return {"gps_route_d_median": float(np.median(d)),
            "gps_route_d_p95": float(np.percentile(d, 95)),
            "gps_route_d_max": float(d.max()), "gps_within_30m": float((d <= 30).mean())}


def trip_stats(trip: dict, partition: str) -> dict:
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    route = trip.get("route")
    n_nodes, n_arcs = len(nodes), len(arcs)
    s = {"partition": partition, "trip_id": trip["trip_id"], "n_gps": len(tr),
         "n_nodes": n_nodes, "n_arcs": n_arcs}

    # GPS quality
    s["gps_nan"] = int(tr.isna().sum().sum())
    s["lonlat_invalid"] = int(((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).sum())
    dt = np.diff(tr.t.to_numpy())
    s["t_monotonic"] = bool((dt > 0).all())
    s["dt_median"], s["dt_max"] = float(np.median(dt)), float(dt.max())
    s["n_dt_gt1"], s["n_dt_gt10"] = int((dt > 1).sum()), int((dt > 10).sum())
    s["duration_s"] = float(tr.t.iloc[-1] - tr.t.iloc[0])
    s["lon_min"], s["lon_max"] = float(tr.lon.min()), float(tr.lon.max())
    s["lat_min"], s["lat_max"] = float(tr.lat.min()), float(tr.lat.max())
    step = haversine_m(tr.lon.values[:-1], tr.lat.values[:-1], tr.lon.values[1:], tr.lat.values[1:])
    speed = step / np.maximum(dt, 1e-9)
    s["track_len_m"] = float(step.sum())
    s["speed_p99_ms"], s["speed_max_ms"] = float(np.percentile(speed, 99)), float(speed.max())

    # road network
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    s["arc_ref_invalid"] = int(((fn < 0) | (fn >= n_nodes) | (tn < 0) | (tn >= n_nodes)).sum())
    s["self_loops"] = int((fn == tn).sum())
    s["dup_arcs"] = int(arcs.duplicated().sum())
    s["isolated_nodes"] = int(n_nodes - len(np.unique(np.concatenate([fn, tn]))))
    g = coo_matrix((np.ones(n_arcs), (fn, tn)), shape=(n_nodes, n_nodes))
    s["weak_components"] = int(connected_components(g, directed=False)[0])

    # reference route
    if route is not None:
        r = route.arc_id.to_numpy()
        s["route_len"] = len(r)
        s["route_invalid_ids"] = int(((r < 0) | (r >= n_arcs)).sum())
        if s["route_invalid_ids"] == 0 and len(r) > 1:
            s["route_connected_frac"] = float((tn[r[:-1]] == fn[r[1:]]).mean())
            s["route_repeat_arcs"] = int(len(r) - len(np.unique(r)))
            s.update(gps_to_route(trip, r))
    return s


def make_plots(df: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for part, g in df.groupby("partition"):
        ax[0, 0].hist(g.n_gps, bins=20, alpha=.6, label=part)
        ax[0, 1].hist(g.dt_max.clip(upper=300), bins=20, alpha=.6, label=part)
        ax[1, 0].hist(g.speed_p99_ms, bins=20, alpha=.6, label=part)
        ax[1, 1].hist(g.gps_route_d_median.dropna(), bins=20, alpha=.6, label=part)
    titles = ["GPS points per trip", "max dt per trip (s, clipped at 300)",
              "p99 speed (m/s)", "median GPS-to-route distance (m)"]
    for a, t in zip(ax.ravel(), titles):
        a.set_title(t)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "eda_overview.png", dpi=130)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for part, d in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        for td in list_trips(d):
            rows.append(trip_stats(load_trip(td), part))
            print(part, td.name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_stats.csv", index=False)
    summary = df.groupby("partition").agg(
        trips=("trip_id", "count"), gps=("n_gps", "sum"), nodes=("n_nodes", "sum"),
        arcs=("n_arcs", "sum"), route_entries=("route_len", "sum"),
        gps_median=("n_gps", "median"), route_median=("route_len", "median"),
        dt_max=("dt_max", "max"), n_dt_gt1=("n_dt_gt1", "sum"),
        nonmonotonic_trips=("t_monotonic", lambda x: int((~x).sum())),
        route_disconnected_trips=("route_connected_frac", lambda x: int((x < 1).sum())),
        gps_within_30m_mean=("gps_within_30m", "mean"))
    summary.T.to_csv(OUT / "partition_summary.csv")
    print(summary.T.to_string())
    make_plots(df)


if __name__ == "__main__":
    run()
