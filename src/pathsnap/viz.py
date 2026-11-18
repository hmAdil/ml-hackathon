"""Folium maps and static trip plots.

trip_map      one trip: GPS, nearby network, reference route, optional prediction, plus EDA
              layers (gaps, spikes, off-route points, route breaks, start/end)
world_map     every trip's location from results/eda/trip_stats.csv, with its quality flags
trip_png      static matplotlib version of trip_map for the report

Usage: python -m pathsnap.viz 00000001 | python -m pathsnap.viz --world
"""
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from .config import RESULTS

GPS_C, ROUTE_C, PRED_C, NET_C = "#2a78d6", "#008300", "#e34948", "#888888"
WARN_C = "#eb6834"


def _network_subset(tr, nodes, arcs, pad=0.002, cap=4000):
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    fn = arcs.from_node.to_numpy()
    inb = ((lon[fn] > tr.lon.min() - pad) & (lon[fn] < tr.lon.max() + pad)
           & (lat[fn] > tr.lat.min() - pad) & (lat[fn] < tr.lat.max() + pad))
    idx = np.where(inb)[0]
    if len(idx) > cap:  # spread the cap over the whole bbox instead of the first rows
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    return idx


def trip_map(trip: dict, pred_arcs=None, out: Path | None = None, max_network_arcs: int = 4000,
             eda_layers: bool = True):
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()

    def line(i):
        return [(lat[fn[i]], lon[fn[i]]), (lat[tn[i]], lon[tn[i]])]

    net = folium.FeatureGroup("Road network (grey)", show=False).add_to(m)
    for i in _network_subset(tr, nodes, arcs, cap=max_network_arcs):
        folium.PolyLine(line(i), color=NET_C, weight=1).add_to(net)

    gps = folium.FeatureGroup("GPS observations (blue)").add_to(m)
    for k, r in enumerate(tr.itertuples()):
        folium.CircleMarker((r.lat, r.lon), radius=2, color=GPS_C, fill=True,
                            tooltip=f"obs {k}, t={r.t:g}").add_to(gps)
    if "route" in trip:
        ref = folium.FeatureGroup("Reference route (green)").add_to(m)
        for k, a in enumerate(trip["route"].arc_id):
            folium.PolyLine(line(a), color=ROUTE_C, weight=4, opacity=0.8,
                            tooltip=f"route pos {k}, arc {a}").add_to(ref)
    if pred_arcs is not None:
        pr = folium.FeatureGroup("Predicted route (red dashed)").add_to(m)
        for a in pred_arcs:
            folium.PolyLine(line(a), color=PRED_C, weight=3, dash_array="6").add_to(pr)
    if eda_layers:
        _eda_layers(m, trip, line)
    folium.LayerControl(collapsed=False).add_to(m)
    out = out or RESULTS / "viz" / f"{trip['trip_id']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def _eda_layers(m, trip, line):
    from .eda_gps import point_table, spike_mask

    tr = trip["track"]
    pt = point_table(tr)
    la, lo = tr.lat.to_numpy(), tr.lon.to_numpy()
    se = folium.FeatureGroup("Start / end").add_to(m)
    folium.Marker((la[0], lo[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(se)
    folium.Marker((la[-1], lo[-1]), tooltip="end", icon=folium.Icon(color="black")).add_to(se)

    gaps = folium.FeatureGroup("Sampling gaps > 10 s (orange dashed)").add_to(m)
    for i in np.where(pt.dt.to_numpy() > 10)[0]:
        folium.PolyLine([(la[i - 1], lo[i - 1]), (la[i], lo[i])], color=WARN_C, weight=3,
                        dash_array="8", tooltip=f"gap {pt.dt.iloc[i]:g} s, "
                        f"{pt.step_m.iloc[i]:.0f} m").add_to(gaps)
    sp = folium.FeatureGroup("Spikes (orange)").add_to(m)
    for i in np.where(spike_mask(pt))[0]:
        folium.CircleMarker((la[i], lo[i]), radius=6, color=WARN_C, fill=True,
                            tooltip=f"spike obs {i}, {pt.speed_ms.iloc[i]:.0f} m/s").add_to(sp)
    if "route" not in trip:
        return
    from .eda_network import nearest_arc_dist

    r = trip["route"].arc_id.to_numpy()
    d, _ = nearest_arc_dist(tr, trip["nodes"], trip["arcs"].iloc[r].reset_index(drop=True),
                            pad_m=1e7)
    off = folium.FeatureGroup("GPS > 30 m from route (orange)").add_to(m)
    for i in np.where(d > 30)[0]:
        folium.CircleMarker((la[i], lo[i]), radius=4, color=WARN_C, fill=True,
                            tooltip=f"obs {i}: {d[i]:.0f} m from route").add_to(off)
    fn, tn = trip["arcs"].from_node.to_numpy(), trip["arcs"].to_node.to_numpy()
    brk = np.where((tn[r[:-1]] != fn[r[1:]]) & (r[:-1] != r[1:]))[0]
    bl = folium.FeatureGroup("Route breaks (red)").add_to(m)
    for k in brk:
        (a0, a1), (b0, b1) = line(r[k])[1], line(r[k + 1])[0]
        folium.PolyLine([(a0, a1), (b0, b1)], color=PRED_C, weight=4,
                        tooltip=f"break after route pos {k}").add_to(bl)


def world_map(stats_csv: Path | None = None, out: Path | None = None):
    """One marker per trip at its GPS bbox centre, coloured by partition, with key stats."""
    df = pd.read_csv(stats_csv or RESULTS / "eda" / "trip_stats.csv", dtype={"trip_id": str})
    fpath = RESULTS / "eda" / "trip_flags.csv"
    flags = pd.read_csv(fpath, dtype={"trip_id": str}) if fpath.exists() else None
    m = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    groups = {p: folium.FeatureGroup(f"{p} trips").add_to(m) for p in df.partition.unique()}
    colors = {"train": GPS_C, "student_test": WARN_C}
    for r in df.itertuples():
        bad = ""
        if flags is not None:
            f = flags[flags.trip_id == r.trip_id].iloc[0]
            bad = ", ".join(c for c in flags.columns[2:-1] if f[c]) or "none"
        tip = (f"<b>{r.trip_id}</b> ({r.partition})<br>GPS {r.n_gps}, route arcs {r.route_len}"
               f"<br>network arcs {r.n_arcs}<br>duration {r.duration_s / 60:.0f} min, "
               f"{r.track_len_m / 1000:.1f} km<br>max dt {r.dt_max:g} s<br>"
               f"GPS within 30 m of route {r.gps_within_30m:.1%}<br>flags: {bad}")
        folium.CircleMarker(((r.lat_min + r.lat_max) / 2, (r.lon_min + r.lon_max) / 2),
                            radius=6, color=colors.get(r.partition, NET_C), fill=True,
                            fill_opacity=0.8, tooltip=folium.Tooltip(tip)).add_to(
            groups[r.partition])
    folium.LayerControl(collapsed=False).add_to(m)
    out = out or RESULTS / "viz" / "world_trips.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def trip_png(trip: dict, out: Path | None = None):
    """Static plot in local metres: network (grey), route (green), GPS (blue)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from .geo import make_projector

    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    px, py = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()

    def segs(ids):
        return np.stack([np.c_[nx[fn[ids]], ny[fn[ids]]], np.c_[nx[tn[ids]], ny[tn[ids]]]], 1)

    w, h = np.ptp(px) + 600, np.ptp(py) + 600
    k = 9 / max(w, h)
    fig, ax = plt.subplots(figsize=(max(4, w * k) + 1.5, max(4, h * k)), facecolor="#fcfcfb")
    ax.add_collection(LineCollection(segs(_network_subset(tr, nodes, arcs, cap=20000)),
                                     colors=NET_C, linewidths=0.4, label="road network"))
    if "route" in trip:
        ax.add_collection(LineCollection(segs(trip["route"].arc_id.to_numpy()),
                                         colors=ROUTE_C, linewidths=4, alpha=0.7,
                                         label="reference route", zorder=2))
    ax.scatter(px, py, s=1, color=GPS_C, zorder=3, label="GPS")
    ax.set_xlim(px.min() - 300, px.max() + 300)
    ax.set_ylim(py.min() - 300, py.max() + 300)
    ax.set_aspect("equal")
    ax.set_title(f"Trip {trip['trip_id']} (local metres)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.tick_params(labelsize=7)
    out = out or RESULTS / "viz" / f"{trip['trip_id']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    if "--world" in sys.argv:
        print(world_map())
    else:
        from .eda_trip import get_trip

        tid = sys.argv[1] if len(sys.argv) > 1 else "00000001"
        part = sys.argv[2] if len(sys.argv) > 2 else "train"
        t = get_trip(tid, part)
        print(trip_png(t))
        # keep the committed source map intact when working from a recovered trip
        name = f"{tid}_eda.html" if t.get("source") != "raw" else f"{tid}.html"
        print(trip_map(t, out=RESULTS / "viz" / name))
