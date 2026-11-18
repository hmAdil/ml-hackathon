"""Phases 2.2 + 2.4 deep dive for one trip: GPS sampling, speeds, GPS-to-route geometry and
the GPS -> route-position alignment that Phase 3 labels depend on.

Usage: python -m pathsnap.eda_trip 00000001 [train|student_test]
Loads the raw trip if present, otherwise falls back to results/viz/<id>.html (partial).
"""
import json
import sys

import numpy as np

from .config import RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .eda_gps import gap_table, gps_summary, point_table, spike_mask
from .eda_inventory import GRID, INK, INK2, SURF, _style
from .eda_network import network_analysis
from .eda_route import NEAR_M, route_analysis
from .io import load_trip

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def get_trip(tid: str, partition: str = "train") -> dict:
    d = (TRAIN_DIR if partition == "train" else STUDENT_TEST_DIR) / tid
    if d.exists():
        t = load_trip(d)
        t["source"] = "raw"
        return t
    from .eda_recover import recover_trip

    return recover_trip(RESULTS / "viz" / f"{tid}.html")


def analyse(trip: dict) -> dict:
    out_dir = RESULTS / "eda" / f"trip_{trip['trip_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = trip["track"]
    pt = point_table(tr)
    g = gps_summary(tr)
    n, aux = network_analysis(trip["nodes"], trip["arcs"], tr)
    r, align = route_analysis(trip, aux, aux.get("gps_nearest_arc_d")) if "route" in trip \
        else ({}, None)
    summary = {"trip_id": trip["trip_id"], "source": trip.get("source", "raw"),
               "gps": g, "network": n, "route": r}
    if align is not None:
        # how many GPS points snap to each route arc (0 = arc never observed directly)
        hits = np.bincount(align.route_pos[align.route_d <= NEAR_M], minlength=len(trip["route"]))
        summary["route_arc_gps_hits"] = {"zero": int((hits == 0).sum()),
                                         "one": int((hits == 1).sum()),
                                         "median": float(np.median(hits)),
                                         "max": int(hits.max())}
        summary["step_vs_arc"] = {
            "gps_step_median_m": float(np.nanmedian(pt.step_m)),
            "route_arc_len_median_m": float(np.median(aux["arc_len"][trip["route"].arc_id])),
            "frac_steps_longer_than_median_route_arc": float(
                np.nanmean(pt.step_m > np.median(aux["arc_len"][trip["route"].arc_id])))}
    if trip.get("source") == "recovered_from_folium_html":
        summary["caveat"] = ("network stats are PARTIAL (only the map's bbox arcs); timestamps "
                             "rounded to whole seconds; node/arc ids rebuilt from coordinates")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    gap_table(tr, trip["trip_id"], 2).to_csv(out_dir / "gaps.csv", index=False)
    plots(trip, pt, align, aux, out_dir)
    return summary


def plots(trip, pt, align, aux, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tid = trip["trip_id"]
    tr = trip["track"]
    t = tr.t.to_numpy() - tr.t.iloc[0]

    # A) sampling and movement over time
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor=SURF)
    ax[0].plot(t, pt.dt, color=BLUE, lw=1)
    ax[0].set_yscale("log")
    _style(ax[0], f"Trip {tid}: sampling interval dt (s, log)")
    ax[1].plot(t, pt.speed_ms, color=BLUE, lw=1)
    sp = spike_mask(pt)
    ax[1].scatter(t[sp], pt.speed_ms[sp], color=ORANGE, s=20, zorder=3, label="spike")
    _style(ax[1], "Speed between consecutive points (m/s)")
    ax[2].plot(t, np.abs(pt.turn), color=BLUE, lw=0.8)
    _style(ax[2], "Absolute heading change (deg)")
    ax[2].set_xlabel("seconds since trip start", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gps_timeline.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    if align is None:
        return
    # B) GPS -> route distance and alignment (route position of nearest route arc over time)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURF)
    ax[0].hist(align.route_d, bins=40, color=BLUE, edgecolor=SURF)
    if "net_d" in align:
        ax[0].hist(align.net_d, bins=40, histtype="step", color=INK, lw=1.2,
                   label="nearest network arc")
        ax[0].legend(frameon=False, fontsize=8)
    _style(ax[0], "Distance GPS -> nearest route arc (m)")
    ok = align.route_d <= NEAR_M
    ax[1].scatter(np.arange(len(align))[ok], align.route_pos[ok], s=3, color=BLUE)
    ax[1].scatter(np.arange(len(align))[~ok], align.route_pos[~ok], s=6, color=ORANGE,
                  label=f">{NEAR_M:.0f} m from route")
    if (~ok).any():
        ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("GPS observation index", color=INK2, fontsize=8)
    ax[1].set_ylabel("route position", color=INK2, fontsize=8)
    _style(ax[1], "Alignment: nearest route position per GPS point")
    hits = np.bincount(align.route_pos[ok], minlength=len(trip["route"]))
    ax[2].hist(hits, bins=np.arange(hits.max() + 2) - 0.5, color=BLUE, edgecolor=SURF)
    _style(ax[2], "GPS points snapped to each route arc")
    ax[2].set_xlabel("points per arc (0 = arc never observed)", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "route_alignment.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # C) GPS step length vs route-arc length: how many arcs a vehicle crosses per sample
    L = aux["arc_len"][trip["route"].arc_id]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURF)
    bins = np.logspace(0, np.log10(max(np.nanmax(pt.step_m), L.max()) + 1), 40)
    ax.hist(L, bins=bins, color=BLUE, edgecolor=SURF, label="route arc length")
    ax.hist(pt.step_m.dropna(), bins=bins, histtype="step", color=ORANGE, lw=2,
            label="GPS step length")
    ax.set_xscale("log")
    ax.set_xlabel("metres (log)", color=INK2, fontsize=8)
    _style(ax, "Route-arc length vs distance between GPS samples")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "step_vs_arc.png", dpi=130, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "00000001"
    part = sys.argv[2] if len(sys.argv) > 2 else "train"
    print(json.dumps(analyse(get_trip(tid, part)), indent=2, default=float))
