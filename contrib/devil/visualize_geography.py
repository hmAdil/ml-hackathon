#!/usr/bin/env python3
"""PathSnap Phase 2.5: visualize GPS tracks, road networks, and reference routes."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PARTITIONS = ("train", "student_test")

def read_trip(folder: Path, trip_id: str):
    stem = folder / trip_id / trip_id
    track = pd.read_csv(str(stem) + ".track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes = pd.read_csv(str(stem) + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(str(stem) + ".arcs", sep=r"\s+", header=None, names=["source", "target"], dtype="int64")
    route = pd.read_csv(str(stem) + ".route", sep=r"\s+", header=None, names=["arc_id"], dtype="int64")["arc_id"].to_numpy()
    return track, nodes, arcs, route

def make_plot(track, nodes, arcs, route, output: Path, title: str, max_network_arcs: int, seed: int):
    # Arc IDs in .route refer to row positions in .arcs; .arcs endpoints refer to row positions in .nodes.
    valid = (arcs.source.ge(0) & arcs.source.lt(len(nodes)) & arcs.target.ge(0) & arcs.target.lt(len(nodes)))
    valid_arcs = arcs.loc[valid]
    if len(valid_arcs) > max_network_arcs:
        sampled = valid_arcs.sample(n=max_network_arcs, random_state=seed).sort_index()
    else:
        sampled = valid_arcs

    xy = nodes[["lon", "lat"]].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 9))
    # Background road network, sampled deterministically to keep plotting manageable.
    if len(sampled):
        s = sampled.source.to_numpy(); t = sampled.target.to_numpy()
        segments = np.stack([xy[s], xy[t]], axis=1)
        ax.add_collection(LineCollection(segments, colors="0.72", linewidths=0.35, alpha=0.38, zorder=1))

    route_ok = route[(route >= 0) & (route < len(arcs))]
    if len(route_ok):
        ra = arcs.iloc[route_ok]
        rs = ra.source.to_numpy(); rt = ra.target.to_numpy()
        good = (rs >= 0) & (rs < len(nodes)) & (rt >= 0) & (rt < len(nodes))
        rs, rt = rs[good], rt[good]
        route_segments = np.stack([xy[rs], xy[rt]], axis=1)
        ax.add_collection(LineCollection(route_segments, colors="#e4572e", linewidths=1.7, alpha=0.95, zorder=6))

    ax.plot(track.lon, track.lat, color="#1769aa", linewidth=1.15, alpha=0.9,
            marker=".", markersize=2.0, label="Observed GPS trajectory", zorder=4)
    if len(track):
        ax.scatter(track.lon.iloc[0], track.lat.iloc[0], s=55, marker="o", color="#2ca02c", label="GPS start", zorder=5)
        ax.scatter(track.lon.iloc[-1], track.lat.iloc[-1], s=65, marker="X", color="#9467bd", label="GPS end", zorder=5)
    # Legend proxy for the road layers.
    ax.plot([], [], color="0.55", linewidth=1, label=f"Road network (sampled {len(sampled):,}/{len(arcs):,} arcs)")
    ax.plot([], [], color="#e4572e", linewidth=2, label=f"Reference route ({len(route_ok):,} arcs)")
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)"); ax.set_ylabel("Latitude (degrees)")
    ax.grid(True, alpha=0.2); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    return {"track_points": int(len(track)), "nodes": int(len(nodes)), "arcs": int(len(arcs)),
            "route_arcs": int(len(route)), "valid_route_arc_ids": int(len(route_ok)),
            "sampled_background_arcs": int(len(sampled)), "valid_arc_endpoint_rows": int(valid.sum())}

def find_trip_ids(partition_dir: Path):
    return sorted(p.name for p in partition_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Folder containing train/ and student_test/")
    ap.add_argument("--output-dir", default="outputs/phase2_5")
    ap.add_argument("--partition", choices=["both", *PARTITIONS], default="both")
    ap.add_argument("--trip-id", default=None, help="Optional trip ID; used with --partition train or student_test")
    ap.add_argument("--max-network-arcs", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data_root = Path(args.data_dir); out_root = Path(args.output_dir)
    partitions = PARTITIONS if args.partition == "both" else (args.partition,)
    report = {"note": "Background road-network arcs are sampled for display only. Reference-route arcs are plotted in full when their IDs and endpoints are valid.", "plots": {}}
    for part in partitions:
        pdir = data_root / part
        if not pdir.exists(): raise FileNotFoundError(f"Missing partition folder: {pdir}")
        ids = find_trip_ids(pdir)
        if not ids: raise RuntimeError(f"No trip folders found in {pdir}")
        tid = args.trip_id if args.trip_id and len(partitions) == 1 else ids[0]
        if tid not in ids: raise ValueError(f"Trip {tid} not found in {pdir}; available examples: {ids[:5]}")
        track, nodes, arcs, route = read_trip(pdir, tid)
        output = out_root / f"{part}_trip_{tid}_geographic_overlay.png"
        metrics = make_plot(track, nodes, arcs, route, output,
                            f"PathSnap {part} trip {tid}: GPS, road network, reference route",
                            args.max_network_arcs, args.seed)
        metrics.update({"partition": part, "trip_id": tid, "plot": str(output)})
        report["plots"][f"{part}/{tid}"] = metrics
        print(f"Created: {output}")
        print(f"  GPS points={metrics['track_points']:,}; nodes={metrics['nodes']:,}; arcs={metrics['arcs']:,}; route arcs={metrics['route_arcs']:,}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "geographic_visualization_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSummary: {out_root / 'geographic_visualization_summary.json'}")

if __name__ == "__main__": main()
