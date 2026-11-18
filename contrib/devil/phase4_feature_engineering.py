#!/usr/bin/env python3
"""PathSnap Phase 4: motion/road feature prototypes and reference-alignment diagnostics.

This script does NOT train a model and does NOT use route information from student_test.
It preserves all source IDs and writes derived tables separately from raw data.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6_371_000.0

FEATURE_DICTIONARY = {
    "gps_observation_features": {
        "dt_prev_s": "Timestamp difference from previous GPS observation; NaN for first row.",
        "distance_prev_m": "Haversine distance from previous GPS observation.",
        "speed_prev_mps": "distance_prev_m / dt_prev_s only when dt_prev_s > 0.",
        "heading_prev_rad": "Initial bearing of previous-to-current GPS displacement, radians.",
        "heading_prev_sin": "Sine of heading_prev_rad; circular encoding.",
        "heading_prev_cos": "Cosine of heading_prev_rad; circular encoding.",
        "heading_change_rad": "Wrapped absolute change between consecutive displacement bearings.",
        "acceleration_prev_mps2": "Difference between consecutive estimated speeds / current dt.",
        "gap_gt_3s": "Flag for dt_prev_s > 3 seconds; exploratory threshold.",
        "gap_gt_5s": "Flag for dt_prev_s > 5 seconds; exploratory threshold.",
        "gap_gt_10s": "Flag for dt_prev_s > 10 seconds; exploratory threshold.",
        "speed_gt_50mps": "Flag for estimated speed > 50 m/s; review, do not delete.",
        "speed_gt_70mps": "Flag for estimated speed > 70 m/s; review, do not delete.",
        "same_position_as_prev": "Flag for zero displacement from previous GPS observation."
    },
    "road_arc_features_for_later_candidate_join": {
        "arc_length_m": "Haversine distance between source and destination node coordinates.",
        "arc_bearing_sin/cos": "Directed source-to-destination bearing encoded circularly.",
        "src_outdegree, src_indegree": "Directed graph degrees of source node.",
        "dst_outdegree, dst_indegree": "Directed graph degrees of destination node.",
        "is_self_loop": "Source node ID equals destination node ID.",
        "duplicate_directed_arc_count": "Number of duplicate rows for the same ordered endpoint pair."
    },
    "candidate_pair_features_for_next_candidate-generation_stage": {
        "point_to_segment_distance_m": "Perpendicular distance from GPS point to candidate road segment.",
        "projection_fraction": "Clamped projection position on candidate segment, [0, 1].",
        "distance_to_source_m/destination_m": "Distances from GPS point to directed candidate endpoints.",
        "heading_alignment_cos": "Cosine of difference between GPS heading and directed arc bearing.",
        "candidate_rank_by_distance": "Rank among spatially retrieved candidate arcs.",
        "previous_candidate_connects": "Whether previous selected candidate's destination equals this source.",
        "next_candidate_connects": "Whether this candidate's destination connects to a next candidate source."
    }
}


def local_xy(lon, lat, lon0, lat0):
    """Trip-local equirectangular projection (metres), adequate for local feature geometry."""
    lon = np.asarray(lon, dtype=float); lat = np.asarray(lat, dtype=float)
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2)])
    dlon, dlat = lon2-lon1, lat2-lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_rad(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2)])
    dlon = lon2-lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return np.arctan2(y, x)


def read_track(path: Path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"], dtype=float)


def read_nodes(path: Path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=float)


def read_arcs(path: Path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["src", "dst"], dtype="int64")


def make_gps_features(track: pd.DataFrame, split: str, trip_id: str):
    d = track.copy().reset_index(drop=True)
    n = len(d)
    dt = d.timestamp.diff().to_numpy()
    dist = np.full(n, np.nan)
    heading = np.full(n, np.nan)
    if n > 1:
        dist[1:] = haversine(d.lon.to_numpy()[:-1], d.lat.to_numpy()[:-1],
                             d.lon.to_numpy()[1:], d.lat.to_numpy()[1:])
        heading[1:] = bearing_rad(d.lon.to_numpy()[:-1], d.lat.to_numpy()[:-1],
                                  d.lon.to_numpy()[1:], d.lat.to_numpy()[1:])
    speed = np.divide(dist, dt, out=np.full(n, np.nan), where=(dt > 0) & np.isfinite(dist))
    heading_change = np.full(n, np.nan)
    if n > 2:
        raw = heading[2:] - heading[1:-1]
        heading_change[2:] = np.abs(np.arctan2(np.sin(raw), np.cos(raw)))
    acceleration = np.full(n, np.nan)
    if n > 2:
        ds = speed[2:] - speed[1:-1]
        acceleration[2:] = np.divide(ds, dt[2:], out=np.full(n-2, np.nan), where=(dt[2:] > 0))
    out = pd.DataFrame({
        "partition": split, "trip_id": trip_id, "gps_index": np.arange(n),
        "lon": d.lon, "lat": d.lat, "timestamp": d.timestamp,
        "dt_prev_s": dt, "distance_prev_m": dist, "speed_prev_mps": speed,
        "heading_prev_rad": heading, "heading_prev_sin": np.sin(heading),
        "heading_prev_cos": np.cos(heading), "heading_change_rad": heading_change,
        "acceleration_prev_mps2": acceleration,
        "gap_gt_3s": (dt > 3).astype("int8"),
        "gap_gt_5s": (dt > 5).astype("int8"),
        "gap_gt_10s": (dt > 10).astype("int8"),
        "speed_gt_50mps": (speed > 50).astype("int8"),
        "speed_gt_70mps": (speed > 70).astype("int8"),
        "same_position_as_prev": (dist == 0).astype("int8"),
        "nonpositive_dt": ((dt <= 0) & np.isfinite(dt)).astype("int8")
    })
    return out


def make_road_summary(nodes, arcs):
    n, m = len(nodes), len(arcs)
    src = arcs.src.to_numpy(); dst = arcs.dst.to_numpy()
    valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    srcv, dstv = src[valid], dst[valid]
    outdeg = np.bincount(srcv, minlength=n); indeg = np.bincount(dstv, minlength=n)
    lengths = haversine(nodes.lon.to_numpy()[srcv], nodes.lat.to_numpy()[srcv],
                        nodes.lon.to_numpy()[dstv], nodes.lat.to_numpy()[dstv]) if valid.any() else np.array([])
    bears = bearing_rad(nodes.lon.to_numpy()[srcv], nodes.lat.to_numpy()[srcv],
                        nodes.lon.to_numpy()[dstv], nodes.lat.to_numpy()[dstv]) if valid.any() else np.array([])
    pairs = pd.Series(list(zip(srcv.tolist(), dstv.tolist())))
    dup_counts = pairs.value_counts()
    dup_rows = int(sum(v-1 for v in dup_counts.values if v > 1))
    return {
        "node_count": n, "arc_count": m, "invalid_arc_endpoint_rows": int((~valid).sum()),
        "zero_outdegree_nodes": int((outdeg == 0).sum()), "zero_indegree_nodes": int((indeg == 0).sum()),
        "median_outdegree": float(np.median(outdeg)) if n else 0.0,
        "max_outdegree": int(outdeg.max()) if n else 0,
        "median_arc_length_m": float(np.median(lengths)) if len(lengths) else None,
        "p95_arc_length_m": float(np.quantile(lengths, .95)) if len(lengths) else None,
        "p99_arc_length_m": float(np.quantile(lengths, .99)) if len(lengths) else None,
        "max_arc_length_m": float(np.max(lengths)) if len(lengths) else None,
        "zero_length_arcs": int(np.sum(lengths == 0)),
        "arcs_over_500m": int(np.sum(lengths > 500)), "arcs_over_1000m": int(np.sum(lengths > 1000)),
        "self_loop_arcs": int(np.sum(srcv == dstv)), "duplicate_directed_arc_rows": dup_rows,
        "mean_arc_bearing_sin": float(np.nanmean(np.sin(bears))) if len(bears) else None,
        "mean_arc_bearing_cos": float(np.nanmean(np.cos(bears))) if len(bears) else None
    }


def route_alignment_diagnostic(track, nodes, arcs, route, sample_spacing_m=15.0):
    """Nearest sampled point on any reference-route arc; diagnostic proxy, not final labels."""
    valid_ids = route[(route >= 0) & (route < len(arcs))]
    selected = arcs.iloc[valid_ids].to_numpy(dtype=int)
    n = len(nodes)
    ok = (selected[:, 0] >= 0) & (selected[:, 0] < n) & (selected[:, 1] >= 0) & (selected[:, 1] < n)
    selected = selected[ok]
    if len(selected) == 0 or len(track) == 0:
        return {"valid_route_arc_ids": int(len(valid_ids)), "route_arc_geometry_count": int(len(selected)),
                "gps_points": int(len(track)), "median_nearest_route_distance_m": None,
                "p95_nearest_route_distance_m": None, "p99_nearest_route_distance_m": None,
                "max_nearest_route_distance_m": None, "nearest_route_order_backward_steps": None,
                "nearest_route_order_backward_step_rate": None}, None
    lon0 = float(track.lon.median()); lat0 = float(track.lat.median())
    gx, gy = local_xy(track.lon.to_numpy(), track.lat.to_numpy(), lon0, lat0)
    sx, sy = local_xy(nodes.lon.to_numpy()[selected[:, 0]], nodes.lat.to_numpy()[selected[:, 0]], lon0, lat0)
    tx, ty = local_xy(nodes.lon.to_numpy()[selected[:, 1]], nodes.lat.to_numpy()[selected[:, 1]], lon0, lat0)
    dx, dy = tx-sx, ty-sy
    lens = np.hypot(dx, dy)
    samples, positions = [], []
    # Sample along each route arc; this is an approximation to segment distance, with <= sample_spacing_m spacing.
    for pos, (x1, y1, x2, y2, length) in enumerate(zip(sx, sy, tx, ty, lens)):
        steps = max(1, int(math.ceil(length / sample_spacing_m)))
        t = np.linspace(0.0, 1.0, steps + 1)
        pts = np.column_stack((x1 + t*dx[pos], y1 + t*dy[pos]))
        samples.append(pts)
        positions.extend([pos] * len(pts))
    sample_xy = np.vstack(samples)
    tree = cKDTree(sample_xy)
    gps_xy = np.column_stack((gx, gy))
    dists, nearest_idx = tree.query(gps_xy, k=1, workers=-1)
    route_pos = np.asarray(positions, dtype=np.int32)[nearest_idx]
    backwards = int(np.sum(np.diff(route_pos) < 0))
    steps_count = max(0, len(route_pos)-1)
    diag = {
        "valid_route_arc_ids": int(len(valid_ids)), "route_arc_geometry_count": int(len(selected)),
        "gps_points": int(len(track)),
        "median_nearest_route_distance_m": float(np.median(dists)),
        "p95_nearest_route_distance_m": float(np.quantile(dists, .95)),
        "p99_nearest_route_distance_m": float(np.quantile(dists, .99)),
        "max_nearest_route_distance_m": float(np.max(dists)),
        "nearest_route_order_backward_steps": backwards,
        "nearest_route_order_backward_step_rate": float(backwards/steps_count) if steps_count else 0.0,
        "sampling_spacing_m": float(sample_spacing_m),
        "warning": "Nearest sampled route point is only a diagnostic proxy. It is not an aligned label and must not be used as an input feature."
    }
    return diag, route_pos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Extracted folder containing train/ and student_test/")
    ap.add_argument("--output-dir", default="outputs/phase4")
    ap.add_argument("--route-sample-spacing-m", type=float, default=15.0)
    ap.add_argument("--include-student-test-gps", action="store_true", help="Create GPS motion features for student_test; do not read its route file.")
    args = ap.parse_args()
    root = Path(args.data_dir); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    gps_outputs, road_rows, align_rows, trip_rows = [], [], [], []
    partitions = ["train"] + (["student_test"] if args.include_student_test_gps else [])
    for split in partitions:
        pdir = root / split
        if not pdir.exists():
            print(f"[WARN] Missing partition: {pdir}"); continue
        trip_dirs = sorted(p for p in pdir.iterdir() if p.is_dir())
        for td in trip_dirs:
            tid = td.name; stem = td / tid
            required = [Path(str(stem)+s) for s in (".track", ".nodes", ".arcs")]
            if not all(p.exists() for p in required):
                print(f"[WARN] Skipping incomplete trip {split}/{tid}"); continue
            track = read_track(Path(str(stem)+".track"))
            nodes = read_nodes(Path(str(stem)+".nodes"))
            arcs = read_arcs(Path(str(stem)+".arcs"))
            gpsf = make_gps_features(track, split, tid)
            gps_outputs.append(gpsf)
            road = make_road_summary(nodes, arcs); road.update({"partition": split, "trip_id": tid})
            road_rows.append(road)
            trip_rows.append({"partition": split, "trip_id": tid, "gps_rows": len(track), **{k:v for k,v in road.items() if k not in ("partition", "trip_id")}})
            # Reference route is deliberately read ONLY from train.
            route_path = Path(str(stem)+".route")
            if split == "train" and route_path.exists():
                route = np.loadtxt(route_path, dtype=np.int64, ndmin=1).reshape(-1)
                diag, _ = route_alignment_diagnostic(track, nodes, arcs, route, args.route_sample_spacing_m)
                diag.update({"partition": split, "trip_id": tid, "route_arc_count": int(len(route))})
                align_rows.append(diag)
            print(f"[OK] {split}/{tid}: GPS={len(track):,}, nodes={len(nodes):,}, arcs={len(arcs):,}")
    if not gps_outputs:
        raise SystemExit("No trip data processed. Check --data-dir and folder layout.")
    pd.concat(gps_outputs, ignore_index=True).to_csv(out/"gps_motion_features.csv.gz", index=False, compression="gzip")
    pd.DataFrame(road_rows).to_csv(out/"road_network_feature_summary.csv", index=False)
    pd.DataFrame(trip_rows).to_csv(out/"trip_feature_summary.csv", index=False)
    if align_rows:
        pd.DataFrame(align_rows).to_csv(out/"reference_alignment_diagnostics_train_only.csv", index=False)
    (out/"feature_dictionary.json").write_text(json.dumps(FEATURE_DICTIONARY, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 4 feature prototypes to: {out.resolve()}")
    print("Note: no ML fitting, no route-derived input predictors, no automatic deletion, no student-test route access.")

if __name__ == "__main__":
    main()
