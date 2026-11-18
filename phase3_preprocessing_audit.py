#!/usr/bin/env python3
"""PathSnap Phase 3: non-destructive data quality audit and preprocessing flags.

Reads .track/.nodes/.arcs/.route files and writes audit reports. It never edits
or removes source records. Node/arc/route IDs remain local to each trip.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EARTH_R_M = 6_371_000.0
PARTITIONS = ("train", "student_test")


def read_numeric(path: Path, dtype=float) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return np.asarray(arr)


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def safe_quantile(values, q):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else None


def audit_trip(trip_dir: Path, partition: str, gap_s: float,
               speed_warn: float, speed_severe: float,
               edge_warn: float, edge_severe: float):
    tid = trip_dir.name
    stem = trip_dir / tid
    paths = {ext: Path(str(stem) + f".{ext}") for ext in ("track", "nodes", "arcs", "route")}
    missing = [ext for ext, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{partition}/{tid}: missing {missing}")

    track = read_numeric(paths["track"], float)
    nodes = read_numeric(paths["nodes"], float)
    arcs = read_numeric(paths["arcs"], np.int64)
    route = np.loadtxt(paths["route"], dtype=np.int64, ndmin=1).reshape(-1)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(f"{partition}/{tid}: expected .track with 3 columns, got {track.shape}")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"{partition}/{tid}: expected .nodes with 2 columns, got {nodes.shape}")
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError(f"{partition}/{tid}: expected .arcs with 2 columns, got {arcs.shape}")

    # GPS checks. Keep source row order and original values.
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    coord_finite = np.isfinite(lon) & np.isfinite(lat)
    timestamp_finite = np.isfinite(ts)
    coord_range_ok = coord_finite & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
    gps_rows = []
    valid_triplets = coord_range_ok & timestamp_finite
    valid_indices = np.flatnonzero(valid_triplets)
    dt = np.diff(ts)
    step_dist = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]) if len(track) > 1 else np.array([])
    valid_step = (coord_range_ok[:-1] & coord_range_ok[1:] & timestamp_finite[:-1] & timestamp_finite[1:]) if len(track) > 1 else np.array([], dtype=bool)
    dt_valid = (dt > 0) & valid_step if len(dt) else np.array([], dtype=bool)
    speed = np.full(len(dt), np.nan, dtype=float)
    if len(dt):
        speed[dt_valid] = step_dist[dt_valid] / dt[dt_valid]

    long_gaps = np.flatnonzero((dt > gap_s) & valid_step) if len(dt) else np.array([], dtype=int)
    bad_time = np.flatnonzero((dt <= 0) & valid_step) if len(dt) else np.array([], dtype=int)
    warn_speed = np.flatnonzero(speed > speed_warn) if len(speed) else np.array([], dtype=int)
    severe_speed = np.flatnonzero(speed > speed_severe) if len(speed) else np.array([], dtype=int)
    flag_idx = sorted(set(long_gaps.tolist() + bad_time.tolist() + warn_speed.tolist() + severe_speed.tolist()))
    for i in flag_idx:
        flags = []
        if i in long_gaps: flags.append("long_gap_after_previous_point")
        if i in bad_time: flags.append("nonpositive_time_delta")
        if i in warn_speed: flags.append("speed_warning")
        if i in severe_speed: flags.append("speed_severe")
        gps_rows.append({
            "partition": partition, "trip_id": tid,
            "previous_row_index": int(i), "current_row_index": int(i + 1),
            "previous_timestamp": float(ts[i]), "current_timestamp": float(ts[i + 1]),
            "delta_t_seconds": float(dt[i]),
            "step_distance_m": float(step_dist[i]) if np.isfinite(step_dist[i]) else None,
            "estimated_speed_mps": float(speed[i]) if np.isfinite(speed[i]) else None,
            "flags": ";".join(flags)
        })

    # Road graph checks. Do not deduplicate: IDs are row-position based.
    node_finite = np.isfinite(nodes[:, :2]).all(axis=1)
    node_range_ok = node_finite & (nodes[:, 0] >= -180) & (nodes[:, 0] <= 180) & (nodes[:, 1] >= -90) & (nodes[:, 1] <= 90)
    endpoint_ok = (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes)) & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    endpoint_coords_ok = np.zeros(len(arcs), dtype=bool)
    valid_arc_rows = np.flatnonzero(endpoint_ok)
    if len(valid_arc_rows):
        endpoint_coords_ok[valid_arc_rows] = node_range_ok[arcs[valid_arc_rows, 0]] & node_range_ok[arcs[valid_arc_rows, 1]]
    edge_length = np.full(len(arcs), np.nan, dtype=float)
    if len(valid_arc_rows):
        s, t = arcs[valid_arc_rows, 0], arcs[valid_arc_rows, 1]
        edge_length[valid_arc_rows] = haversine_m(nodes[s, 0], nodes[s, 1], nodes[t, 0], nodes[t, 1])
    duplicated_arc = pd.DataFrame(arcs, columns=["from", "to"]).duplicated(keep="first").to_numpy()
    self_loop = arcs[:, 0] == arcs[:, 1]
    duplicate_node_coord_rows = len(nodes) - len(np.unique(nodes, axis=0))

    # Route checks. A route value indexes a row in .arcs, not a global arc class.
    route_id_ok = (route >= 0) & (route < len(arcs))
    route_transitions = max(len(route) - 1, 0)
    broken_route_rows = []
    connected_count = 0
    for i in range(route_transitions):
        if route_id_ok[i] and route_id_ok[i + 1]:
            a0 = arcs[route[i]]
            a1 = arcs[route[i + 1]]
            connected = int(a0[1]) == int(a1[0])
            connected_count += int(connected)
            if not connected:
                broken_route_rows.append({
                    "partition": partition, "trip_id": tid,
                    "route_transition_index": i,
                    "previous_route_position": i, "previous_arc_row_id": int(route[i]),
                    "previous_to_node": int(a0[1]),
                    "next_route_position": i + 1, "next_arc_row_id": int(route[i + 1]),
                    "next_from_node": int(a1[0]),
                    "issue": "consecutive_reference_arcs_not_directly_connected"
                })
        else:
            broken_route_rows.append({
                "partition": partition, "trip_id": tid,
                "route_transition_index": i,
                "previous_route_position": i,
                "previous_arc_row_id": int(route[i]),
                "previous_to_node": None,
                "next_route_position": i + 1,
                "next_arc_row_id": int(route[i + 1]),
                "next_from_node": None,
                "issue": "invalid_route_arc_id"
            })
    valid_route_arcs = arcs[route[route_id_ok]] if route_id_ok.any() else np.empty((0, 2), dtype=np.int64)
    route_arc_lengths = np.full(len(valid_route_arcs), np.nan, dtype=float)
    route_endpoint_ok = ((valid_route_arcs[:, 0] >= 0) & (valid_route_arcs[:, 0] < len(nodes)) &
                         (valid_route_arcs[:, 1] >= 0) & (valid_route_arcs[:, 1] < len(nodes))) if len(valid_route_arcs) else np.array([], bool)
    if route_endpoint_ok.any():
        rr = valid_route_arcs[route_endpoint_ok]
        route_arc_lengths[route_endpoint_ok] = haversine_m(nodes[rr[:, 0], 0], nodes[rr[:, 0], 1], nodes[rr[:, 1], 0], nodes[rr[:, 1], 1])

    gps_summary = {
        "gps_rows": int(len(track)),
        "nonfinite_coordinate_rows": int((~coord_finite).sum()),
        "out_of_range_coordinate_rows": int((coord_finite & ~((lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90))).sum()),
        "nonfinite_timestamp_rows": int((~timestamp_finite).sum()),
        "nonpositive_time_intervals": int((dt <= 0).sum()) if len(dt) else 0,
        "gaps_over_threshold": int(len(long_gaps)),
        "speed_over_warning": int(len(warn_speed)),
        "speed_over_severe": int(len(severe_speed)),
        "median_positive_interval_s": safe_quantile(dt[dt > 0], 0.5) if len(dt) else None,
        "p99_positive_interval_s": safe_quantile(dt[dt > 0], 0.99) if len(dt) else None,
        "max_positive_interval_s": float(np.max(dt[dt > 0])) if len(dt[dt > 0]) else None,
        "median_speed_mps": safe_quantile(speed, 0.5),
        "p95_speed_mps": safe_quantile(speed, 0.95),
        "p99_speed_mps": safe_quantile(speed, 0.99),
        "max_speed_mps": float(np.nanmax(speed)) if np.isfinite(speed).any() else None,
    }
    network_summary = {
        "node_rows": int(len(nodes)), "invalid_node_coordinate_rows": int((~node_range_ok).sum()),
        "arc_rows": int(len(arcs)), "invalid_arc_endpoint_rows": int((~endpoint_ok).sum()),
        "arcs_with_invalid_endpoint_coordinates": int((endpoint_ok & ~endpoint_coords_ok).sum()),
        "duplicate_directed_arc_rows_after_first": int(duplicated_arc.sum()),
        "self_loop_arc_rows": int(self_loop.sum()),
        "duplicate_node_coordinate_rows_after_first": int(duplicate_node_coord_rows),
        "zero_length_arc_rows": int(np.sum(edge_length == 0)),
        "arcs_over_warning_length": int(np.sum(edge_length > edge_warn)),
        "arcs_over_severe_length": int(np.sum(edge_length > edge_severe)),
        "median_arc_length_m": safe_quantile(edge_length, 0.5),
        "p95_arc_length_m": safe_quantile(edge_length, 0.95),
        "p99_arc_length_m": safe_quantile(edge_length, 0.99),
        "max_arc_length_m": float(np.nanmax(edge_length)) if np.isfinite(edge_length).any() else None,
    }
    route_summary = {
        "route_arc_rows": int(len(route)), "invalid_route_arc_ids": int((~route_id_ok).sum()),
        "valid_route_arc_ids": int(route_id_ok.sum()),
        "repeated_route_arc_occurrences": int(route_id_ok.sum() - len(np.unique(route[route_id_ok]))),
        "route_transitions": int(route_transitions),
        "connected_route_transitions": int(connected_count),
        "broken_or_invalid_route_transitions": int(len(broken_route_rows)),
        "valid_selected_route_edge_length_sum_m": float(np.nansum(route_arc_lengths)),
    }
    summary = {"partition": partition, "trip_id": tid, **gps_summary, **network_summary, **route_summary}
    return summary, gps_rows, broken_route_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Extracted folder containing train/ and student_test/")
    p.add_argument("--output-dir", default="outputs/phase3")
    p.add_argument("--gap-seconds", type=float, default=5.0,
                   help="Flag intervals above this threshold; does not delete or interpolate points")
    p.add_argument("--speed-warning-mps", type=float, default=50.0)
    p.add_argument("--speed-severe-mps", type=float, default=70.0)
    p.add_argument("--edge-warning-m", type=float, default=500.0)
    p.add_argument("--edge-severe-m", type=float, default=1000.0)
    args = p.parse_args()
    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trip_summaries, gps_flags, route_issues = [], [], []
    for part in PARTITIONS:
        folder = root / part
        if not folder.exists():
            print(f"[SKIP] Partition not found: {folder}")
            continue
        trip_dirs = sorted(x for x in folder.iterdir() if x.is_dir())
        for td in trip_dirs:
            try:
                s, gf, ri = audit_trip(td, part, args.gap_seconds,
                                       args.speed_warning_mps, args.speed_severe_mps,
                                       args.edge_warning_m, args.edge_severe_m)
                trip_summaries.append(s); gps_flags.extend(gf); route_issues.extend(ri)
                print(f"[OK] {part}/{td.name}: GPS={s['gps_rows']:,}, nodes={s['node_rows']:,}, arcs={s['arc_rows']:,}, route={s['route_arc_rows']:,}")
            except Exception as e:
                trip_summaries.append({"partition": part, "trip_id": td.name, "audit_error": str(e)})
                print(f"[ERROR] {part}/{td.name}: {e}")
    if not trip_summaries:
        raise SystemExit(f"No trips found under {root}. Check --data-dir.")

    sdf = pd.DataFrame(trip_summaries)
    sdf.to_csv(out / "trip_preprocessing_audit.csv", index=False)
    pd.DataFrame(gps_flags, columns=["partition", "trip_id", "previous_row_index", "current_row_index", "previous_timestamp", "current_timestamp", "delta_t_seconds", "step_distance_m", "estimated_speed_mps", "flags"]).to_csv(out / "gps_flagged_steps.csv", index=False)
    pd.DataFrame(route_issues, columns=["partition", "trip_id", "route_transition_index", "previous_route_position", "previous_arc_row_id", "previous_to_node", "next_route_position", "next_arc_row_id", "next_from_node", "issue"]).to_csv(out / "route_connectivity_issues.csv", index=False)

    good = sdf[~sdf.get("audit_error", pd.Series(index=sdf.index, dtype=object)).notna()] if "audit_error" in sdf.columns else sdf
    summary = {
        "data_root": str(root.resolve()),
        "trips_audited": int(len(good)),
        "trips_with_errors": int(len(sdf) - len(good)),
        "partition_trip_counts": {str(k): int(v) for k, v in good.groupby("partition").size().to_dict().items()},
        "totals": {},
        "thresholds_used": {
            "gps_gap_seconds": args.gap_seconds,
            "speed_warning_mps": args.speed_warning_mps,
            "speed_severe_mps": args.speed_severe_mps,
            "edge_warning_m": args.edge_warning_m,
            "edge_severe_m": args.edge_severe_m,
        },
        "policy": [
            "Source files are read-only; this audit does not delete, reorder, interpolate, or deduplicate observations.",
            "Node IDs, arc IDs, and route arc IDs are row-position references local to each trip; preserve those indices.",
            "Reference routes are labels/evaluation targets, never predictor features.",
            "Use trip-level grouped validation; never randomly split GPS points from the same trip across folds.",
            "Fit imputers, scalers, encoders, and feature selection only inside each training fold.",
            "Use flags for unusual GPS speeds, GPS gaps, long edges, duplicate records, and disconnected route transitions; review before excluding anything.",
            "Do not apply SMOTE to synthetic road-candidate feature rows unless a graph-consistent augmentation method is separately justified."
        ]
    }
    count_cols = ["gps_rows", "node_rows", "arc_rows", "route_arc_rows", "invalid_route_arc_ids", "broken_or_invalid_route_transitions", "gaps_over_threshold", "speed_over_warning", "speed_over_severe", "invalid_arc_endpoint_rows", "duplicate_directed_arc_rows_after_first", "self_loop_arc_rows", "duplicate_node_coordinate_rows_after_first"]
    for c in count_cols:
        if c in good.columns:
            summary["totals"][c] = int(pd.to_numeric(good[c], errors="coerce").fillna(0).sum())
    (out / "preprocessing_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PHASE 3 AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved to: {out.resolve()}")

if __name__ == "__main__":
    main()
