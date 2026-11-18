#!/usr/bin/env python3
"""PathSnap Phase 5: candidate-radius diagnostics and augmentation-readiness report.

This stage DOES NOT fabricate labels or alter raw GPS/network files. It reads the
Phase 4 per-trip route-proximity diagnostic CSV and reports conservative
candidate-search-radius evidence plus trips requiring alignment review.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    "partition", "trip_id", "gps_points", "median_nearest_route_distance_m",
    "p95_nearest_route_distance_m", "p99_nearest_route_distance_m",
    "nearest_route_order_backward_step_rate", "valid_route_arc_ids",
    "route_arc_geometry_count"
}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics", required=True, help="Phase 4 reference_alignment_diagnostics_train_only.csv")
    ap.add_argument("--output-dir", default="outputs/phase5")
    ap.add_argument("--radii-m", default="10,15,20,30,50,75,100,150,250",
                    help="Comma-separated radius sweep in metres")
    ap.add_argument("--median-review-m", type=float, default=25.0)
    ap.add_argument("--p95-review-m", type=float, default=50.0)
    ap.add_argument("--backward-review-rate", type=float, default=0.05)
    args = ap.parse_args()

    src = Path(args.diagnostics)
    if not src.is_file():
        raise FileNotFoundError(f"Diagnostics file not found: {src}")
    df = pd.read_csv(src)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Diagnostics CSV missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Diagnostics CSV contains no trip rows.")

    numeric = [
        "gps_points", "median_nearest_route_distance_m", "p95_nearest_route_distance_m",
        "p99_nearest_route_distance_m", "nearest_route_order_backward_step_rate",
        "valid_route_arc_ids", "route_arc_geometry_count"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric].isna().any().any():
        bad = {c: int(df[c].isna().sum()) for c in numeric if df[c].isna().any()}
        raise ValueError(f"Missing/non-numeric values in required diagnostic columns: {bad}")

    radii = sorted(set(float(x.strip()) for x in args.radii_m.split(",") if x.strip()))
    if not radii or any(x <= 0 for x in radii):
        raise ValueError("--radii-m must contain positive metre values.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Each row describes the nearest sampled reference-route geometry for a trip.
    # p95 <= R implies at least approximately 95% of that trip's GPS points were
    # within R of that sampled geometry (sampling is at a finite interval).
    sweep = []
    for radius in radii:
        sweep.append({
            "radius_m": radius,
            "trips_with_median_distance_le_radius": int((df.median_nearest_route_distance_m <= radius).sum()),
            "trips_with_p95_distance_le_radius": int((df.p95_nearest_route_distance_m <= radius).sum()),
            "trips_with_p99_distance_le_radius": int((df.p99_nearest_route_distance_m <= radius).sum()),
            "total_trips": int(len(df)),
            "pct_trips_p95_le_radius": float(100 * (df.p95_nearest_route_distance_m <= radius).mean())
        })
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out / "candidate_radius_sweep.csv", index=False)

    review = df.copy()
    review["review_high_median_distance"] = review.median_nearest_route_distance_m > args.median_review_m
    review["review_high_p95_distance"] = review.p95_nearest_route_distance_m > args.p95_review_m
    review["review_non_monotonic_nearest_route_order"] = (
        review.nearest_route_order_backward_step_rate > args.backward_review_rate
    )
    review["review_route_geometry_count_mismatch"] = (
        review.valid_route_arc_ids != review.route_arc_geometry_count
    )
    review["review_required"] = review[[
        "review_high_median_distance", "review_high_p95_distance",
        "review_non_monotonic_nearest_route_order", "review_route_geometry_count_mismatch"
    ]].any(axis=1)
    review.to_csv(out / "alignment_review_trips.csv", index=False)

    # Recommend experiments, not automatic acceptance thresholds.
    summary = {
        "source_file": str(src),
        "partition": str(df.partition.iloc[0]) if df.partition.nunique() == 1 else "mixed",
        "trip_count": int(len(df)),
        "total_gps_points": int(df.gps_points.sum()),
        "route_id_geometry_matches": int((df.valid_route_arc_ids == df.route_arc_geometry_count).sum()),
        "median_of_trip_median_distance_m": float(df.median_nearest_route_distance_m.median()),
        "median_of_trip_p95_distance_m": float(df.p95_nearest_route_distance_m.median()),
        "median_of_trip_p99_distance_m": float(df.p99_nearest_route_distance_m.median()),
        "trips_median_distance_over_review_threshold": int((review.review_high_median_distance).sum()),
        "trips_p95_distance_over_review_threshold": int((review.review_high_p95_distance).sum()),
        "trips_backward_order_rate_over_threshold": int((review.review_non_monotonic_nearest_route_order).sum()),
        "trips_requiring_alignment_review": int(review.review_required.sum()),
        "candidate_radius_sweep": sweep,
        "interpretation_warning": (
            "These are per-trip distances to sampled reference-route geometry, not "
            "per-GPS aligned labels and not measured candidate recall. Do not use "
            "reference-route distances or route-derived positions as model input features."
        ),
        "recommended_experiments": {
            "candidate_generation": [
                "Benchmark 20m, 30m, 50m and 100m search radii using the actual road-segment spatial index.",
                "Use a staged fallback radius rather than one very large global radius.",
                "Measure true-route-arc candidate recall only after GPS-to-route label alignment is validated.",
                "Track candidate count per GPS point and runtime alongside recall."
            ],
            "hard_negative_sampling": [
                "After alignment, retain the positive route candidate and sample nearby competing arcs.",
                "Compare 1:1, 1:3, 1:5 and 1:10 negative-to-positive ratios.",
                "Prioritize close parallel roads, opposite-direction arcs and connected alternatives.",
                "Do not sample negatives that are part of the valid reference route at that observation."
            ],
            "gps_perturbation": [
                "After alignment is trusted, test isotropic coordinate jitter with sigma 2m, 5m and 10m on training trips only.",
                "Rebuild candidates after every perturbation.",
                "Keep an augmented sample only when its aligned positive arc remains in the candidate set.",
                "Never perturb validation or student-test data."
            ],
            "not_recommended_initially": [
                "Do not apply SMOTE to mixed geometric/graph candidate features before a validated candidate-label table exists.",
                "Do not use MixUp on longitude/latitude or graph identifiers; interpolated coordinates can fall between roads.",
                "Do not synthesize routes by interpolating route arc IDs.",
                "Do not augment validation or test data."
            ]
        }
    }
    (out / "phase5_readiness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# PathSnap Phase 5 – Data Generation & Augmentation Readiness",
        "",
        f"- Trips analyzed: {summary['trip_count']}",
        f"- GPS points represented: {summary['total_gps_points']:,}",
        f"- Median of per-trip median route distances: {summary['median_of_trip_median_distance_m']:.2f} m",
        f"- Median of per-trip 95th-percentile route distances: {summary['median_of_trip_p95_distance_m']:.2f} m",
        f"- Trips flagged for alignment review: {summary['trips_requiring_alignment_review']}",
        "",
        "## Candidate-radius diagnostic",
        "",
        "| Radius (m) | Trips with median <= R | Trips with p95 <= R | Trips with p99 <= R |",
        "|---:|---:|---:|---:|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['radius_m']:g} | {row['trips_with_median_distance_le_radius']} | "
            f"{row['trips_with_p95_distance_le_radius']} | {row['trips_with_p99_distance_le_radius']} |"
        )
    lines += [
        "",
        "A trip with p95 distance <= R has at least approximately 95% of its GPS points within R "
        "of the sampled reference-route geometry. This is not the same as candidate recall.",
        "",
        "## Required next experiments",
        "",
        "1. Build a spatial index over road segments and measure candidate counts/runtime.",
        "2. Validate GPS-to-route label alignment and report positive-candidate recall.",
        "3. Only then create hard negatives and train-only GPS-jitter variants.",
        "",
        "## Guardrails",
        "",
        "- Do not run SMOTE on raw coordinates, node IDs, arc IDs or unvalidated candidate rows.",
        "- Do not interpolate graph identifiers or generate arbitrary synthetic routes.",
        "- Do not augment validation or student-test data.",
        "- Keep original observations and stable trip-local IDs.",
        "- Report baseline versus augmented validation performance by held-out trip.",
        "",
        "This report uses only Phase 4 diagnostics and creates no synthetic training rows."
    ]
    (out / "phase5_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "trips": summary["trip_count"],
        "gps_points": summary["total_gps_points"],
        "median_trip_median_distance_m": round(summary["median_of_trip_median_distance_m"], 3),
        "median_trip_p95_distance_m": round(summary["median_of_trip_p95_distance_m"], 3),
        "trips_requiring_alignment_review": summary["trips_requiring_alignment_review"],
        "output_dir": str(out.resolve())
    }, indent=2))

if __name__ == "__main__":
    main()
