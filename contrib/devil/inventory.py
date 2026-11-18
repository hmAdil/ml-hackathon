#!/usr/bin/env python3
"""PathSnap Phase 2.1: reproducible dataset inventory."""
from pathlib import Path
import argparse
import csv
import json
import statistics
from collections import defaultdict

EXTENSIONS = [".track", ".nodes", ".arcs", ".route"]

def inspect_file(path: Path):
    nonempty = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                nonempty.append(line.strip())
    return len(nonempty), nonempty[:3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help='Root folder containing train/ and student_test/')
    parser.add_argument("--output-dir", default="outputs/phase2_1")
    args = parser.parse_args()

    root = Path(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    trip_rows = []
    partition_summary = {}
    for partition in ["train", "student_test", "evaluator_test"]:
        pdir = root / partition
        if not pdir.exists():
            continue
        trip_dirs = sorted(x for x in pdir.iterdir() if x.is_dir())
        sums = defaultdict(int)
        sizes = defaultdict(int)
        for trip_dir in trip_dirs:
            trip_id = trip_dir.name
            row = {"partition": partition, "trip_id": trip_id}
            for ext in EXTENSIONS:
                matches = list(trip_dir.glob(f"*{ext}"))
                if not matches:
                    row[f"{ext[1:]}_rows"] = None
                    row[f"{ext[1:]}_bytes"] = None
                    continue
                path = matches[0]
                count, sample = inspect_file(path)
                row[f"{ext[1:]}_rows"] = count
                row[f"{ext[1:]}_bytes"] = path.stat().st_size
                row[f"{ext[1:]}_sample"] = " | ".join(sample)
                sums[ext] += count
                sizes[ext] += path.stat().st_size
            row["complete_four_file_set"] = all(row.get(f"{e[1:]}_rows") is not None for e in EXTENSIONS)
            trip_rows.append(row)

        partition_summary[partition] = {
            "trip_directories": len(trip_dirs),
            "row_totals": {e: sums[e] for e in EXTENSIONS},
            "file_bytes": {e: sizes[e] for e in EXTENSIONS},
            "complete_four_file_trips": sum(
                r["complete_four_file_set"] for r in trip_rows
                if r["partition"] == partition
            ),
        }

    csv_path = output / "trip_inventory.csv"
    if trip_rows:
        columns = sorted({k for row in trip_rows for k in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(trip_rows)

    report = {
        "data_root": str(root.resolve()),
        "partitions_found": list(partition_summary),
        "partition_summary": partition_summary,
        "notes": [
            "Row counts count non-empty lines in each headerless file.",
            "The script inventories files; it does not train a model.",
            "Evaluator data, if present, is inventoried only and must not be used for development."
        ]
    }
    json_path = output / "inventory_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()
