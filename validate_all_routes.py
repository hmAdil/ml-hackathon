
from pathlib import Path
import csv
import json

ROOT = Path("data/Student folder/train")
OUTPUT = Path("outputs/phase2_3")

OUTPUT.mkdir(parents=True, exist_ok=True)


def read_rows(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            [int(float(value)) for value in line.split()]
            for line in file
            if line.strip()
        ]


results = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    trip_id = folder.name

    try:
        nodes = read_rows(folder / f"{trip_id}.nodes")
        arcs = read_rows(folder / f"{trip_id}.arcs")
        route_rows = read_rows(folder / f"{trip_id}.route")

        route = [row[0] for row in route_rows]

        route_ids_valid = all(
            0 <= arc_id < len(arcs) for arc_id in route
        )

        all_arc_endpoints_valid = all(
            len(arc) == 2
            and 0 <= arc[0] < len(nodes)
            and 0 <= arc[1] < len(nodes)
            for arc in arcs
        )

        directed_matches = 0
        total_pairs = max(0, len(route) - 1)

        if route_ids_valid:
            selected_arcs = [arcs[i] for i in route]

            for first, second in zip(
                selected_arcs, selected_arcs[1:]
            ):
                if first[1] == second[0]:
                    directed_matches += 1

        continuity = (
            directed_matches / total_pairs * 100
            if total_pairs else 100.0
        )

        passed = (
            route_ids_valid
            and all_arc_endpoints_valid
            and directed_matches == total_pairs
        )

        results.append({
            "trip_id": trip_id,
            "nodes": len(nodes),
            "arcs": len(arcs),
            "route_rows": len(route),
            "route_ids_valid": route_ids_valid,
            "all_arc_endpoints_valid": all_arc_endpoints_valid,
            "directed_matches": directed_matches,
            "consecutive_pairs": total_pairs,
            "directed_continuity_percent": round(continuity, 2),
            "passed": passed,
            "error": ""
        })

    except Exception as error:
        results.append({
            "trip_id": trip_id,
            "passed": False,
            "error": str(error)
        })


csv_path = OUTPUT / "route_validation_all.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

passed_count = sum(row.get("passed", False) for row in results)

summary = {
    "total_trips": len(results),
    "passed_trips": passed_count,
    "failed_trips": len(results) - passed_count,
    "all_trips_passed": len(results) > 0
        and passed_count == len(results)
}

with (OUTPUT / "route_validation_summary.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nALL-TRIP ROUTE VALIDATION")
print("-" * 40)
print("Total trips:", summary["total_trips"])
print("Passed:", summary["passed_trips"])
print("Failed:", summary["failed_trips"])
print("All trips passed:", summary["all_trips_passed"])
print("\nSaved results:", csv_path)
