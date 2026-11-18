
from pathlib import Path
from collections import Counter
import json
import math

DATA_ROOT = Path("data/Student folder/train")
OUTPUT = Path("outputs/phase2_2/schema_summary.json")

EXTENSIONS = [".track", ".nodes", ".arcs", ".route"]

if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Training folder not found: {DATA_ROOT}")

summary = {}

for ext in EXTENSIONS:
    column_counts = Counter()
    minimum = []
    maximum = []
    examples = []
    total_rows = 0
    bad_rows = 0
    files_found = 0

    for path in sorted(DATA_ROOT.rglob(f"*{ext}")):
        files_found += 1

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                values = line.split()

                if not values:
                    continue

                total_rows += 1
                column_counts[len(values)] += 1

                try:
                    numbers = [float(value) for value in values]

                    if not all(math.isfinite(x) for x in numbers):
                        raise ValueError("Non-finite value")

                except ValueError:
                    bad_rows += 1
                    continue

                while len(minimum) < len(numbers):
                    minimum.append(float("inf"))
                    maximum.append(float("-inf"))

                for i, number in enumerate(numbers):
                    minimum[i] = min(minimum[i], number)
                    maximum[i] = max(maximum[i], number)

                if len(examples) < 3:
                    examples.append({
                        "file": str(path.relative_to(DATA_ROOT)),
                        "line": line_number,
                        "values": numbers
                    })

    summary[ext] = {
        "files_found": files_found,
        "total_rows": total_rows,
        "column_counts": dict(column_counts),
        "minimum": minimum,
        "maximum": maximum,
        "invalid_numeric_rows": bad_rows,
        "examples": examples
    }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\nSchema report saved to: {OUTPUT}")
