
from pathlib import Path

ROOT = Path("data/Student folder/train")
TRIP = "00000098"
FOLDER = ROOT / TRIP

def read_data(extension):
    path = FOLDER / f"{TRIP}{extension}"
    with path.open("r", encoding="utf-8") as f:
        return [
            [float(x) for x in line.split()]
            for line in f if line.strip()
        ]

track = read_data(".track")
nodes = read_data(".nodes")
arcs = read_data(".arcs")
route = read_data(".route")

print("TRIP:", TRIP)
print("-" * 45)

print("Track rows:", len(track))
print("Node rows:", len(nodes))
print("Arc rows:", len(arcs))
print("Route rows:", len(route))

# Examine the third track column
sequence = [row[2] for row in track]
strictly_increasing = all(
    sequence[i] < sequence[i + 1]
    for i in range(len(sequence) - 1)
)

print("\nTRACK SEQUENCE")
print("Minimum:", min(sequence))
print("Maximum:", max(sequence))
print("Unique values:", len(set(sequence)))
print("Strictly increasing:", strictly_increasing)

# Examine arc identifiers
arc_col_0 = {int(row[0]) for row in arcs}
arc_col_1 = {int(row[1]) for row in arcs}
route_ids = {int(row[0]) for row in route}

print("\nARC IDENTIFIERS")
for i in range(2):
    values = [int(row[i]) for row in arcs]
    print(
        f"Column {i + 1}: min={min(values)}, "
        f"max={max(values)}, unique={len(set(values))}"
    )

# Check possible route identifier relationships
print("\nROUTE RELATIONSHIPS")
print("Unique route IDs:", len(route_ids))
print("Route IDs within node index range:",
      sum(0 <= x < len(nodes) for x in route_ids))
print("Route IDs within arc row index range:",
      sum(0 <= x < len(arcs) for x in route_ids))
print("Route IDs found in arc column 1:",
      len(route_ids & arc_col_0))
print("Route IDs found in arc column 2:",
      len(route_ids & arc_col_1))

print("\nCoordinate ranges")
for name, data in [("Track", track), ("Nodes", nodes)]:
    print(
        name,
        "Column 1:", min(r[0] for r in data),
        "to", max(r[0] for r in data),
        "| Column 2:", min(r[1] for r in data),
        "to", max(r[1] for r in data)
    )

print("\nRelationship inspection completed.")
