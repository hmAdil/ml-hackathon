
from pathlib import Path

ROOT = Path("data/Student folder/train")
TRIP = "00000098"
FOLDER = ROOT / TRIP

def load_file(extension):
    path = FOLDER / f"{TRIP}{extension}"
    with path.open("r", encoding="utf-8") as f:
        return [
            [int(float(x)) for x in line.split()]
            for line in f if line.strip()
        ]

nodes = load_file(".nodes")
arcs = load_file(".arcs")
route = [row[0] for row in load_file(".route")]

print("ROUTE GRAPH VALIDATION")
print("-" * 45)

# Check whether route IDs are valid arc row indices
valid = all(0 <= x < len(arcs) for x in route)

print("All route IDs are valid arc indices:", valid)

if not valid:
    raise ValueError("Some route IDs are outside the arc index range.")

# Retrieve arcs in route order
selected_arcs = [arcs[x] for x in route]

# Check node references
valid_nodes = all(
    0 <= node_id < len(nodes)
    for arc in selected_arcs
    for node_id in arc
)

print("All selected arc endpoints reference nodes:", valid_nodes)

# Check connectivity between consecutive arcs
directed_matches = 0
undirected_matches = 0

for first, second in zip(selected_arcs, selected_arcs[1:]):
    if first[1] == second[0]:
        directed_matches += 1

    if first[1] == second[0] or first[0] == second[1]:
        undirected_matches += 1

total = max(0, len(selected_arcs) - 1)

print("\nROUTE CONNECTIVITY")
print("Consecutive arc pairs:", total)
print("Directed connections:", directed_matches)
print("Undirected connections:", undirected_matches)

if total:
    print("Directed continuity:",
          round(directed_matches / total * 100, 2), "%")
    print("Undirected continuity:",
          round(undirected_matches / total * 100, 2), "%")

print("\nFirst five selected arcs:")
for arc in selected_arcs[:5]:
    print(arc)

print("\nRoute graph validation completed.")
