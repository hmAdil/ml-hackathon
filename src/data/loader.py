"""Read raw trip files. Format: whitespace-separated, no headers.
.track: lon lat t | .nodes: lon lat | .arcs: from_node to_node | .route: arc_id
"""
from pathlib import Path
import pandas as pd


def load_trip(trip_dir: Path, with_route: bool = True) -> dict:
    tid = trip_dir.name
    rd = lambda ext, cols: pd.read_csv(
        trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, names=cols)
    out = {
        "trip_id": tid,
        "track": rd("track", ["lon", "lat", "t"]),
        "nodes": rd("nodes", ["lon", "lat"]),
        "arcs": rd("arcs", ["from_node", "to_node"]),
    }
    if with_route and (trip_dir / f"{tid}.route").exists():
        out["route"] = rd("route", ["arc_id"])
    return out
