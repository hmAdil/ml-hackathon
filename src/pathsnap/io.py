"""Read raw trip files (whitespace separated, no headers)."""
from pathlib import Path

import pandas as pd

COLS = {
    "track": ["lon", "lat", "t"],
    "nodes": ["lon", "lat"],
    "arcs": ["from_node", "to_node"],
    "route": ["arc_id"],
}


def list_trips(partition_dir: Path) -> list[Path]:
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())


def load_trip(trip_dir: Path, with_route: bool = True) -> dict:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name

    def rd(ext):
        return pd.read_csv(trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, names=COLS[ext])

    trip = {"trip_id": tid, "track": rd("track"), "nodes": rd("nodes"), "arcs": rd("arcs")}
    if with_route and (trip_dir / f"{tid}.route").exists():
        trip["route"] = rd("route")
    return trip
