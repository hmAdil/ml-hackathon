"""Rebuild a trip dict from a Folium map written by pathsnap.viz.trip_map.

Stop-gap for when the raw files are not available locally. What survives in the HTML:
  * every GPS point (lat/lon at 6 dp, timestamp rounded to whole seconds by the tooltip),
  * every reference-route arc, in order (green polylines),
  * at most `max_network_arcs` (default 4000) network arcs inside the track bbox (grey).
So GPS and route analyses are exact up to rounding, but network statistics are PARTIAL and
node/arc IDs are new (rebuilt from unique coordinates), not the original row indices.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

_PT = re.compile(r"L\.circleMarker\(\s*\[([-\d.]+), ([-\d.]+)\]")
_TT = re.compile(r"<div>\s*t=([-\d.]+)\s*</div>")
_PL = re.compile(r"L\.polyline\(\s*\[\[([-\d.]+), ([-\d.]+)\], \[([-\d.]+), ([-\d.]+)\]\],"
                 r"\s*\{[^}]*?\"color\": \"([^\"]+)\"")


def recover_trip(html_path: Path) -> dict:
    html = Path(html_path).read_text()
    pts = np.array(_PT.findall(html), float)
    ts = np.array(_TT.findall(html), float)
    if len(pts) != len(ts):
        raise ValueError(f"{len(pts)} GPS markers but {len(ts)} timestamps")
    track = pd.DataFrame({"lon": pts[:, 1], "lat": pts[:, 0], "t": ts})

    lines = _PL.findall(html)
    seg = np.array([[float(v) for v in m[:4]] for m in lines])
    color = np.array([m[4] for m in lines])
    ends = np.r_[seg[:, [1, 0]], seg[:, [3, 2]]]                  # (lon, lat) of both ends
    uniq, inv = np.unique(np.round(ends, 6), axis=0, return_inverse=True)
    n = len(seg)
    fn, tn = inv[:n], inv[n:]
    nodes = pd.DataFrame({"lon": uniq[:, 0], "lat": uniq[:, 1]})
    pair = pd.DataFrame({"from_node": fn, "to_node": tn})
    arcs = pair.drop_duplicates().reset_index(drop=True)
    arc_id = {(f, t): i for i, (f, t) in enumerate(zip(arcs.from_node, arcs.to_node))}
    is_route = color == "green"
    route = pd.DataFrame({"arc_id": [arc_id[(f, t)] for f, t in zip(fn[is_route], tn[is_route])]})
    return {"trip_id": Path(html_path).stem, "track": track, "nodes": nodes, "arcs": arcs,
            "route": route, "n_network_arcs_in_map": int((color == "#888888").sum()),
            "source": "recovered_from_folium_html"}
