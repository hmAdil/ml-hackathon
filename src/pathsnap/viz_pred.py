"""Folium comparison map: GPS, reference route, classifier-only and HMM predicted routes."""
from pathlib import Path

import folium
import numpy as np

from .viz import GPS_C, ROUTE_C, _base_map

CLF_C, HMM_C = "#e0a800", "#e34948"


def compare_map(trip: dict, ref_arcs, clf_pred, hmm_pred, out: Path):
    """ref_arcs: arc ids; clf_pred/hmm_pred: DataFrame[obs_idx, arc_id] (canonical ids)."""
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    m = _base_map([tr.lat.mean(), tr.lon.mean()], 14)

    def draw(name, ids, color, weight, dash=None):
        g = folium.FeatureGroup(name).add_to(m)
        for i in ids:
            folium.PolyLine([(lat[fn[i]], lon[fn[i]]), (lat[tn[i]], lon[tn[i]])], color=color,
                            weight=weight, dash_array=dash).add_to(g)

    g = folium.FeatureGroup("GPS observations (blue)").add_to(m)
    for k in range(0, len(tr), max(1, len(tr) // 1500)):
        folium.CircleMarker((tr.lat[k], tr.lon[k]), radius=2, color=GPS_C, fill=True,
                            tooltip=f"obs {k}").add_to(g)
    draw("Reference route (green)", np.unique(ref_arcs), ROUTE_C, 7)
    draw("Classifier-only top-1 arcs (amber)", np.unique(clf_pred.arc_id), CLF_C, 4, dash="6")
    draw("HMM route (red)", np.unique(hmm_pred.arc_id), HMM_C, 3)
    folium.LayerControl().add_to(m)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
