"""Folium maps: GPS points, nearby road network, reference route, optional predicted route."""
from pathlib import Path

import folium

from .config import RESULTS


def trip_map(trip: dict, pred_arcs=None, out: Path | None = None, max_network_arcs: int = 4000):
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    # OSM's public tile servers return 403 for pages opened from file://, so CartoDB is the default
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=16, tiles="CartoDB positron")
    folium.TileLayer("CartoDB dark_matter", name="Dark basemap", show=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (blocked from file://)", show=False).add_to(m)
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()

    def line(i):
        return [(lat[fn[i]], lon[fn[i]]), (lat[tn[i]], lon[tn[i]])]

    pad = 0.002
    inb = ((lon[fn] > tr.lon.min() - pad) & (lon[fn] < tr.lon.max() + pad)
           & (lat[fn] > tr.lat.min() - pad) & (lat[fn] < tr.lat.max() + pad))
    net = folium.FeatureGroup("Road network (grey)", show=False).add_to(m)
    for i in arcs.index[inb][:max_network_arcs]:
        folium.PolyLine(line(i), color="#888888", weight=1).add_to(net)

    gps = folium.FeatureGroup("GPS observations (blue)").add_to(m)
    for r in tr.itertuples():
        folium.CircleMarker((r.lat, r.lon), radius=2, color="blue", fill=True,
                            tooltip=f"t={r.t:.0f}").add_to(gps)
    if "route" in trip:
        ref = folium.FeatureGroup("Reference route (green)").add_to(m)
        for a in trip["route"].arc_id:
            folium.PolyLine(line(a), color="green", weight=4, opacity=0.8).add_to(ref)
    if pred_arcs is not None:
        pr = folium.FeatureGroup("Predicted route (red dashed)").add_to(m)
        for a in pred_arcs:
            folium.PolyLine(line(a), color="red", weight=3, dash_array="6").add_to(pr)
    folium.LayerControl().add_to(m)
    out = out or RESULTS / "viz" / f"{trip['trip_id']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


if __name__ == "__main__":
    import sys

    from .config import TRAIN_DIR
    from .io import load_trip

    tid = sys.argv[1] if len(sys.argv) > 1 else "00000001"
    print(trip_map(load_trip(TRAIN_DIR / tid)))
