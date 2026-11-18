"""Synthetic-trip tests for the Phase 2 EDA modules (no real data needed)."""
import numpy as np
import pandas as pd

from pathsnap.eda_gps import gap_table, gps_summary, point_table, spike_mask, turn_deg
from pathsnap.eda_network import network_analysis
from pathsnap.eda_route import route_analysis

DEG_M = 111_195.0  # metres per degree latitude


def make_trip():
    """Straight road north along lon=8.0: nodes every ~111 m, arcs both ways.
    GPS drives north at 10 m/s, 1 Hz, with one 120 s gap and one 1 km spike."""
    lat = 49.0 + np.arange(11) * 0.001
    nodes = pd.DataFrame({"lon": 8.0, "lat": lat})
    fwd = [(i, i + 1) for i in range(10)]
    bwd = [(i + 1, i) for i in range(10)]
    arcs = pd.DataFrame(fwd + bwd + [(3, 3), (0, 1)], columns=["from_node", "to_node"])
    t = np.arange(100, dtype=float)
    t[50:] += 120  # one 121 s interval between obs 49 and 50
    glat = 49.0 + (np.arange(100) * 10.0) / DEG_M
    glon = np.full(100, 8.0)
    glon[30] += 1000 / (DEG_M * np.cos(np.radians(49)))  # out-and-back spike
    track = pd.DataFrame({"lon": glon, "lat": glat, "t": t})
    route = pd.DataFrame({"arc_id": [0, 1, 2, 4, 5, 6, 7, 8]})  # arc 2->4 skips node 3->4
    return {"trip_id": "T", "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def test_turn_deg_wraps():
    assert turn_deg(350, 10) == 20 and turn_deg(10, 350) == -20 and turn_deg(0, 180) == 180


def test_gps_summary_gap_and_spike():
    trip = make_trip()
    s = gps_summary(trip["track"])
    assert s["n_dt_gt60"] == 1 and s["dt_max"] == 121
    assert s["n_segments_gap60"] == 2
    assert s["n_spikes"] == 1
    assert abs(s["speed_p50_ms"] - 10) < 0.1
    assert spike_mask(point_table(trip["track"]))[30]
    g = gap_table(trip["track"], "T", 60)
    assert len(g) == 1 and g.obs_idx.iloc[0] == 50


def test_network_analysis():
    trip = make_trip()
    s, aux = network_analysis(trip["nodes"], trip["arcs"], trip["track"])
    assert s["self_loops"] == 1 and s["dup_arcs"] == 1
    assert s["isolated_nodes"] == 0 and s["strong_components"] == 1
    assert abs(s["arc_len_p50_m"] - 111.2) < 1
    assert s["gps_near_arc_p50_m"] < 1  # GPS lies on the road


def test_route_analysis_break_and_alignment():
    trip = make_trip()
    _, aux = network_analysis(trip["nodes"], trip["arcs"], trip["track"])
    s, align = route_analysis(trip, aux, aux["gps_nearest_arc_d"])
    assert s["route_breaks"] == 1 and abs(s["route_connected_frac"] - 6 / 7) < 1e-9
    assert abs(s["route_break_gap_median_m"] - 111.2) < 1
    assert s["align_spearman"] > 0.95
    assert len(align) == len(trip["track"])
