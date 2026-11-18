import numpy as np
import pandas as pd

from pathsnap.candidates import build_candidates, canonical_arc_ids
from pathsnap.features import FEATURES, build_features
from pathsnap.labels import align


def toy_trip():
    # three collinear eastward arcs (~110 m each) plus a distant decoy arc; GPS 5 m north of road
    lon = np.array([8.0, 8.0014, 8.0028, 8.0042])
    lat = np.full(4, 50.0)
    nodes = pd.DataFrame({"lon": np.r_[lon, 8.002], "lat": np.r_[lat, 50.01]})
    arcs = pd.DataFrame({"from_node": [0, 1, 2, 4], "to_node": [1, 2, 3, 4]})
    gps_lon = np.linspace(8.0001, 8.004, 12)
    track = pd.DataFrame({"lon": gps_lon, "lat": 50.00005, "t": np.arange(1.0, 13.0)})
    return {"trip_id": "toy", "track": track, "nodes": nodes, "arcs": arcs,
            "route": pd.DataFrame({"arc_id": [0, 1, 2]})}


def test_align_is_monotone_and_on_route():
    a = align(toy_trip())
    assert (np.diff(a.route_pos) >= 0).all()
    assert a.arc_id.iloc[0] == 0 and a.arc_id.iloc[-1] == 2
    assert a.align_dist_m.max() < 15


def test_candidates_and_features():
    trip = toy_trip()
    c = build_candidates(trip)
    assert c.obs_idx.nunique() == len(trip["track"])
    assert 3 not in set(c.arc_id)  # decoy is far away and canonical ids exclude nothing else
    f = build_features(trip, c)
    assert list(f[FEATURES].columns) == FEATURES
    assert f.groupby("obs_idx").dist_rank.min().eq(1).all()


def test_canonical_collapses_duplicates():
    arcs = pd.DataFrame({"from_node": [0, 0, 1], "to_node": [1, 1, 2]})
    assert list(canonical_arc_ids(arcs, 3)) == [0, 0, 2]
