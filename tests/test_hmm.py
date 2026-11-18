"""HMM decoder on a tiny synthetic directed network.

Layout (metres, x east): a straight road 0->1->2->3->4 (arcs 0..3, 100 m each, y=0) and a
parallel unconnected road 5->6 (arc 4, y=15, spanning x 100..200) that is a decoy.
"""
import numpy as np
import pandas as pd

from pathsnap.geo import make_projector
from pathsnap.hmm import decode

LON0, LAT0 = 8.0, 50.0
PROJ = make_projector(LON0, LAT0)
M_LON = 1 / PROJ(LON0 + 1, LAT0)[0]  # degrees lon per metre
M_LAT = 1 / PROJ(LON0, LAT0 + 1)[1]


def _trip(times, xs, ys):
    nodes_xy = [(0, 0), (100, 0), (200, 0), (300, 0), (400, 0), (100, 15), (200, 15)]
    nodes = pd.DataFrame({"lon": [LON0 + x * M_LON for x, _ in nodes_xy],
                          "lat": [LAT0 + y * M_LAT for _, y in nodes_xy]})
    arcs = pd.DataFrame({"from_node": [0, 1, 2, 3, 5], "to_node": [1, 2, 3, 4, 6]})
    track = pd.DataFrame({"lon": [LON0 + x * M_LON for x in xs],
                          "lat": [LAT0 + y * M_LAT for y in ys], "t": times})
    return {"trip_id": "syn", "track": track, "nodes": nodes, "arcs": arcs}


def _scored(rows):
    """rows: (obs, arc, p, frac). arc_len_m filled from the layout, dist_m unused (w_dist=0)."""
    df = pd.DataFrame(rows, columns=["obs_idx", "arc_id", "p", "frac"])
    df["arc_len_m"] = 100.0
    df["dist_m"] = 5.0
    return df


def test_repairs_disconnected_classifier_jump():
    # vehicle drives along arcs 0,1,2 at 10 m/s, 1 Hz is too slow for 100 m steps so use 5 s.
    xs = [20, 70, 120, 170, 220, 270]
    t = [0, 5, 10, 15, 20, 25]
    trip = _trip(t, xs, [0] * 6)
    arc_of = [0, 0, 1, 1, 2, 2]
    frac = [0.2, 0.7, 0.2, 0.7, 0.2, 0.7]
    rows = []
    for o in range(6):
        rows.append((o, arc_of[o], 0.6, frac[o]))
        # decoy arc 4 (parallel, disconnected from arc 0 / 2) wins the classifier at obs 3
        rows.append((o, 4, 0.9 if o == 3 else 0.2, min(max((xs[o] - 100) / 100, 0), 1)))
    s = _scored(rows)
    clf = s.loc[s.groupby("obs_idx").p.idxmax()].sort_values("obs_idx").arc_id.tolist()
    assert clf[3] == 4  # classifier-only takes the disconnected jump
    out = decode(trip, s, {"w_dist": 0.0})
    assert out.arc_id.tolist() == arc_of


def test_one_arc_per_observation_and_single_candidate():
    trip = _trip([0, 1, 2], [10, 20, 30], [0, 0, 0])
    s = _scored([(0, 0, 0.5, 0.1), (1, 0, 0.5, 0.2), (2, 0, 0.5, 0.3)])  # one candidate each
    out = decode(trip, s, {})
    assert out.obs_idx.tolist() == [0, 1, 2]
    assert out.arc_id.tolist() == [0, 0, 0]


def test_gap_splits_segments():
    # After a 100 s gap the vehicle is on the unconnected decoy road; with no transition across
    # the gap the second segment must follow its own emissions (arc 4) while the first stays on 0.
    trip = _trip([0, 1, 200, 201], [10, 20, 150, 160], [0, 0, 15, 15])
    s = _scored([(0, 0, 0.9, 0.1), (0, 4, 0.1, 0.0), (1, 0, 0.9, 0.2), (1, 4, 0.1, 0.0),
                 (2, 0, 0.1, 1.0), (2, 4, 0.9, 0.5), (3, 0, 0.1, 1.0), (3, 4, 0.9, 0.6)])
    out = decode(trip, s, {"gap_s": 60.0})
    assert out.arc_id.tolist() == [0, 0, 4, 4]
    # Without the gap the (finite) fallback penalty must still give a valid decode.
    out2 = decode(trip, s, {"gap_s": 1e9})
    assert len(out2) == 4 and np.isfinite(out2.arc_id.to_numpy()).all()
