import numpy as np

from pathsnap.geo import haversine_m, point_segment_dist
from pathsnap.metrics import candidate_miss_rate, route_continuity, top1_accuracy


def test_haversine_one_degree_lat():
    assert abs(haversine_m(8, 49, 8, 50) - 111_195) < 200


def test_point_segment():
    d, f = point_segment_dist(np.array([5.0]), np.array([3.0]), 0.0, 0.0, 10.0, 0.0)
    assert d[0] == 3.0 and f[0] == 0.5


def test_metrics():
    t, o, p, y = [0, 0, 0, 0], [0, 0, 1, 1], [0.9, 0.1, 0.2, 0.8], [1, 0, 0, 1]
    assert top1_accuracy(t, o, p, y) == 1.0
    assert candidate_miss_rate(t, o, y) == 0.0
    assert route_continuity([0, 1], [0, 5], [5, 9]) == 1.0
