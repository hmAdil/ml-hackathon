import numpy as np
import pandas as pd

from pathsnap.feat_heading import windowed_heading

DEG_M = 111_195.0


def test_windowed_heading_is_stable_under_jitter():
    """Walking north at 3 m/s with 4 m sideways jitter: 1-step heading is noisy,
    windowed heading should stay close to 0 deg (north)."""
    rng = np.random.default_rng(0)
    n = 200
    north_m = np.arange(n) * 3.0
    east_m = rng.normal(0, 4, n)
    lat = 49.0 + north_m / DEG_M
    lon = 8.0 + east_m / (DEG_M * np.cos(np.radians(49)))
    tr = pd.DataFrame({"lon": lon, "lat": lat, "t": np.arange(n, dtype=float)})
    h, span = windowed_heading(tr)
    ok = ~np.isnan(h)
    assert ok[20:-20].mean() > 0.8  # invalid only where net displacement < 30 m
    err = np.abs((h[ok] + 180) % 360 - 180)
    assert np.percentile(err, 95) < 25


def test_windowed_heading_respects_gaps():
    tr = pd.DataFrame({"lon": 8.0, "lat": 49.0 + np.r_[np.arange(5), 100 + np.arange(5)] / DEG_M,
                       "t": np.r_[np.arange(5), 500 + np.arange(5)].astype(float)})
    h, _ = windowed_heading(tr)
    assert np.isnan(h).all()  # each 4 m segment is shorter than the 30 m window
