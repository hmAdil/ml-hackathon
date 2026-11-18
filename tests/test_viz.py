from pathsnap.viz import trip_map, trip_png
from test_eda import make_trip


def test_trip_map_and_png(tmp_path):
    trip = make_trip()
    html = trip_map(trip, out=tmp_path / "t.html").read_text()
    assert "Route breaks" in html and "Sampling gaps" in html
    assert trip_png(trip, out=tmp_path / "t.png").stat().st_size > 0
