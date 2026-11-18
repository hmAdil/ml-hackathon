from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("PATHSNAP_EVAL_DIR", str(ROOT / "data" / "student_test"))
    from pathsnap.api import app

    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_trips(client):
    assert len(client.get("/trips").json()["trips"]) == 10


def test_predict(client):
    tid = client.get("/trips").json()["trips"][0]
    r = client.post(f"/predict/{tid}")
    assert r.status_code == 200
    d = r.json()
    for k in ("trip_id", "gps", "route", "metrics", "timings_ms", "model", "decoder"):
        assert k in d
    assert d["trip_id"] == tid and d["gps"] and d["route"]
    assert set(d["route"][0]) == {"arc_id", "polyline"}
    assert 0 <= d["metrics"]["top1_accuracy"] <= 1
    blob = r.text.lower()
    assert "reference" not in blob and "is_true" not in blob and "label" not in blob


def test_unknown_trip(client):
    assert client.post("/predict/nope").status_code == 404
