import numpy as np

from src.common.metrics import candidate_miss_rate, route_continuity, top1_accuracy
from src.models import REQUIRED, discover


def test_every_model_follows_contract():
    registry = discover()
    assert "baseline_prior" in registry
    for name, mod in registry.items():
        for attr in REQUIRED:
            assert hasattr(mod, attr), f"{name} missing {attr}"
        est = mod.build()
        X = np.random.RandomState(0).rand(40, 3)
        y = np.array([0, 1] * 20)
        est.fit(X, y)
        assert est.predict_proba(X).shape == (40, 2), name


def test_metrics_basic():
    t, o = [0, 0, 0, 0], [0, 0, 1, 1]
    p, y = [0.9, 0.1, 0.2, 0.8], [1, 0, 0, 1]
    assert top1_accuracy(t, o, p, y) == 1.0
    assert candidate_miss_rate(t, o, y) == 0.0
    assert route_continuity([0, 1], arcs_from=[0, 5], arcs_to=[5, 9]) == 1.0
