import joblib
import numpy as np
import pytest

from pathsnap.models import mlp, xgboost


def _data(n=3000, d=22, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * r.normal(size=n) > 0).astype(int)
    X[r.random(X.shape) < 0.05] = np.nan
    return X, y


def _check(m, X, y):
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(1), 1)
    assert ((p[:, 1] > 0.5) == y).mean() > 0.8
    return p


def test_xgboost_fit_predict():
    X, y = _data()
    _check(xgboost.build(n_estimators=30), X, y)


def test_mlp_fit_predict_pickle(tmp_path, monkeypatch):
    monkeypatch.setattr(mlp, "RESULTS", tmp_path)
    X, y = _data()
    m = mlp.build(max_epochs=5, batch_size=1024)
    p = _check(m, X, y)
    assert len(m.history_) >= 1 and (tmp_path / "models" / "mlp_history.json").exists()
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(m2.predict_proba(X), p, atol=1e-4)


def test_gpu_used_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    assert mlp._dev().type == "cuda" and xgboost._device() == "cuda"
