import joblib
import numpy as np

from pathsnap.models import lgbm_rank, random_forest


def _data(n_obs=60, k=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs * k, 4)).astype(np.float32)
    X[::7, 0] = np.nan
    y = np.zeros(n_obs * k, dtype=int)
    for i in range(n_obs):
        j = i * k + rng.integers(k)
        y[j] = 1
        X[j, 1] += 3
    return X, y, np.full(n_obs, k)


def _check(m, X, y):
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2) and np.all((p[:, 1] >= 0) & (p[:, 1] <= 1))
    top = p[:, 1].reshape(-1, 5).argmax(1)
    assert (y.reshape(-1, 5)[np.arange(len(top)), top]).mean() > 0.8


def test_random_forest():
    X, y, _ = _data()
    m = random_forest.build(n_estimators=20, max_samples=None, min_samples_leaf=2, n_jobs=2)
    _check(m.fit(X, y), X, y)


def test_lgbm_rank_and_pickle(tmp_path):
    X, y, g = _data()
    m = lgbm_rank.build(n_estimators=30, min_child_samples=5, n_jobs=2)
    assert m.needs_groups
    m.fit(X, y, groups=g)
    _check(m, X, y)
    joblib.dump(m, tmp_path / "m.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "m.joblib").predict_proba(X),
                               m.predict_proba(X))
