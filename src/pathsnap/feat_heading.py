"""Experiment: distance-windowed heading for low-speed direction errors.

Diagnosis (label_quality): ~60% of the classifier's errors on reliable labels pick the reverse
arc of the true road, mostly at 2-5 m/s, where the 1-step heading in features.py is dominated
by GPS jitter. Here heading is taken from the displacement between the first point at least
WINDOW_M of path distance behind and the first point WINDOW_M ahead (within the same gap
segment and MAX_SPAN_S), which averages out jitter while still following turns.

Track-only and computed per trip, so it is inference-safe (the API receives the whole trip).

Usage: PYTHONPATH=src python -m pathsnap.feat_heading   (needs data/processed from dataset.py
and obs_*.parquet from label_quality.py)
"""
import importlib
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from .config import RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .dataset import PROCESSED
from .features import FEATURES
from .geo import bearing_deg, make_projector
from .io import list_trips, load_trip

WINDOW_M = 30.0
MAX_SPAN_S = 30.0
SEG_GAP_S = 60.0
NEW_FEATURES = ["w_ang_diff", "w_cos_diff", "w_heading_valid", "w_ang_rank", "w_ang_minus_min",
                "w_span_m", "heading_disagree"]


def windowed_heading(tr: pd.DataFrame, window_m: float = WINDOW_M, max_span_s: float = MAX_SPAN_S):
    """Returns (heading_deg or NaN, span_m) per point."""
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    x, y = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    t = tr.t.to_numpy(float)
    n = len(tr)
    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    seg = np.cumsum(np.r_[0, np.diff(t) > SEG_GAP_S])
    cum = np.cumsum(step)
    idx = np.arange(n)
    # first index behind with cum <= cum[i] - W; first index ahead with cum >= cum[i] + W
    back = np.searchsorted(cum, cum - window_m, side="right") - 1
    fwd = np.searchsorted(cum, cum + window_m, side="left")
    back = np.clip(back, 0, n - 1)
    fwd = np.clip(fwd, 0, n - 1)
    # stay inside the segment and the time span; shrink to the segment edge otherwise
    seg_start = np.searchsorted(seg, seg, side="left")
    seg_end = np.searchsorted(seg, seg, side="right") - 1
    back = np.maximum(back, seg_start)
    fwd = np.minimum(fwd, seg_end)
    lo = np.searchsorted(t, t - max_span_s, side="left")
    hi = np.searchsorted(t, t + max_span_s, side="right") - 1
    back = np.maximum(back, lo)
    fwd = np.minimum(fwd, hi)
    span = np.hypot(x[fwd] - x[back], y[fwd] - y[back])
    h = bearing_deg(x[back], y[back], x[fwd], y[fwd])
    ok = (span >= window_m) & (fwd > idx - 1) & (back < idx + 1)
    return np.where(ok, h, np.nan), span


def trip_features(trip_dir, cand: pd.DataFrame) -> pd.DataFrame:
    trip = load_trip(trip_dir)
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    h, span = windowed_heading(tr)
    o, a = cand.obs_idx.to_numpy(), cand.candidate_arc_id.to_numpy()
    ab = bearing_deg(nx[fn[a]], ny[fn[a]], nx[tn[a]], ny[tn[a]])
    hw = h[o]
    valid = ~np.isnan(hw)
    diff = np.abs((np.nan_to_num(hw) - ab + 180.0) % 360.0 - 180.0)
    f = pd.DataFrame(index=cand.index)
    f["w_heading_valid"] = valid.astype(np.float32)
    f["w_ang_diff"] = np.where(valid, diff, 90.0).astype(np.float32)
    f["w_cos_diff"] = np.where(valid, np.cos(np.radians(diff)), 0.0).astype(np.float32)
    f["w_span_m"] = span[o].astype(np.float32)
    g = f.assign(obs=o).groupby("obs").w_ang_diff
    f["w_ang_rank"] = g.rank(method="first").to_numpy()
    f["w_ang_minus_min"] = (f.w_ang_diff - g.transform("min")).to_numpy()
    # local (1-step) and windowed heading disagree on direction for this candidate
    f["heading_disagree"] = (np.sign(cand.cos_diff.to_numpy()) * np.sign(f.w_cos_diff) < 0
                             ).astype(np.float32)
    return f


def add_features(df: pd.DataFrame, directory) -> pd.DataFrame:
    dirs = {d.name: d for d in list_trips(directory)}
    parts = Parallel(n_jobs=6)(delayed(trip_features)(dirs[t], g)
                               for t, g in df.groupby("trip_id", sort=False))
    return df.join(pd.concat(parts))


def evaluate(df, p, obs):
    d = df[["trip_id", "obs_idx", "is_true", "dist_m"]].assign(p=p)
    top = d.loc[d.groupby(["trip_id", "obs_idx"]).p.idxmax()]
    true_d = d[d.is_true == 1].set_index(["trip_id", "obs_idx"]).dist_m
    top = top.set_index(["trip_id", "obs_idx"]).join(true_d.rename("true_d"))
    rel = obs.set_index(["trip_id", "obs_idx"]).reliable.reindex(top.index).to_numpy(bool)
    t = top[rel]
    same_geom_wrong = (t.is_true == 0) & ((t.dist_m - t.true_d).abs() < 0.5)
    return {"top1_all": float(top.is_true.mean()), "top1_reliable": float(t.is_true.mean()),
            "top1_reliable_per_trip": float(t.groupby(level=0).is_true.mean().mean()),
            "reverse_err_reliable": float(same_geom_wrong.mean())}


def run(model_name: str = "lgbm_rank"):
    mod = importlib.import_module(f"pathsnap.models.{model_name}")
    train = add_features(pd.read_parquet(PROCESSED / "train.parquet"), TRAIN_DIR)
    test = add_features(pd.read_parquet(PROCESSED / "student_test.parquet"), STUDENT_TEST_DIR)
    obs_tr = pd.read_parquet(PROCESSED / "obs_train.parquet")
    obs_te = pd.read_parquet(PROCESSED / "obs_student_test.parquet")
    folds = list(GroupKFold(n_splits=5).split(train, groups=train.trip_id))
    out = {"window_m": WINDOW_M, "max_span_s": MAX_SPAN_S, "model": model_name, "runs": {}}
    for name, feats in (("base", FEATURES), ("base+windowed", FEATURES + NEW_FEATURES)):
        oof = np.zeros(len(train))
        fold_top1 = []
        for tri, vai in folds:
            m = fit_cols(mod, train.iloc[tri], feats)
            oof[vai] = m.predict_proba(train.iloc[vai][feats].to_numpy(np.float32))[:, 1]
            fold_top1.append(evaluate(train.iloc[vai], oof[vai], obs_tr)["top1_reliable"])
        m = fit_cols(mod, train, feats)
        p_te = m.predict_proba(test[feats].to_numpy(np.float32))[:, 1]
        out["runs"][name] = {"cv_oof": evaluate(train, oof, obs_tr),
                             "cv_fold_top1_reliable": fold_top1,
                             "student_test": evaluate(test, p_te, obs_te)}
        print(name, json.dumps(out["runs"][name], indent=1), flush=True)
    a = np.array(out["runs"]["base+windowed"]["cv_fold_top1_reliable"])
    b = np.array(out["runs"]["base"]["cv_fold_top1_reliable"])
    out["paired_fold_diff"] = {"mean": float((a - b).mean()), "std": float((a - b).std()),
                               "folds_better": int((a > b).sum())}
    dst = RESULTS / "models" / "label_quality"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"windowed_heading_{model_name}.json").write_text(json.dumps(out, indent=2))
    print(out["paired_fold_diff"])
    return out


def fit_cols(mod, train: pd.DataFrame, feats: list):
    """Same rule as cv.fit_scorer (train only on observations that have a positive), with an
    explicit feature list."""
    has_pos = train.groupby(["trip_id", "obs_idx"]).is_true.transform("max") > 0
    tr = train[has_pos]
    model = mod.build(**mod.DEFAULT_PARAMS)
    X, y = tr[feats].to_numpy(np.float32), tr.is_true.to_numpy()
    if getattr(model, "needs_groups", False):
        model.fit(X, y, groups=tr.groupby(["trip_id", "obs_idx"], sort=False).size().to_numpy())
    else:
        model.fit(X, y)
    return model


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "lgbm_rank")
