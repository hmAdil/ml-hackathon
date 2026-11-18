"""Phase 7/8: trip-grouped cross-validation and held-out evaluation shared by every model.

Metrics follow docs/PLAN.md: per observation the highest-scoring candidate is the prediction.
- top1: share of observations whose top candidate is the reference arc (observations whose
  reference arc is not among the candidates count as wrong)
- macro_f1: macro F1 over candidate rows (class 1 = "chosen and correct arc"), where each
  observation marks exactly one candidate as predicted-positive
- log_loss: candidate-level probability quality
- route_continuity: classifier-only predicted arc sequence, mean over trips
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from .config import TRAIN_DIR
from .features import FEATURES
from .io import list_trips
from .metrics import route_continuity


def load_arcs(partition_dir):
    return {d.name: pd.read_csv(d / f"{d.name}.arcs", sep=r"\s+", header=None,
                                names=["f", "t"]) for d in list_trips(partition_dir)}


def evaluate(df: pd.DataFrame, p: np.ndarray, arcs: dict | None = None) -> dict:
    d = df[["trip_id", "obs_idx", "candidate_arc_id", "is_true"]].copy()
    d["p"] = p
    top = d.loc[d.groupby(["trip_id", "obs_idx"]).p.idxmax()]
    pred = np.zeros(len(d), dtype=np.int8)
    pred[d.index.get_indexer(top.index)] = 1
    y = d.is_true.to_numpy()
    covered = d.groupby(["trip_id", "obs_idx"]).is_true.max()
    out = {
        "top1": float(top.is_true.mean()),
        "top1_covered": float(top.is_true.mean() / max(covered.mean(), 1e-9)),
        "top1_per_trip": float(top.groupby("trip_id").is_true.mean().mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "candidate_miss_rate": float(1 - covered.mean()),
        "n_obs": int(len(top)), "n_rows": int(len(d)),
    }
    if arcs is not None:
        cont = [route_continuity(g.sort_values("obs_idx").candidate_arc_id.to_numpy(),
                                 arcs[t].f.to_numpy(), arcs[t].t.to_numpy())
                for t, g in top.groupby("trip_id")]
        out["route_continuity"] = float(np.mean(cont))
    return out


def fit_scorer(build, params, train: pd.DataFrame):
    """Train only on observations whose reference arc is a candidate (others carry no signal)."""
    has_pos = train.groupby(["trip_id", "obs_idx"]).is_true.transform("max") > 0
    tr = train[has_pos]
    model = build(**params)
    t0 = time.perf_counter()
    model.fit(tr[FEATURES].to_numpy(np.float32), tr.is_true.to_numpy())
    return model, time.perf_counter() - t0


def score(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]


def cross_validate(build, params, train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tri, vai) in enumerate(gkf.split(train, groups=train.trip_id)):
        model, fit_s = fit_scorer(build, params, train.iloc[tri])
        val = train.iloc[vai]
        m = evaluate(val, score(model, val))
        m.update(fold=k, fit_s=fit_s, val_trips=val.trip_id.nunique())
        rows.append(m)
        print(f"  fold {k}: top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f} ({fit_s:.0f}s)", flush=True)
    return pd.DataFrame(rows)


__all__ = ["cross_validate", "evaluate", "fit_scorer", "score", "load_arcs", "TRAIN_DIR"]
