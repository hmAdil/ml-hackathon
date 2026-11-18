"""Label-quality experiment: does training on unreliable labels hurt, and how much of the
reported top-1 error is label ambiguity rather than model error?

Usage: PYTHONPATH=src python -m pathsnap.label_quality [logreg lgbm_rank ...]

For every scorer and two training-label policies
    all       : every observation whose reference arc is a candidate (current behaviour)
    reliable  : additionally require align_dist_m <= RELIABLE_M (point within 30 m of its
                aligned reference arc; decision 4 in results/eda/FINDINGS.md)
we run the same 5-fold GroupKFold over train trips (out-of-fold, OOF, scores) and a refit on
all train trips scored on student_test. Each run is evaluated on several observation subsets:
    all obs | reliable obs | reliable obs in trips that pass the trip rule
Trip rule (fixed before looking at any model result): a trip is "mismatched" when fewer than
50% of its points lie within 30 m of their aligned arc (033 and 054 on this data).

Metrics: strict top-1 (prediction == aligned arc) and tolerant top-1 (prediction is the aligned
arc or the route arc just before/after it, i.e. the junction ambiguity of the alignment), both
obs-weighted and mean-per-trip. OOF scores are written to data/processed/oof_<model>_<policy>
.parquet for the leakage-safe HMM tuning in hmm_oof.py. Evaluator trips are never read.
"""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from .candidates import canonical_arc_ids
from .config import RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .cv import fit_scorer, score
from .dataset import PROCESSED
from .geo import make_projector, point_segment_dist
from .io import list_trips, load_trip
from .labels import align

RELIABLE_M = 30.0
MISMATCH_TRIP_FRAC = 0.5
N_SPLITS = 5
OUT = RESULTS / "models" / "label_quality"


def trip_obs_info(trip_dir) -> tuple[pd.DataFrame, dict]:
    """Per-observation alignment info + route arrays used to classify errors."""
    trip = load_trip(trip_dir)
    lab = align(trip)
    canon = canonical_arc_ids(trip["arcs"], len(trip["nodes"]))
    rc = canon[trip["route"].arc_id.to_numpy()]
    pos = lab.route_pos.to_numpy()
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    proj = make_projector(tr.lon.mean(), tr.lat.mean())
    nx, ny = proj(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    px, py = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()

    def d_to(route_idx):
        a = rc[np.clip(route_idx, 0, len(rc) - 1)]
        d, _ = point_segment_dist(px, py, nx[fn[a]], ny[fn[a]], nx[tn[a]], ny[tn[a]])
        return d

    df = pd.DataFrame({"trip_id": trip["trip_id"], "obs_idx": lab.obs_idx, "route_pos": pos,
                       "true_arc": canon[lab.arc_id.to_numpy()],
                       "align_dist_m": lab.align_dist_m, "d_prev_route_arc": d_to(pos - 1),
                       "d_next_route_arc": d_to(pos + 1)})
    return df, {"rc": rc, "fn": fn, "tn": tn}


def obs_tables(directory):
    res = Parallel(n_jobs=6)(delayed(trip_obs_info)(d) for d in list_trips(directory))
    obs = pd.concat([r[0] for r in res], ignore_index=True)
    trips = {r[0].trip_id.iloc[0]: r[1] for r in res}
    frac = obs.groupby("trip_id").align_dist_m.apply(lambda d: (d <= RELIABLE_M).mean())
    obs["reliable"] = obs.align_dist_m <= RELIABLE_M
    obs["trip_ok"] = obs.trip_id.map(frac >= MISMATCH_TRIP_FRAC)
    return obs, trips


def top_predictions(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    d = df[["trip_id", "obs_idx", "candidate_arc_id", "is_true"]].assign(p=p)
    top = d.loc[d.groupby(["trip_id", "obs_idx"]).p.idxmax()]
    return top.rename(columns={"candidate_arc_id": "pred_arc", "is_true": "correct"})[
        ["trip_id", "obs_idx", "pred_arc", "correct"]]


def classify_errors(top: pd.DataFrame, obs: pd.DataFrame, trips: dict) -> pd.DataFrame:
    """Adds tolerant-correctness and an error category for every observation."""
    m = obs.merge(top, on=["trip_id", "obs_idx"], how="left")
    cat = np.full(len(m), "correct", dtype=object)
    tol = m.correct.to_numpy().astype(bool)
    for tid, g in m.groupby("trip_id"):
        t = trips[tid]
        rc, fn, tn = t["rc"], t["fn"], t["tn"]
        i = g.index.to_numpy()
        pos, pred, true = g.route_pos.to_numpy(), g.pred_arc.to_numpy(), g.true_arc.to_numpy()
        prev_a, next_a = rc[np.maximum(pos - 1, 0)], rc[np.minimum(pos + 1, len(rc) - 1)]
        adj = (pred == prev_a) | (pred == next_a)
        rev = (fn[pred] == tn[true]) & (tn[pred] == fn[true])
        on_route = np.isin(pred, rc)
        wrong = ~g.correct.to_numpy().astype(bool)
        tol[i] = ~wrong | adj
        c = np.where(adj, "adjacent_route_arc",
                     np.where(rev, "reverse_of_true", np.where(on_route, "other_route_arc",
                                                                "off_route_arc")))
        cat[i[wrong]] = c[wrong]
    m["tolerant_correct"] = tol
    m["error_type"] = cat
    return m


def summarise(m: pd.DataFrame, covered: pd.Series) -> dict:
    m = m.assign(covered=covered.reindex(pd.MultiIndex.from_frame(m[["trip_id", "obs_idx"]]))
                 .to_numpy())
    m.loc[~m.covered.astype(bool), "error_type"] = "true_arc_not_candidate"
    subsets = {"all": np.ones(len(m), bool), "reliable": m.reliable.to_numpy(),
               "reliable_trip_ok": (m.reliable & m.trip_ok).to_numpy()}
    out = {}
    for name, s in subsets.items():
        g = m[s]
        out[name] = {
            "n_obs": int(len(g)), "n_trips": int(g.trip_id.nunique()),
            "top1": float(g.correct.mean()),
            "top1_per_trip": float(g.groupby("trip_id").correct.mean().mean()),
            "top1_tolerant": float(g.tolerant_correct.mean()),
            "top1_tolerant_per_trip": float(g.groupby("trip_id").tolerant_correct.mean().mean()),
            "errors": {k: float(v) for k, v in
                       g.error_type.value_counts(normalize=True).items() if k != "correct"},
        }
    return out


def train_mask(df: pd.DataFrame, obs: pd.DataFrame, policy: str) -> np.ndarray:
    if policy == "all":
        return np.ones(len(df), bool)
    rel = obs.set_index(["trip_id", "obs_idx"]).reliable
    return rel.reindex(pd.MultiIndex.from_frame(df[["trip_id", "obs_idx"]])).to_numpy(bool)


def run(models=("logreg", "lgbm_rank"), policies=("all", "reliable")):
    import importlib

    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "student_test.parquet")
    obs_tr, trips_tr = obs_tables(TRAIN_DIR)
    obs_te, trips_te = obs_tables(STUDENT_TEST_DIR)
    obs_tr.to_parquet(PROCESSED / "obs_train.parquet", index=False)
    obs_te.to_parquet(PROCESSED / "obs_student_test.parquet", index=False)
    cov_tr = train.groupby(["trip_id", "obs_idx"]).is_true.max().astype(bool)
    cov_te = test.groupby(["trip_id", "obs_idx"]).is_true.max().astype(bool)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(train, groups=train.trip_id))

    results = {"reliable_m": RELIABLE_M, "mismatch_trip_frac": MISMATCH_TRIP_FRAC,
               "mismatched_trips": sorted(obs_tr.loc[~obs_tr.trip_ok, "trip_id"].unique()),
               "label_stats": label_stats(obs_tr, obs_te), "runs": {}}
    for name in models:
        mod = importlib.import_module(f"pathsnap.models.{name}")
        for policy in policies:
            key = f"{name}/{policy}"
            print(key, flush=True)
            keep = train_mask(train, obs_tr, policy)
            oof = np.zeros(len(train))
            for tri, vai in folds:
                tri = tri[keep[tri]]
                model, _ = fit_scorer(mod.build, mod.DEFAULT_PARAMS, train.iloc[tri])
                oof[vai] = score(model, train.iloc[vai])
            train[["trip_id", "obs_idx", "candidate_arc_id"]].assign(p=oof).to_parquet(
                PROCESSED / f"oof_{name}_{policy}.parquet", index=False)
            m_cv = classify_errors(top_predictions(train, oof), obs_tr, trips_tr)
            model, _ = fit_scorer(mod.build, mod.DEFAULT_PARAMS, train[keep])
            p_te = score(model, test)
            test[["trip_id", "obs_idx", "candidate_arc_id"]].assign(p=p_te).to_parquet(
                PROCESSED / f"test_{name}_{policy}.parquet", index=False)
            m_te = classify_errors(top_predictions(test, p_te), obs_te, trips_te)
            results["runs"][key] = {"cv_oof": summarise(m_cv, cov_tr),
                                    "student_test": summarise(m_te, cov_te),
                                    "n_train_rows": int(keep.sum())}
            r = results["runs"][key]
            print(f"  cv top1 all={r['cv_oof']['all']['top1']:.4f} "
                  f"reliable={r['cv_oof']['reliable']['top1']:.4f} "
                  f"tolerant={r['cv_oof']['reliable']['top1_tolerant']:.4f} | "
                  f"test top1={r['student_test']['all']['top1']:.4f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    table(results).to_csv(OUT / "summary.csv", index=False)
    print(table(results).to_string(index=False))
    return results


def label_stats(obs_tr, obs_te) -> dict:
    out = {}
    for part, o in (("train", obs_tr), ("student_test", obs_te)):
        amb = np.minimum(o.d_prev_route_arc, o.d_next_route_arc) - o.align_dist_m
        out[part] = {
            "n_obs": int(len(o)), "reliable_frac": float(o.reliable.mean()),
            "align_dist_p50": float(o.align_dist_m.median()),
            "align_dist_p95": float(o.align_dist_m.quantile(0.95)),
            # adjacent route arc within 2 m of the aligned one: label is a coin flip there
            "ambiguous_2m_frac": float((amb < 2).mean()),
            "ambiguous_5m_frac": float((amb < 5).mean()),
        }
    return out


def table(results: dict) -> pd.DataFrame:
    rows = []
    for key, r in results["runs"].items():
        for split in ("cv_oof", "student_test"):
            for sub, s in r[split].items():
                rows.append({"run": key, "split": split, "subset": sub, "n_obs": s["n_obs"],
                             "top1": s["top1"], "top1_per_trip": s["top1_per_trip"],
                             "top1_tolerant": s["top1_tolerant"],
                             "top1_tolerant_per_trip": s["top1_tolerant_per_trip"]})
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    args = sys.argv[1:]
    run(models=tuple(args) if args else ("logreg", "lgbm_rank"))
