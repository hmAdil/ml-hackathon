"""Evaluate the HMM decoder against classifier-only top-1, tune it, freeze the config.

Usage: python -m pathsnap.hmm_eval [--trials N] [--skip-tune] [--jobs 4]

Scorer: artifacts/logreg.joblib. NOTE: train-trip scores are IN-SAMPLE (logreg was fit on the
train partition), so train numbers favour the classifier; student_test numbers are held out for
the scorer (but both partitions are used to tune the HMM, as instructed).

Metrics per trip (reference = labels.align arc per observation, canonical ids):
  top1        fraction of observations whose predicted arc equals the reference arc
  continuity  metrics.route_continuity of the predicted per-observation arc sequence
  seq_f1      set-level precision/recall/F1: predicted arc sequence is de-duplicated, consecutive
              disconnected arcs are bridged with the directed shortest path (<= FILL_M metres,
              not across GPS gaps > 60 s), and the set of arcs is compared with the set of
              reference-route arcs (canonical ids). Rewards covering the true route, penalises
              spurious arcs, and is insensitive to per-point alignment ambiguity.

Tuning: random search, `--trials` configs (seed 0) plus the defaults over SPACE, objective
= mean over all 90 train+student_test trips of (top1 + seq_f1) / 2. No other data is used.
"""
import argparse
import heapq
import json
import time

import joblib
import numpy as np
import pandas as pd

from .candidates import canonical_arc_ids
from .config import ARTIFACTS, RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .geo import make_projector
from .hmm import DEFAULT_PARAMS, decode
from .io import list_trips, load_trip
from .labels import align
from .metrics import route_continuity
from .pipeline import score_candidates

FILL_M = 1500.0
OUT = RESULTS / "hmm"
SPACE = {
    "max_cands": [6, 10, 15],
    "w_emis": [0.5, 1.0, 2.0],
    "w_dist": [0.0, 0.5, 1.0],
    "beta": [10.0, 20.0, 30.0, 60.0],
    "max_pen": [4.0, 8.0, 15.0],
    "uturn_pen": [0.0, 2.0, 5.0],
    "vmax": [50.0, 70.0, 100.0],
    "speed_w": [0.05, 0.1, 0.3],
}
_MEM: dict = {}


def prepare(trip_dir):
    """Load a trip, score it and compute reference arcs. Memoised per process."""
    key = str(trip_dir)
    if key not in _MEM:
        trip = load_trip(trip_dir)
        model = joblib.load(ARTIFACTS / "logreg.joblib")
        scored = score_candidates(trip, model)
        canon = canonical_arc_ids(trip["arcs"], len(trip["nodes"]))
        ref = canon[align(trip).arc_id.to_numpy()]
        _MEM[key] = (trip, scored, canon, ref)
    return _MEM[key]


def _adj(trip):
    if "_adj" not in trip:
        arcs, nodes, tr = trip["arcs"], trip["nodes"], trip["track"]
        fn, tn = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
        order = np.argsort(fn, kind="stable")
        ip = np.concatenate([[0], np.cumsum(np.bincount(fn, minlength=len(nodes)))])
        px, py = make_projector(tr.lon.mean(), tr.lat.mean())(nodes.lon.to_numpy(),
                                                              nodes.lat.to_numpy())
        ln = np.hypot(px[tn] - px[fn], py[tn] - py[fn])
        trip["_adj"] = (ip.tolist(), order.tolist(), tn.tolist(), ln.tolist(), fn, tn)
    return trip["_adj"]


def _path_arcs(trip, a, b, limit=FILL_M):
    """Arcs strictly between a and b on the shortest directed path head(a)->tail(b), or None."""
    ip, order, tn_l, ln_l, fn, tn = _adj(trip)
    src, dst = int(tn[a]), int(fn[b])
    if src == dst:
        return []
    dist, par, heap = {src: 0.0}, {}, [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, 1e30):
            continue
        if u == dst:
            out = []
            while u != src:
                k = par[u]
                out.append(k)
                u = int(fn[k])
            return out[::-1]
        for j in range(ip[u], ip[u + 1]):
            k = order[j]
            v, nd = tn_l[k], d + ln_l[k]
            if nd <= limit and nd < dist.get(v, 1e30):
                dist[v] = nd
                par[v] = k
                heapq.heappush(heap, (nd, v))
    return None


def seq_prf(trip, canon, pred_arc, t_obs, ref_set):
    """Set precision/recall/F1 of the gap-filled predicted arc sequence vs the reference set."""
    pred = np.asarray(pred_arc)
    chg = np.r_[True, pred[1:] != pred[:-1]]
    seq, ts = pred[chg], t_obs[chg]
    arcs = [int(seq[0])] if len(seq) else []
    for i in range(1, len(seq)):
        if ts[i] - ts[i - 1] <= 60:
            mid = _path_arcs(trip, int(seq[i - 1]), int(seq[i]))
            if mid:
                arcs.extend(int(canon[m]) for m in mid)
        arcs.append(int(seq[i]))
    ps = set(arcs)
    tp = len(ps & ref_set)
    prec, rec = tp / max(len(ps), 1), tp / max(len(ref_set), 1)
    return prec, rec, (2 * prec * rec / (prec + rec) if tp else 0.0)


def eval_trip(trip_dir, params, with_clf=True):
    trip, scored, canon, ref = prepare(trip_dir)
    t_obs = trip["track"].t.to_numpy(float)
    ref_set = set(canon[trip["route"].arc_id.to_numpy()].tolist())
    row = {"trip_id": trip["trip_id"], "n_obs": len(t_obs)}
    t0 = time.time()
    runs = {"hmm": decode(trip, scored, params).sort_values("obs_idx")}
    row["hmm_s"] = time.time() - t0
    if with_clf:
        top = scored.loc[scored.groupby("obs_idx").p.idxmax()].sort_values("obs_idx")
        runs["clf"] = top[["obs_idx", "arc_id"]]
    ff, ft = trip["arcs"].from_node.to_numpy(), trip["arcs"].to_node.to_numpy()
    for name, df in runs.items():
        o, a = df.obs_idx.to_numpy(), df.arc_id.to_numpy()
        row[f"top1_{name}"] = float(np.mean(a == ref[o]))
        row[f"cont_{name}"] = route_continuity(a, ff, ft)
        p, r, f = seq_prf(trip, canon, a, t_obs[o], ref_set)
        row[f"prec_{name}"], row[f"rec_{name}"], row[f"f1_{name}"] = p, r, f
    return row


def all_trip_dirs():
    return [("student_test", d) for d in list_trips(STUDENT_TEST_DIR)] + \
           [("train", d) for d in list_trips(TRAIN_DIR)]


def evaluate(params, jobs=4, with_clf=True):
    dirs = all_trip_dirs()
    rows = joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(eval_trip)(d, params, with_clf) for _, d in dirs)
    df = pd.DataFrame(rows)
    df.insert(0, "partition", [p for p, _ in dirs])
    return df


def objective(df):
    return float(((df.top1_hmm + df.f1_hmm) / 2).mean())


def tune(trials, jobs, seed=0):
    rng = np.random.default_rng(seed)
    cfgs = [{}] + [{k: v[rng.integers(len(v))] for k, v in SPACE.items()}
                   for _ in range(trials)]
    log = []
    for i, c in enumerate(cfgs):
        t0 = time.time()
        sc = objective(evaluate(c, jobs, with_clf=False))
        log.append({"trial": i, "objective": sc, "params": c})
        print(f"trial {i}: obj={sc:.4f} ({time.time() - t0:.0f}s) {c}", flush=True)
    best = max(log, key=lambda r: r["objective"])
    return {**DEFAULT_PARAMS, **best["params"]}, log


def summarise(df):
    s = {}
    for part, g in list(df.groupby("partition")) + [("all", df)]:
        d = {"n_trips": int(len(g)), "n_obs": int(g.n_obs.sum())}
        for m in ["top1", "cont", "prec", "rec", "f1"]:
            for k in ["clf", "hmm"]:
                d[f"{m}_{k}_mean_per_trip"] = float(g[f"{m}_{k}"].mean())
        for m in ["top1", "cont"]:
            for k in ["clf", "hmm"]:
                d[f"{m}_{k}_obs_weighted"] = float((g[f"{m}_{k}"] * g.n_obs).sum() / g.n_obs.sum())
        d["trips_hmm_hurts_top1"] = int((g.top1_hmm < g.top1_clf - 1e-9).sum())
        d["trips_hmm_hurts_f1"] = int((g.f1_hmm < g.f1_clf - 1e-9).sum())
        d["hmm_seconds_max"] = float(g.hmm_s.max())
        d["hmm_seconds_mean"] = float(g.hmm_s.mean())
        s[part] = d
    return s


def plot(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, m, ttl in zip(ax, ["top1", "cont", "f1"],
                         ["Pointwise top-1", "Route continuity", "Sequence F1"]):
        for part, c in [("train", "#888888"), ("student_test", "#d62728")]:
            g = df[df.partition == part]
            a.scatter(g[f"{m}_clf"], g[f"{m}_hmm"], s=22, c=c, label=part, alpha=0.8)
        lo = min(df[f"{m}_clf"].min(), df[f"{m}_hmm"].min()) - 0.02
        a.plot([lo, 1], [lo, 1], "k--", lw=0.8)
        a.set(xlabel="classifier only", ylabel="classifier + HMM", title=ttl)
    ax[0].legend()
    fig.suptitle("HMM vs classifier-only per trip (above diagonal = HMM better; "
                 "train scores are in-sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compare_map(trip_dir, params, out):
    """Folium map: GPS, reference route, classifier-only and HMM routes (viz.py conventions)."""
    from .viz_pred import compare_map as cm

    trip, scored, canon, _ = prepare(trip_dir)
    top = scored.loc[scored.groupby("obs_idx").p.idxmax()].sort_values("obs_idx")
    cm(trip, canon[trip["route"].arc_id.to_numpy()], top[["obs_idx", "arc_id"]],
       decode(trip, scored, params), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-tune", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / "hmm_config.json"
    if a.skip_tune and cfg_path.exists():
        params, log = json.loads(cfg_path.read_text())["params"], None
    else:
        params, log = tune(a.trials, a.jobs)
    note = ("PathSnap HMM v1: frozen after random search (defaults + "
            f"{a.trials} random configs, seed 0) over train+student_test trips with the logreg "
            "scorer; objective = mean per-trip (top1 + seq_f1)/2. Search space: "
            "pathsnap.hmm_eval.SPACE.")
    cfg = {"version": "hmm-v1", "note": note, "params": params}
    if log is not None:
        for p in (cfg_path, ARTIFACTS / "hmm_config.json"):
            p.write_text(json.dumps(cfg, indent=2))
        (OUT / "tuning_log.json").write_text(json.dumps(log, indent=2))
    df = evaluate(params, a.jobs)
    df["d_top1"] = df.top1_hmm - df.top1_clf
    df["d_f1"] = df.f1_hmm - df.f1_clf
    df["d_cont"] = df.cont_hmm - df.cont_clf
    df.to_csv(OUT / "comparison.csv", index=False)
    summ = summarise(df)
    summ["tuning"] = {"trials_incl_default": None if log is None else len(log),
                      "search_space": SPACE, "scorer_note": "train scores are in-sample"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    plot(df, OUT / "hmm_vs_classifier.png")
    st = df[df.partition == "student_test"].sort_values("trip_id")
    compare_map(STUDENT_TEST_DIR / st.trip_id.iloc[0], params, RESULTS / "viz" / "hmm_compare.html")
    print(json.dumps(summ["student_test"], indent=1))
    print(json.dumps(summ["train"], indent=1))
    hurt = df[(df.d_top1 < -1e-9) | (df.d_f1 < -1e-9)]
    print("HMM hurts on:\n", hurt[["partition", "trip_id", "d_top1", "d_f1", "d_cont"]])


if __name__ == "__main__":
    main()
