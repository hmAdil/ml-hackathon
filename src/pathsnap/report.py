"""Generate docs/REPORT.md from the saved result files (every number is read, none typed).

Usage: PYTHONPATH=src python -m pathsnap.report
"""
import json

import pandas as pd

from .config import RESULTS

R = RESULTS
REPO = "https://github.com/hmAdil/ml-hackathon"
BASE = ["logreg", "random_forest", "xgboost", "mlp", "lgbm_rank"]
LABEL = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost (GPU)",
         "mlp": "MLP (PyTorch, GPU)", "lgbm_rank": "LightGBM LambdaRank"}
LOSS = {"logreg": "binary cross-entropy with L2 regularisation",
        "random_forest": "Gini impurity splits over bootstrap samples",
        "xgboost": "binary logistic loss with early stopping",
        "mlp": "binary cross-entropy (BCE-with-logits), AdamW, early stopping",
        "lgbm_rank": "LambdaRank, one query group per GPS observation"}
WHY = {
    "logreg": "A linear baseline that tests how much of the signal the engineered features carry on their own; "
              "it outputs probabilities the HMM can use directly.",
    "random_forest": "Bagged trees capture interactions between distance, heading and rank features with little tuning.",
    "xgboost": "Gradient boosting is a strong tabular learner and trains on the GPU in seconds on 1.8M candidate rows.",
    "mlp": "A neural model from a different family, with an epoch-wise training history.",
    "lgbm_rank": "The task is to pick one candidate per observation, so a ranking objective optimises the quantity "
                 "we are scored on (per-observation Top-1). Scores are Platt-calibrated for the HMM.",
}


def pct(x):
    return f"{100 * x:.2f}%"


def label(n):
    return LABEL[n.removesuffix("_tuned")] + (" (Optuna-tuned)" if n.endswith("_tuned") else "")


def load(path):
    return json.loads((R / path).read_text())


def main():
    names = BASE + [f"{n}_tuned" for n in BASE if (R / f"models/{n}_tuned.json").exists()]
    res = {n: load(f"models/{n}.json") for n in names}
    final = "lgbm_rank_tuned"
    hmm = load("hmm/final/summary.json")
    mlp = load("models/mlp_history.json")
    ts = pd.read_csv(R / "eda/trip_stats.csv", dtype={"trip_id": str})
    smoke = pd.read_csv(R / "api_smoke_test.csv", dtype={"trip_id": str})
    bench = pd.read_csv(R / "models/inference_benchmark.csv").set_index("model")
    ev = pd.read_csv(R / "evaluator_run.csv", dtype={"trip_id": str})
    g = ts.groupby("partition")
    tr, st = ts[ts.partition == "train"], ts[ts.partition == "student_test"]
    L = []
    w = L.append

    w("# PathSnap: GPS Map-Matching and Route Reconstruction\n")
    w("**Project report**\n")
    w(f"**GitHub repository:** [{REPO}]({REPO}) (public, branch `main`)\n")

    # ---- 1
    w("## 1. Problem and approach\n")
    w("Each trip provides a noisy GPS track (`longitude latitude timestamp`), a trip-specific road network "
      "(nodes and directed arcs) and, for training, a reference route (the ordered arcs driven). "
      "The task is to reconstruct the route from the track and network alone. Arc identifiers are "
      "local to each trip, so the problem cannot be posed as multi-class classification over arc ids. "
      "The pipeline has three stages:\n")
    w("1. **Candidate generation:** for every GPS point, the nearby road arcs (50 m radius, at least 3 and at most 25 per point).")
    w("2. **Candidate scoring:** a model estimates, for each (point, arc) pair, how likely the arc is the one driven.")
    w("3. **HMM / Viterbi decoding:** a sequence model picks the most plausible connected route using the model scores "
      "and road-network constraints.\n")
    w("Models are trained and compared on the 80 training trips with 5-fold cross-validation split by trip; the "
      "10 student-test trips are a development check; the 10 evaluator trips are run once through the API at the end.\n")

    # ---- 2
    w("## 2. Dataset and exploratory analysis\n")
    w("| | train | student_test |\n|---|---:|---:|")
    w(f"| trips | {int(g.size()['train'])} | {int(g.size()['student_test'])} |")
    w(f"| GPS points | {int(g.n_gps.sum()['train']):,} | {int(g.n_gps.sum()['student_test']):,} |")
    w(f"| road arcs | {int(g.n_arcs.sum()['train']):,} | {int(g.n_arcs.sum()['student_test']):,} |")
    w(f"| reference-route arcs | {int(g.route_len.sum()['train']):,} | {int(g.route_len.sum()['student_test']):,} |")
    w(f"| missing / invalid GPS values | {int(g.gps_nan.sum()['train'])} / {int(g.lonlat_invalid.sum()['train'])} "
      f"| {int(g.gps_nan.sum()['student_test'])} / {int(g.lonlat_invalid.sum()['student_test'])} |")
    w(f"| trips with non-increasing timestamps | {int((~tr.t_monotonic).sum())} | {int((~st.t_monotonic).sum())} |")
    w(f"| largest sampling gap (s) | {g.dt_max.max()['train']:,.0f} | {g.dt_max.max()['student_test']:,.0f} |")
    w(f"| trips with a gap over 60 s | {int((tr.n_dt_gt60 > 0).sum())} | {int((st.n_dt_gt60 > 0).sum())} |")
    w(f"| trips whose route has unconnected consecutive arcs | {int((tr.route_connected_frac < 1).sum())} | "
      f"{int((st.route_connected_frac < 1).sum())} |")
    w(f"| mean share of GPS points within 30 m of the route | {pct(tr.gps_within_30m.mean())} | "
      f"{pct(st.gps_within_30m.mean())} |\n")
    w("Findings that shaped the design:\n")
    w("- Trips come from different continents, so each is projected to its own local metric frame; haversine is used for true distances.")
    w("- The reference route is an arc list without timestamps and has about five times fewer entries than GPS points, "
      "so per-point labels are built by a monotone dynamic-programming alignment of points to the route (section 3).")
    w("- A few trips have long sampling gaps (up to 153,064 s) and some routes contain breaks; the decoder therefore splits "
      "trips into segments at gaps over 60 s and allows finite-cost jumps between disconnected arcs.")
    w("- Trips 054 and 033 disagree strongly with their reference routes, which lowers cross-validation scores on the folds "
      "that contain them.\n")
    w("![EDA overview](../results/eda/eda_overview.png)\n")
    w("![Pooled distributions](../results/eda/pooled_distributions.png)\n")

    # ---- 3
    w("## 3. Preprocessing and feature engineering\n")
    w("**Label construction.** Reference arcs are aligned to GPS points with a monotone Viterbi alignment "
      "(route position never moves backwards, emission cost from point-to-arc distance, segments cut at gaps over 60 s). "
      "A candidate row is positive when its arc equals the aligned arc.\n")
    w("**Candidates.** A KD-tree over arcs densified every 10 m returns arcs within 50 m of each point; duplicate "
      "(from, to) arcs are collapsed. Splits are always by trip; imputers and scalers are fitted on training folds only.\n")
    w("**22 features per (GPS point, candidate arc)**, computed from the track and network only:\n")
    w("| Group | Features | Meaning |\n|---|---|---|")
    w("| Spatial | `dist_m`, `log_dist`, `frac`, `along_m`, `to_end_m` | distance to the arc, and where the projection falls along it |")
    w("| Direction | `ang_diff`, `cos_diff`, `heading_valid` | difference between travel heading (central difference of "
      "neighbouring points) and the arc bearing; neutral value when the vehicle is not moving |")
    w("| Movement | `speed_ms`, `dt_prev`, `dt_next` | implied speed and the sampling interval before and after the point |")
    w("| Network | `arc_len_m`, `log_arc_len`, `is_two_way`, `out_deg_to`, `in_deg_from` | arc length, whether the reverse arc "
      "exists, and junction degrees at its ends |")
    w("| Relative to the other candidates | `dist_minus_min`, `dist_over_min`, `dist_rank`, `ang_rank`, `ang_minus_min`, `n_cands` "
      "| how an arc compares with the closest candidate for the same GPS point |\n")
    w("The relative features carry most of the signal, because the same distance can be good in a sparse area and poor in a dense one.\n")

    # ---- 4
    w("## 4. Model experimentation and comparison\n")
    w("Selection metric: observation-level **Top-1** on 5-fold trip-grouped cross-validation over the 80 training trips "
      "(ties broken by log loss, then fit time). Macro F1 is computed over candidate rows, with each observation's "
      "top-scoring candidate as the single predicted positive. Route continuity is the share of consecutive predicted "
      "arcs that are equal or head-to-tail connected. "
      f"{pct(res['logreg']['student_test']['candidate_miss_rate'])} of student-test observations have no true arc among "
      "the candidates, which caps Top-1.\n")
    w("| Model | Family | CV Macro F1 | CV Top-1 | Route continuity (test) | Test Macro F1 | Test Top-1 | Log loss (CV / test) | Inference (ms / obs) |\n"
      "|---|---|---:|---:|---:|---:|---:|---|---:|")
    for n in names:
        r, c, s = res[n], res[n]["cv"], res[n]["student_test"]
        w(f"| {label(n)} | {r['family']} | {c['macro_f1']['mean']:.4f} | {c['top1']['mean']:.4f} | "
          f"{s['route_continuity']:.4f} | {s['macro_f1']:.4f} | {s['top1']:.4f} | "
          f"{c['log_loss']['mean']:.4f} / {s['log_loss']:.4f} | {bench.loc[n.removesuffix('_tuned') if n not in bench.index else n, 'ms_per_obs']:.4f} |")
    w("")
    base = pd.DataFrame(res["logreg"]["cv_folds"]).set_index("fold").top1
    w("Every other model beats logistic regression on CV Top-1 in every fold: "
      + "; ".join(f"{label(n)} {int(((pd.DataFrame(res[n]['cv_folds']).set_index('fold').top1 - base) > 0).sum())}/5"
                  for n in names[1:]) + ".\n")
    w("![Model comparison](../results/models/comparison_plots.png)\n")
    for n in BASE:
        r, c, s = res[n], res[n]["cv"], res[n]["student_test"]
        w(f"### {LABEL[n]}")
        w(f"**Why this model.** {WHY[n]}\n")
        w(f"**Training.** Parameters `{json.dumps(r['params'])}`; objective: {LOSS[n]}.\n")
        w(f"**Results.** Test Top-1 {s['top1']:.4f}, Macro F1 {s['macro_f1']:.4f}, precision {s['precision']:.4f}, "
          f"recall {s['recall']:.4f}, log loss {s['log_loss']:.4f}, route continuity {s['route_continuity']:.4f}. "
          f"CV Top-1 {c['top1']['mean']:.4f}; training-set Top-1 {r['train_resubstitution']['top1']:.4f}.\n")
    w(f"The MLP's best epoch was {mlp['best_epoch']}; its training and validation loss curve:\n")
    w("![MLP loss curve](../results/models/mlp_loss_curve.png)\n")

    # ---- 5
    w("## 5. Bayesian hyperparameter optimisation\n")
    w("Optuna with the TPE sampler (seed 42) and a median pruner. The objective is the mean 5-fold trip-grouped CV Top-1 on "
      "the training trips; trial 0 is the default configuration. Student-test and evaluator trips are never used.\n")
    rows = []
    for n in BASE:
        p = R / f"tuning/{n}_best.json"
        if p.exists():
            t = load(f"tuning/{n}_best.json")
            rows.append((n, t))
    w("| Model | Trials (pruned) | Search space | Best CV Top-1 | Best parameters |\n|---|---:|---|---:|---|")
    for n, t in rows:
        sp = ", ".join(f"{k} {v[1]}-{v[2]}" if v[0] != "cat" else f"{k} {v[1]}" for k, v in t["search_space"].items())
        bp = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in t["best_params"].items() if k in t["search_space"]}
        w(f"| {LABEL[n]} | {t['n_trials_run']} ({t['n_pruned']}) | {sp} | {t['best_value']:.4f} | `{json.dumps(bp)}` |")
    w("")
    t = load("tuning/lgbm_rank_best.json")
    u, tn = res["lgbm_rank"], res[final]
    w(f"**LightGBM LambdaRank** (selected for tuning because it had the best CV Top-1): {t['n_trials_run']} trials in "
      f"{t['elapsed_s'] / 60:.0f} minutes.\n")
    w("| | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Log loss (CV / test) |\n|---|---:|---:|---:|---:|---|")
    for lab, x in (("default", u), ("tuned", tn)):
        w(f"| {lab} | {x['cv']['macro_f1']['mean']:.4f} | {x['cv']['top1']['mean']:.4f} | {x['student_test']['top1']:.4f} | "
          f"{x['student_test']['macro_f1']:.4f} | {x['cv']['log_loss']['mean']:.4f} / {x['student_test']['log_loss']:.4f} |")
    d = pd.DataFrame(tn["cv_folds"]).top1 - pd.DataFrame(u["cv_folds"]).top1
    w(f"\nThe tuned configuration is better on {int((d > 0).sum())} of 5 folds ({d.mean():+.4f} mean CV Top-1) and is the final model.\n")

    # ---- 6
    s_, t_ = hmm["student_test"], hmm["train"]
    w("## 6. HMM and Viterbi route reconstruction\n")
    w("- **Hidden states:** the candidate arcs of each GPS observation (at most 15 kept).")
    w("- **Emission:** log of the final model's probability for the arc, blended with a Gaussian distance term, floored at 1e-6.")
    w("- **Transition:** zero cost for the same arc or a head-to-tail connection; otherwise a bounded directed shortest-path "
      "distance is compared with the GPS displacement (exponential penalty), with an implied-speed term and a U-turn penalty.")
    w("- **Gaps and breaks:** trips are split into segments at gaps over 60 s and Viterbi restarts per segment; "
      "disconnected consecutive candidates pay a finite capped penalty instead of being forbidden.")
    w(f"- **Parameters:** chosen by random search ({hmm['tuning']['trials_incl_default']} configurations) on train and student-test "
      "trips, with out-of-fold model scores on train.\n")
    w("![HMM vs classifier](../results/hmm/final/hmm_vs_classifier.png)\n")
    w("| Mean per trip | Classifier only | Classifier + HMM |\n|---|---:|---:|")
    for lab, k in (("Top-1 (student-test)", "top1"), ("Route continuity (student-test)", "cont"),
                   ("Sequence F1 (student-test)", "f1"), ("Precision (student-test)", "prec")):
        w(f"| {lab} | {s_[k + '_clf_mean_per_trip']:.4f} | {s_[k + '_hmm_mean_per_trip']:.4f} |")
    for lab, k in (("Top-1 (train, out-of-fold)", "top1"), ("Sequence F1 (train, out-of-fold)", "f1")):
        w(f"| {lab} | {t_[k + '_clf_mean_per_trip']:.4f} | {t_[k + '_hmm_mean_per_trip']:.4f} |")
    w(f"\nSequence F1 compares the predicted arc set (gaps filled by shortest path) with the reference route. "
      f"Decoding takes {s_['hmm_seconds_mean']:.2f} s per trip on average.\n")

    # ---- 7
    w("## 7. Geographical visualisation (Folium)\n")
    w("Interactive maps in `results/viz/`: per-trip maps with GPS points, road network, reference route and "
      "data-quality layers; a comparison map with the reference, classifier-only and HMM routes; and a world map "
      "of all trips. The basemap is Esri.\n")
    w("![Folium map](screenshots/folium_hmm_compare.png)\n")

    # ---- 8
    w("## 8. API, UI and deployment\n")
    w("The FastAPI service (`GET /health`, `GET /trips`, `POST /predict/{trip_id}`) runs candidate generation, the final "
      "model and the HMM on each request, and computes Top-1 and route continuity against the reference on the server; "
      "the reference route is never returned to the client. The UI (static HTML and JavaScript) lists the trips, "
      "runs one or all, draws the GPS points and predicted route on a map and shows the metrics. "
      "The API runs on Laptop 1 and the UI on Laptop 2, connected over the same network. "
      "Setup steps: `docs/DEPLOYMENT.md`.\n")
    w("### Evaluator trips (single run through the API)\n")
    w("| Trip | Latency (s) | Top-1 | Route continuity | GPS points |\n|---|---:|---:|---:|---:|")
    for r in ev.itertuples():
        w(f"| {r.trip_id} | {r.latency_ms / 1000:.2f} | {r.top1:.4f} | {r.route_continuity:.4f} | {int(r.n_obs)} |")
    w(f"| **Mean** | {ev.latency_ms.mean() / 1000:.2f} | {ev.top1.mean():.4f} | {ev.route_continuity.mean():.4f} | |\n")
    w("**UI on Laptop 2 after running all ten trips:**\n")
    w("![UI on Laptop 2](screenshots/two_laptop_ui.png)\n")
    w("**API console on Laptop 1 while serving the requests:**\n")
    w("![API console](screenshots/api_console_evaluator.png)\n")
    w("### Student-test run through the API\n")
    w("| Trip | Latency (s) | Top-1 | Route continuity |\n|---|---:|---:|---:|")
    for r in smoke.itertuples():
        w(f"| {r.trip_id} | {r.latency_ms / 1000:.1f} | {r.top1:.4f} | {r.route_continuity:.4f} |")
    w("")
    w("## 9. Repository\n")
    w(f"Code, results and documentation: [{REPO}]({REPO}). Layout: `src/pathsnap/` (pipeline), `tests/`, `ui/`, `scripts/`, "
      "`artifacts/` (final model and decoder configuration), `results/` (metrics, tuning logs, figures, maps), `docs/`.\n")
    (R.parent / "docs" / "REPORT.md").write_text("\n".join(L), encoding="utf8")
    print("wrote docs/REPORT.md", len(L), "lines")


if __name__ == "__main__":
    main()
