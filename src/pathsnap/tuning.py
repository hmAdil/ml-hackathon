"""Phase 7: Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.

Usage: PYTHONPATH=src python -m pathsnap.tuning lgbm_rank [n_trials]

- Data: the 80 train trips only (never student-test or evaluator trips).
- Objective: mean observation-level Top-1 over the same 5 trip-grouped folds as the baseline
  comparison (cv.cross_validate), so tuned and untuned numbers are comparable.
- Sampler: TPE (seeded); pruner: median pruner on per-fold Top-1.
Writes results/tuning/<model>_trials.csv and <model>_best.json. The tuned model is then
evaluated with `python -m pathsnap.train <model> --params results/tuning/<model>_best.json`.
"""
import importlib
import json
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import RANDOM_SEED, RESULTS
from .cv import evaluate, fit_scorer, score
from .dataset import PROCESSED

# search space per model: {"param": (kind, low, high)}; kinds: int | float | log
SPACES = {
    "lgbm_rank": {
        "num_leaves": ("int", 15, 127),
        "learning_rate": ("log", 0.02, 0.2),
        "n_estimators": ("int", 100, 500),
        "min_child_samples": ("int", 10, 200),
        "subsample": ("float", 0.4, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_lambda": ("log", 1e-3, 10.0),
    },
}


def suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for name, (kind, lo, hi) in space.items():
        if kind == "int":
            out[name] = trial.suggest_int(name, lo, hi)
        elif kind == "log":
            out[name] = trial.suggest_float(name, lo, hi, log=True)
        else:
            out[name] = trial.suggest_float(name, lo, hi)
    return out


def run(name: str, n_trials: int = 20, timeout_s: int = 3000):
    mod = importlib.import_module(f"pathsnap.models.{name}")
    space = SPACES[name]
    train = pd.read_parquet(PROCESSED / "train.parquet")
    folds = list(GroupKFold(n_splits=5).split(train, groups=train.trip_id))
    out_dir = RESULTS / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = {**mod.DEFAULT_PARAMS, **suggest(trial, space)}
        scores = []
        for k, (tri, vai) in enumerate(folds):
            model, _ = fit_scorer(mod.build, params, train.iloc[tri])
            val = train.iloc[vai]
            scores.append(evaluate(val, score(model, val))["top1"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", [float(s) for s in scores])
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1))
    # trial 0 = the untuned default configuration, so the search can only match or beat it
    study.enqueue_trial({k: mod.DEFAULT_PARAMS[k] for k in space if k in mod.DEFAULT_PARAMS})
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    elapsed = time.perf_counter() - t0

    df = study.trials_dataframe()
    df.to_csv(out_dir / f"{name}_trials.csv", index=False)
    best = {**mod.DEFAULT_PARAMS, **study.best_params}
    summary = {"model": name, "objective": "mean grouped-CV top-1 (5 folds, 80 train trips)",
               "sampler": "TPE", "seed": RANDOM_SEED, "search_space": space,
               "n_trials_run": len(study.trials),
               "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
               "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
               "best_value": study.best_value, "best_params": best,
               "trial0_value": study.trials[0].value,
               "trial0_note": "default values of searched params; unlisted params (reg_lambda) "
               "were sampled; exact untuned run: results/models/lgbm_rank.json",
               "elapsed_s": elapsed}
    (out_dir / f"{name}_best.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_trials_run", "n_complete", "n_pruned",
                                              "trial0_value", "best_value", "best_params",
                                              "elapsed_s")}, indent=1))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
