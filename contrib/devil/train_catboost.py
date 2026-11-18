#!/usr/bin/env python3
"""Train/evaluate a CatBoost candidate scorer with trip-grouped CV.

Expected candidate table (one row per GPS-observation/road-arc candidate):
  trip_id, gps_index, arc_id, label, <candidate features...>

label must be 1 for a validated positive candidate and 0 for a negative.
Do not include route-derived distances/positions or target/alignment fields as
predictors. The three identifiers and label are never used as model features.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

REQUIRED = {"trip_id", "gps_index", "arc_id", "label"}
ID_COLUMNS = {"trip_id", "gps_index", "arc_id", "candidate_id", "observation_id", "label"}
LEAKAGE_TOKENS = ("reference", "route", "ground_truth", "truth", "aligned", "target", "matched_label")


def metric_or_none(fn, y, p):
    try:
        return float(fn(y, p))
    except Exception:
        return None


def group_ranking_metrics(frame: pd.DataFrame, score_col: str = "probability") -> dict:
    group_cols = ["trip_id", "gps_index"]
    top1, hit5 = [], []
    for _, group in frame.groupby(group_cols, sort=False):
        group = group.sort_values(score_col, ascending=False)
        y = group["label"].to_numpy()
        if not np.any(y == 1):
            continue
        top1.append(int(y[0] == 1))
        hit5.append(int(np.any(y[:5] == 1)))
    return {
        "gps_groups_with_positive": int(len(top1)),
        "top1_accuracy_by_gps_group": float(np.mean(top1)) if top1 else None,
        "hit_at_5_by_gps_group": float(np.mean(hit5)) if hit5 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Validated candidate-level training CSV/CSV.GZ")
    ap.add_argument("--output-dir", default="outputs/catboost")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--threads", type=int, default=-1)
    args = ap.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"Candidate training table not found: {source}\nCreate and validate it after GPS-to-route alignment and candidate-recall checks.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")
    if df.empty:
        raise SystemExit("Candidate table is empty.")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    if df["label"].isna().any() or not set(df["label"].unique()).issubset({0, 1}):
        raise SystemExit("label must contain only non-null binary values 0 and 1.")
    df["trip_id"] = df["trip_id"].astype(str)
    df["gps_index"] = pd.to_numeric(df["gps_index"], errors="raise").astype("int64")
    if df["trip_id"].nunique() < args.folds:
        raise SystemExit(f"Need at least {args.folds} distinct training trips for GroupKFold; found {df['trip_id'].nunique()}.")
    if df["label"].nunique() != 2:
        raise SystemExit("Training requires both positive and negative candidate labels.")

    # Do not silently use labels, IDs or route/reference-derived data as predictors.
    excluded = []
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if col in ID_COLUMNS or lower.endswith("_id") or any(token in lower for token in LEAKAGE_TOKENS):
            if col not in {"label"}:
                excluded.append(col)
            continue
        candidates.append(col)
    if not candidates:
        raise SystemExit("No safe predictor columns remain after excluding identifiers and leakage-like columns.")

    # Drop all-null and constant predictors; CatBoost handles remaining numeric NaNs.
    features = []
    for col in candidates:
        if df[col].isna().all() or df[col].nunique(dropna=True) <= 1:
            excluded.append(col)
        else:
            features.append(col)
    if not features:
        raise SystemExit("No non-constant candidate features remain.")

    X = df[features].copy()
    categorical = []
    for col in features:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(str)
            categorical.append(col)
        elif pd.api.types.is_object_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].fillna("__MISSING__").astype(str)
            categorical.append(col)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int).to_numpy()
    groups = df["trip_id"].to_numpy()

    # Confirm each GPS group has a positive candidate; this is candidate coverage,
    # not proof that the alignment labels themselves are correct.
    group_status = df.groupby(["trip_id", "gps_index"])["label"].max()
    positive_groups = int((group_status == 1).sum())
    total_groups = int(len(group_status))
    if positive_groups < total_groups:
        print(f"WARNING: {total_groups-positive_groups:,}/{total_groups:,} GPS groups have no positive candidate.")

    gkf = GroupKFold(n_splits=args.folds)
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows, importance_rows = [], []
    best_iterations = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            raise SystemExit(f"Fold {fold} contains only one class in train or validation; inspect trip-level labels/splits.")
        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            thread_count=args.threads,
            allow_writing_files=False,
            verbose=200,
        )
        train_pool = Pool(Xtr, ytr, cat_features=categorical)
        valid_pool = Pool(Xva, yva, cat_features=categorical)
        model.fit(train_pool, eval_set=valid_pool,
                  early_stopping_rounds=args.early_stopping_rounds,
                  use_best_model=True)
        prob = model.predict_proba(Xva)[:, 1]
        oof[va] = prob
        pred = (prob >= 0.5).astype(int)
        fold_rows.append({
            "fold": fold,
            "train_trips": int(pd.Series(groups[tr]).nunique()),
            "validation_trips": int(pd.Series(groups[va]).nunique()),
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "best_iteration": int(model.get_best_iteration()),
            "roc_auc": metric_or_none(roc_auc_score, yva, prob),
            "pr_auc_average_precision": metric_or_none(average_precision_score, yva, prob),
            "precision_at_0_5": float(precision_score(yva, pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(yva, pred, zero_division=0)),
            "f1_at_0_5": float(f1_score(yva, pred, zero_division=0)),
        })
        importance_rows.append(model.get_feature_importance())
        print(f"Fold {fold}: best_iteration={model.get_best_iteration()} PR-AUC={fold_rows[-1]['pr_auc_average_precision']}")

    if np.isnan(oof).any():
        raise RuntimeError("Some rows did not receive out-of-fold predictions.")
    oof_df = df[["trip_id", "gps_index", "arc_id", "label"]].copy()
    oof_df["probability"] = oof
    oof_df.to_csv(out / "catboost_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_rows).to_csv(out / "catboost_cv_metrics.csv", index=False)

    rank_metrics = group_ranking_metrics(oof_df)
    overall = {
        "rows": int(len(df)),
        "training_trips": int(df.trip_id.nunique()),
        "gps_groups": total_groups,
        "gps_groups_with_positive_candidate": positive_groups,
        "positive_candidate_rows": int((y == 1).sum()),
        "negative_candidate_rows": int((y == 0).sum()),
        "excluded_columns": sorted(set(excluded)),
        "feature_columns": features,
        "categorical_feature_columns": categorical,
        "oof_roc_auc": metric_or_none(roc_auc_score, y, oof),
        "oof_pr_auc_average_precision": metric_or_none(average_precision_score, y, oof),
        "oof_precision_at_0_5": float(precision_score(y, (oof >= .5).astype(int), zero_division=0)),
        "oof_recall_at_0_5": float(recall_score(y, (oof >= .5).astype(int), zero_division=0)),
        "oof_f1_at_0_5": float(f1_score(y, (oof >= .5).astype(int), zero_division=0)),
        **rank_metrics,
        "cv_folds": args.folds,
        "best_iteration_median": int(np.median([max(1, x["best_iteration"]) for x in fold_rows])),
        "warning": "Metrics are valid only if candidate labels came from reviewed GPS-to-route alignment. Candidate recall must be reported separately before model metrics are trusted."
    }
    (out / "catboost_training_summary.json").write_text(json.dumps(overall, indent=2))

    mean_importance = np.mean(np.vstack(importance_rows), axis=0)
    pd.DataFrame({"feature": features, "mean_fold_importance": mean_importance}).sort_values(
        "mean_fold_importance", ascending=False).to_csv(out / "catboost_feature_importance.csv", index=False)

    # Fit a deployment model on all labelled training trips using the median best iteration.
    final_iterations = max(1, overall["best_iteration_median"])
    final_model = CatBoostClassifier(
        iterations=final_iterations, depth=args.depth, learning_rate=args.learning_rate,
        loss_function="Logloss", eval_metric="PRAUC", auto_class_weights="Balanced",
        l2_leaf_reg=5.0, random_seed=42, thread_count=args.threads,
        allow_writing_files=False, verbose=200,
    )
    final_model.fit(Pool(X, y, cat_features=categorical))
    final_model.save_model(str(out / "catboost_candidate_scorer.cbm"))
    metadata = {"feature_columns": features, "categorical_feature_columns": categorical,
                "excluded_columns": sorted(set(excluded)), "final_iterations": final_iterations,
                "id_columns": ["trip_id", "gps_index", "arc_id"], "label_column": "label"}
    (out / "catboost_model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nDone. Outputs:", out.resolve())
    print(json.dumps(overall, indent=2))

if __name__ == "__main__":
    main()
