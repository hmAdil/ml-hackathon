"""Phase 8: consolidated comparison of every model in results/models/*.json.

Usage: PYTHONPATH=src python -m pathsnap.compare
Writes results/models/comparison.{csv,md}, comparison_plots.png and paired_folds.csv.
Selection rule (docs/PLAN.md): primary metric = grouped-CV observation-level Top-1 (what the
API scores); ties inside fold noise are broken by probability quality (log loss) because the
HMM consumes the probabilities, then by fit time. Student-test is a development check only.
"""
import json

import numpy as np
import pandas as pd

from .config import RESULTS

ORDER = ["logreg", "random_forest", "xgboost", "mlp", "lgbm_rank", "lgbm_rank_tuned"]
OUT = RESULTS / "models"


def load() -> dict:
    return {n: json.loads((OUT / f"{n}.json").read_text()) for n in ORDER
            if (OUT / f"{n}.json").exists()}


def table(res: dict) -> pd.DataFrame:
    rows = []
    for n, r in res.items():
        cv, st = r["cv"], r["student_test"]
        rows.append({
            "model": n, "family": r["family"],
            "cv_top1": cv["top1"]["mean"], "cv_top1_std": cv["top1"]["std"],
            "cv_top1_covered": cv["top1_covered"]["mean"],
            "cv_macro_f1": cv["macro_f1"]["mean"], "cv_log_loss": cv["log_loss"]["mean"],
            "test_top1": st["top1"], "test_macro_f1": st["macro_f1"],
            "test_precision": st["precision"], "test_recall": st["recall"],
            "test_log_loss": st["log_loss"], "test_route_continuity": st["route_continuity"],
            "fit_s": r["fit_s_full_train"], "inference_ms_per_obs": st["inference_ms_per_obs"],
            "train_top1": r["train_resubstitution"]["top1"],
        })
    return pd.DataFrame(rows)


def paired_folds(res: dict) -> pd.DataFrame:
    """Per-fold top-1 minus the logistic-regression baseline on the same folds."""
    base = pd.DataFrame(res["logreg"]["cv_folds"]).set_index("fold").top1
    rows = []
    for n, r in res.items():
        d = pd.DataFrame(r["cv_folds"]).set_index("fold").top1 - base
        rows.append({"model": n, "mean_diff_vs_logreg": d.mean(), "std_diff": d.std(),
                     "folds_better": int((d > 0).sum()), "n_folds": len(d)})
    return pd.DataFrame(rows)


def plots(df: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    x = np.arange(len(df))
    ax[0, 0].bar(x, df.cv_macro_f1, color="#2a78d6")
    ax[0, 0].set_title("Grouped-CV Macro F1 (mean of 5 folds)")
    ax[0, 1].bar(x, df.cv_top1, yerr=df.cv_top1_std, color="#2a78d6", capsize=3)
    ax[0, 1].set_title("Grouped-CV Top-1 (error bar = fold std)")
    w = 0.38
    ax[0, 2].bar(x - w / 2, df.test_top1, w, label="Top-1", color="#008300")
    ax[0, 2].bar(x + w / 2, df.test_macro_f1, w, label="Macro F1", color="#eb6834")
    ax[0, 2].set_title("Student-test Top-1 and Macro F1")
    ax[0, 2].legend()
    ax[1, 0].bar(x - w / 2, df.cv_log_loss, w, label="CV", color="#2a78d6")
    ax[1, 0].bar(x + w / 2, df.test_log_loss, w, label="student-test", color="#008300")
    ax[1, 0].set_title("Log loss (lower is better)")
    ax[1, 0].legend()
    ax[1, 1].bar(x, df.test_route_continuity, color="#e34948")
    ax[1, 1].set_title("Student-test route continuity (classifier only)")
    ax[1, 1].set_ylim(0.85, 0.95)
    ax[1, 2].bar(x, df.fit_s, color="#888888")
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_title("Full-train fit time (s, log scale)")
    for a in ax.ravel():
        a.set_xticks(x)
        a.set_xticklabels(df.model, rotation=20)
    ax[0, 0].set_ylim(0.85, 0.9)
    ax[0, 1].set_ylim(0.6, 0.85)
    ax[0, 2].set_ylim(0.8, 0.95)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_plots.png", dpi=130)


def run():
    res = load()
    df = table(res)
    df.to_csv(OUT / "comparison.csv", index=False)
    pf = paired_folds(res)
    pf.to_csv(OUT / "paired_folds.csv", index=False)
    (OUT / "comparison.md").write_text(df.round(4).to_markdown(index=False))
    plots(df)
    print(df.round(4).to_string(index=False))
    print()
    print(pf.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
