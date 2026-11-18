"""Build the candidate table for a partition: features + is_true label (train/dev only).

Usage: python -m pathsnap.dataset        -> data/processed/{train,student_test}.parquet
                                           results/eda/alignment_quality.csv
Schema (config.KEY_COLS / LABEL_COL): trip_id, obs_idx, arc_id (as candidate_arc_id), features,
is_true. Labels come from labels.align and are used only for training and evaluation.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .candidates import canonical_arc_ids
from .config import DATA, RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .features import build_features
from .io import list_trips, load_trip
from .labels import align

PROCESSED = DATA / "processed"


def trip_table(trip_dir):
    trip = load_trip(trip_dir)
    feats = build_features(trip)
    lab = align(trip)
    canon = canonical_arc_ids(trip["arcs"], len(trip["nodes"]))
    true_arc = canon[lab.arc_id.to_numpy()]
    feats["is_true"] = (feats.arc_id.to_numpy() == true_arc[feats.obs_idx.to_numpy()]).astype(
        np.int8)
    feats.insert(0, "trip_id", trip["trip_id"])
    hit = feats.groupby("obs_idx").is_true.max()
    d = lab.align_dist_m.to_numpy()
    q = {"trip_id": trip["trip_id"], "n_obs": len(lab), "cand_rows": len(feats),
         "miss_rate": float(1 - hit.reindex(range(len(lab)), fill_value=0).mean()),
         "align_within30": float((d <= 30).mean()), "align_p95_m": float(np.percentile(d, 95)),
         "route_arcs_covered": float(lab.route_pos.nunique() / len(trip["route"]))}
    return feats, q


def build_partition(name, directory):
    res = Parallel(n_jobs=6)(delayed(trip_table)(d) for d in list_trips(directory))
    df = pd.concat([r[0] for r in res], ignore_index=True)
    df = df.rename(columns={"arc_id": "candidate_arc_id"})
    q = pd.DataFrame([r[1] for r in res]).assign(partition=name)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    return df, q


def run():
    qs = []
    for name, d in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        df, q = build_partition(name, d)
        qs.append(q)
        print(f"{name}: {len(df):,} candidate rows, {df.trip_id.nunique()} trips, "
              f"positive rate {df.is_true.mean():.3f}, mean miss rate {q.miss_rate.mean():.4f}",
              flush=True)
    out = pd.concat(qs)
    out.to_csv(RESULTS / "eda" / "alignment_quality.csv", index=False)
    print(out.groupby("partition")[["miss_rate", "align_within30", "route_arcs_covered"]]
          .describe().T.round(3).to_string())


if __name__ == "__main__":
    run()
