"""Phase 2 EDA runner over all trips. Usage (repo root):
    PYTHONPATH=src python -m pathsnap.eda [--deep]

Writes to results/eda/:
  trip_stats.csv        one row per trip: GPS (2.2), network (2.3), route (2.4) stats
  gaps.csv              every sampling interval > 2 s with its jump distance and implied speed
  pooled_*.csv          pooled distributions (dt, degree, arc length, GPS->nearest-arc distance)
  network_*.png, gps_*.png   pooled figures
then runs the 2.1 inventory (eda_inventory). --deep also writes a per-trip deep dive
(eda_trip) for every trip. Every row carries its partition; evaluator data is never read.
"""
import sys

import numpy as np
import pandas as pd

from .config import RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from .eda_gps import gap_table, gps_summary, point_table
from .eda_inventory import PART_COLOR, SURF, _style
from .eda_network import network_analysis
from .eda_route import route_analysis
from .io import list_trips, load_trip

OUT = RESULTS / "eda"
PARTS = (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR))


def trip_stats(trip: dict, partition: str):
    """Returns (row, gaps, pooled arrays) for one trip."""
    tr = trip["track"]
    row = {"partition": partition, "trip_id": trip["trip_id"]}
    row.update(gps_summary(tr))
    net, aux = network_analysis(trip["nodes"], trip["arcs"], tr)
    row.update(net)
    if "route" in trip and aux:
        r, _ = route_analysis(trip, aux, aux.get("gps_nearest_arc_d"))
        row.update(r)
    pt = point_table(tr)
    pooled = {"dt": pt.dt.to_numpy()[1:], "speed": pt.speed_ms.to_numpy()[1:]}
    if aux:
        pooled.update(indeg=aux["indeg"], outdeg=aux["outdeg"], arc_len=aux["arc_len"],
                      near_arc_d=aux["gps_nearest_arc_d"])
    return row, gap_table(tr, trip["trip_id"], 2).assign(partition=partition), pooled


def pooled_tables(pooled: dict) -> dict:
    """Small CSV summaries of pooled per-element arrays (keeps raw-size arrays out of git)."""
    out = {}
    dt_bins = [0, 1, 2, 3, 5, 10, 30, 60, 600, 3600, np.inf]
    out["pooled_dt"] = pd.DataFrame({
        p: pd.Series(pd.cut(np.concatenate(v["dt"]), dt_bins)).value_counts(sort=False)
        for p, v in pooled.items()})
    deg = []
    for p, v in pooled.items():
        if v["indeg"]:
            i, o = np.concatenate(v["indeg"]), np.concatenate(v["outdeg"])
            deg.append(pd.crosstab(np.minimum(i, 5), np.minimum(o, 5)).stack()
                       .rename("nodes").reset_index().assign(partition=p))
    if deg:
        out["pooled_degree"] = pd.concat(deg).rename(columns={"row_0": "indeg_cap5",
                                                              "col_0": "outdeg_cap5"})
    q = [1, 5, 25, 50, 75, 90, 95, 99, 99.9, 100]
    rows = []
    for p, v in pooled.items():
        for k in ("arc_len", "near_arc_d", "speed"):
            if v[k]:
                x = np.concatenate(v[k])
                x = x[np.isfinite(x)]
                rows.append({"partition": p, "quantity": k, "n": len(x),
                             **{f"p{qq}": np.percentile(x, qq) for qq in q}})
    out["pooled_quantiles"] = pd.DataFrame(rows)
    return out


def make_plots(df: pd.DataFrame, pooled: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = [p for p in ("train", "student_test") if pooled[p]["dt"]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURF)
    # dt and GPS->nearest-arc distance (pooled, density so partitions are comparable)
    for p in parts:
        v = pooled[p]
        dt = np.concatenate(v["dt"])
        ax[0, 0].hist(dt[dt > 0], bins=np.logspace(0, 5.5, 45), density=True, histtype="step",
                      lw=2, color=PART_COLOR[p], label=p)
        if v["near_arc_d"]:
            d = np.concatenate(v["near_arc_d"])
            ax[0, 1].hist(np.clip(d, 0, 100), bins=50, density=True, histtype="step", lw=2,
                          color=PART_COLOR[p], label=p)
        if v["arc_len"]:
            L = np.concatenate(v["arc_len"])
            ax[1, 0].hist(L[L > 0], bins=np.logspace(-1, 4, 50), density=True,
                          histtype="step", lw=2, color=PART_COLOR[p], label=p)
            o = np.concatenate(v["outdeg"])
            ax[1, 1].bar(np.arange(7) + (0.2 if p == "student_test" else -0.2),
                         np.bincount(np.minimum(o, 6), minlength=7)[:7] / len(o), width=0.4,
                         color=PART_COLOR[p], label=p)
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_yscale("log")
    _style(ax[0, 0], "Sampling interval dt (s), pooled")
    _style(ax[0, 1], "GPS -> nearest road arc distance (m, clipped 100)")
    ax[1, 0].set_xscale("log")
    _style(ax[1, 0], "Road arc length (m), pooled")
    _style(ax[1, 1], "Node out-degree (6 = 6+), share of nodes")
    for a in ax.ravel():
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pooled_distributions.png", dpi=130, facecolor=SURF)
    plt.close(fig)


def run(deep: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, gaps = [], []
    pooled = {p: {k: [] for k in ("dt", "speed", "indeg", "outdeg", "arc_len", "near_arc_d")}
              for p, _ in PARTS}
    for part, d in PARTS:
        if not d.exists():
            print(f"skip {part}: {d} not found")
            continue
        for td in list_trips(d):
            trip = load_trip(td)
            row, g, pl = trip_stats(trip, part)
            rows.append(row)
            gaps.append(g)
            for k, v in pl.items():
                pooled[part][k].append(v)
            if deep:
                from .eda_trip import analyse

                trip["source"] = "raw"
                analyse(trip)
            print(part, td.name, flush=True)
    if not rows:
        raise SystemExit("no trips found: put the dataset in data/train and data/student_test")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_stats.csv", index=False)
    pd.concat(gaps).to_csv(OUT / "gaps.csv", index=False)
    for name, t in pooled_tables(pooled).items():
        t.to_csv(OUT / f"{name}.csv")
    make_plots(df, pooled)

    from .eda_inventory import run as inventory

    inventory()


if __name__ == "__main__":
    run(deep="--deep" in sys.argv)
