"""Phase 2.1: dataset inventory and per-trip data-quality flags.

Input is results/eda/trip_stats.csv (one row per trip, written by `python -m pathsnap.eda`),
so this step re-runs without the raw files. Usage: python -m pathsnap.eda_inventory
"""
import numpy as np
import pandas as pd

from .config import RESULTS

OUT = RESULTS / "eda"
PART_COLOR = {"train": "#2a78d6", "student_test": "#eb6834"}  # categorical slots 1, 2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Fixed, documented thresholds (not tuned on any partition).
FLAGS = {
    "route_gps_mismatch": ("gps_within_30m", lambda x: x < 0.95,
                           "<95% of GPS points within 30 m of the reference route"),
    "long_gap": ("dt_max", lambda x: x > 600, "a sampling gap longer than 10 min"),
    "implausible_speed": ("speed_max_ms", lambda x: x > 80,
                          "a step faster than 80 m/s (288 km/h)"),
    "route_break": ("route_connected_frac", lambda x: x < 1,
                    "consecutive route arcs that are not head-to-tail connected"),
    "route_repeats": ("route_repeat_arcs", lambda x: x > 0, "route revisits an arc"),
    "huge_network": ("n_arcs", lambda x: x > 100_000, "more than 100k arcs"),
}


def load_stats(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or OUT / "trip_stats.csv", dtype={"trip_id": str})
    df["gps_per_route_arc"] = df.n_gps / df.route_len
    df["route_arcs_per_km"] = df.route_len / (df.track_len_m / 1000)
    df["mean_speed_ms"] = df.track_len_m / df.duration_s
    df["arcs_per_gps"] = df.n_arcs / df.n_gps
    return df


def flag_table(df: pd.DataFrame) -> pd.DataFrame:
    f = df[["partition", "trip_id"]].copy()
    for name, (col, rule, _) in FLAGS.items():
        f[name] = rule(df[col]).fillna(False).astype(bool)
    f["n_flags"] = f[list(FLAGS)].sum(axis=1)
    return f


def partition_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["n_gps", "route_len", "n_nodes", "n_arcs", "duration_s", "track_len_m",
            "mean_speed_ms", "dt_max", "n_dt_gt1", "gps_per_route_arc", "route_arcs_per_km",
            "weak_components", "dup_arcs", "gps_route_d_median", "gps_route_d_p95",
            "gps_within_30m"]
    rows = []
    for part, g in df.groupby("partition"):
        for c in cols:
            v = g[c].dropna()
            rows.append({"partition": part, "stat": c, "sum": v.sum(), "min": v.min(),
                         "median": v.median(), "mean": v.mean(), "max": v.max()})
    return pd.DataFrame(rows).pivot(index="stat", columns="partition",
                                    values=["sum", "median", "min", "max"]).round(3)


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def make_plots(df: pd.DataFrame, flags: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) trip size distributions (log x: sizes span 2-3 orders of magnitude)
    specs = [("n_gps", "GPS observations per trip"), ("route_len", "Reference-route arcs per trip"),
             ("n_arcs", "Road-network arcs per trip"), ("duration_s", "Trip duration (s)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
    for ax, (c, t) in zip(axes.ravel(), specs):
        bins = np.logspace(np.log10(df[c].min()), np.log10(df[c].max()), 25)
        parts = ["train", "student_test"]
        ax.hist([df.loc[df.partition == p, c] for p in parts], bins=bins, stacked=True,
                color=[PART_COLOR[p] for p in parts], label=parts, edgecolor=SURF, linewidth=1)
        ax.set_xscale("log")
        _style(ax, t)
        ax.set_ylabel("trips", color=INK2, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "inventory_sizes.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # 2) GPS points vs route arcs: how many observations fall on each reference arc
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=SURF)
    for part, g in df.groupby("partition"):
        ax.scatter(g.route_len, g.n_gps, s=36, color=PART_COLOR[part], label=part,
                   edgecolor=SURF, linewidth=1.5, zorder=3)
    lim = [df.route_len.min() * 0.8, df.n_gps.max() * 1.2]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=2)
    ax.text(lim[1] * 0.5, lim[1] * 0.35, "1 GPS point per arc", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference-route arcs", color=INK2, fontsize=8)
    ax.set_ylabel("GPS observations", color=INK2, fontsize=8)
    _style(ax, "GPS observations vs reference-route length")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "inventory_gps_vs_route.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # 3) route/GPS agreement per trip, sorted; the 95% line is the flag threshold
    d = df.sort_values("gps_within_30m").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=SURF)
    ax.bar(d.index, d.gps_within_30m, color=[PART_COLOR[p] for p in d.partition], width=0.8)
    ax.axhline(0.95, color=INK2, lw=1, ls="--")
    for i, r in d[d.gps_within_30m < 0.95].iterrows():
        ax.text(i, r.gps_within_30m + 0.02, r.trip_id[-3:], ha="center", fontsize=7,
                color=INK, rotation=90)
    ax.set_xticks([])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("trips, sorted (blue = train, orange = student_test)", color=INK2, fontsize=8)
    _style(ax, "Share of GPS points within 30 m of the reference route (dashed = 95% flag)")
    fig.tight_layout()
    fig.savefig(OUT / "inventory_route_agreement.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # 4) flag counts
    counts = flags.groupby("partition")[list(FLAGS)].sum().T
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURF)
    y = np.arange(len(counts))
    for k, part in enumerate(["train", "student_test"]):
        if part in counts:
            ax.barh(y + (k - 0.5) * 0.38, counts[part], height=0.36, color=PART_COLOR[part],
                    label=part)
            for yi, v in zip(y, counts[part]):
                ax.text(v + 0.2, yi + (k - 0.5) * 0.38, str(int(v)), va="center", fontsize=7,
                        color=INK)
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    _style(ax, "Trips per data-quality flag")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "inventory_flags.png", dpi=130, facecolor=SURF)
    plt.close(fig)


def run():
    df = load_stats()
    flags = flag_table(df)
    flags.to_csv(OUT / "trip_flags.csv", index=False)
    pt = partition_table(df)
    pt.to_csv(OUT / "partition_comparison.csv")
    make_plots(df, flags)
    print(pt.to_string())
    print(flags[flags.n_flags > 0].to_string(index=False))
    return df, flags


if __name__ == "__main__":
    run()
