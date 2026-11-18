"""Phase 2.2: GPS trajectory exploration (per trip).

All distances are metres (haversine), times are seconds. Nothing here is fitted, so it is
safe to run on any partition; thresholds are fixed constants, not tuned on data.
"""
import numpy as np
import pandas as pd

from .geo import bearing_deg, haversine_m, make_projector

GAP_THRESHOLDS_S = (1, 2, 5, 10, 60, 600)   # interval buckets reported per trip
SEGMENT_GAP_S = 60                       # gap that splits a trip into segments (EDA default)
MAX_PLAUSIBLE_SPEED_MS = 55.0            # ~200 km/h; above this a step is physically suspect
STATIONARY_SPEED_MS = 0.5


def turn_deg(b1, b2):
    """Signed smallest angle from bearing b1 to b2, in (-180, 180]."""
    d = (np.asarray(b2) - np.asarray(b1) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def point_table(track: pd.DataFrame) -> pd.DataFrame:
    """Per-observation movement features. Row i describes the step from i-1 to i
    (first row is NaN). These are inference-safe: they only use the GPS track."""
    lon, lat, t = track.lon.to_numpy(), track.lat.to_numpy(), track.t.to_numpy(float)
    n = len(track)
    out = pd.DataFrame(index=track.index)
    dt = np.full(n, np.nan)
    step = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(t)
        step[1:] = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    out["dt"] = dt
    out["step_m"] = step
    with np.errstate(divide="ignore", invalid="ignore"):
        out["speed_ms"] = np.where(dt > 0, step / dt, np.nan)
    proj = make_projector(lon.mean(), lat.mean())
    x, y = proj(lon, lat)
    b = np.full(n, np.nan)
    if n > 1:
        b[1:] = bearing_deg(x[:-1], y[:-1], x[1:], y[1:])
        b[1:][step[1:] < 1.0] = np.nan  # bearing is noise when barely moving
    out["bearing"] = b
    tr = np.full(n, np.nan)
    if n > 2:
        tr[2:] = turn_deg(b[1:-1], b[2:])
    out["turn"] = tr
    return out


def segment_ids(dt: np.ndarray, max_gap_s: float = SEGMENT_GAP_S) -> np.ndarray:
    """Segment index per observation; a new segment starts after any gap > max_gap_s."""
    dt = np.nan_to_num(np.asarray(dt, float), nan=0.0)
    return np.cumsum(dt > max_gap_s)


def spike_mask(pt: pd.DataFrame, v_max: float = MAX_PLAUSIBLE_SPEED_MS) -> np.ndarray:
    """Out-and-back outliers: implausible speed into AND out of a point with a near
    reversal of direction. Marks the point itself (not its neighbours)."""
    v = pt.speed_ms.to_numpy()
    turn = np.abs(pt.turn.to_numpy())
    m = np.zeros(len(pt), bool)
    if len(pt) > 2:
        v_in, v_out, rev = v[1:-1], v[2:], turn[2:] > 135
        m[1:-1] = (v_in > v_max) & (v_out > v_max) & rev
    return m


def gps_summary(track: pd.DataFrame) -> dict:
    pt = point_table(track)
    dt = pt.dt.to_numpy()[1:]
    v = pt.speed_ms.to_numpy()[1:]
    s = {"n_gps": len(track)}
    s["gps_nan"] = int(track[["lon", "lat", "t"]].isna().sum().sum())
    s["lonlat_invalid"] = int(((track.lon.abs() > 180) | (track.lat.abs() > 90)).sum())
    s["lonlat_zero"] = int(((track.lon == 0) & (track.lat == 0)).sum())
    s["dup_rows"] = int(track.duplicated().sum())
    s["dup_timestamps"] = int(track.t.duplicated().sum())
    s["t_monotonic"] = bool((dt > 0).all()) if len(dt) else True
    s["n_dt_le0"] = int((dt <= 0).sum())
    if len(dt):
        for q in (50, 90, 99):
            s[f"dt_p{q}"] = float(np.percentile(dt, q))
        s["dt_max"] = float(dt.max())
        for g in GAP_THRESHOLDS_S:
            s[f"n_dt_gt{g}"] = int((dt > g).sum())
        s["time_in_gaps_gt60_frac"] = float(dt[dt > 60].sum() / max(dt.sum(), 1e-9))
    s["n_segments_gap60"] = int(segment_ids(pt.dt.to_numpy(), 60)[-1] + 1) if len(pt) else 0
    s["n_segments_gap600"] = int(segment_ids(pt.dt.to_numpy(), 600)[-1] + 1) if len(pt) else 0
    s["duration_s"] = float(track.t.iloc[-1] - track.t.iloc[0]) if len(track) else 0.0
    s["track_len_m"] = float(np.nansum(pt.step_m))
    s["zero_step_frac"] = float(np.mean(pt.step_m.to_numpy()[1:] == 0)) if len(dt) else 0.0
    vv = v[np.isfinite(v)]
    if len(vv):
        for q in (50, 95, 99):
            s[f"speed_p{q}_ms"] = float(np.percentile(vv, q))
        s["speed_max_ms"] = float(vv.max())
        s["n_speed_gt_max"] = int((vv > MAX_PLAUSIBLE_SPEED_MS).sum())
        s["stationary_frac"] = float((vv < STATIONARY_SPEED_MS).mean())
    s["n_spikes"] = int(spike_mask(pt).sum())
    turns = np.abs(pt.turn.to_numpy())
    turns = turns[np.isfinite(turns)]
    s["turn_abs_median"] = float(np.median(turns)) if len(turns) else np.nan
    s["n_sharp_turns_gt90"] = int((turns > 90).sum())
    s["lon_min"], s["lon_max"] = float(track.lon.min()), float(track.lon.max())
    s["lat_min"], s["lat_max"] = float(track.lat.min()), float(track.lat.max())
    s["bbox_w_km"] = float(haversine_m(s["lon_min"], track.lat.mean(), s["lon_max"],
                                       track.lat.mean()) / 1000)
    s["bbox_h_km"] = float(haversine_m(track.lon.mean(), s["lat_min"], track.lon.mean(),
                                       s["lat_max"]) / 1000)
    return s


def gap_table(track: pd.DataFrame, trip_id: str, min_gap_s: float = 10) -> pd.DataFrame:
    """One row per gap > min_gap_s: where it is, how long, and how far the vehicle moved.
    implied_speed tells whether the gap is a stop (low) or a recording dropout while
    driving (normal driving speed)."""
    pt = point_table(track)
    idx = np.where(pt.dt.to_numpy() > min_gap_s)[0]
    return pd.DataFrame({
        "trip_id": trip_id, "obs_idx": idx, "t_start": track.t.to_numpy()[idx - 1],
        "gap_s": pt.dt.to_numpy()[idx], "jump_m": pt.step_m.to_numpy()[idx],
        "implied_speed_ms": pt.speed_ms.to_numpy()[idx]})
