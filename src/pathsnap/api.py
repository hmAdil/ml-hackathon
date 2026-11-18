"""FastAPI backend (Laptop 1). Reference labels stay server-side and are never returned.

Flow per request: load trip -> candidates+features -> model scores -> HMM (if available)
-> predicted route -> metrics vs the internal reference route -> JSON for the UI.
"""
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .candidates import canonical_arc_ids
from .config import ARTIFACTS, ROOT
from .features import FEATURES, build_features
from .io import load_trip
from .labels import align
from .metrics import route_continuity

log = logging.getLogger("pathsnap.api")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

STATE: dict = {}
_LOCKS: dict = {}
_GUARD = threading.Lock()


def _trip_lock(trip_id: str) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(trip_id, threading.Lock())


def eval_dir() -> Path:
    p = Path(os.environ.get("PATHSNAP_EVAL_DIR", "data/evaluator"))
    return p if p.is_absolute() else ROOT / p


def trip_ids() -> list[str]:
    d = eval_dir()
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def hmm_available() -> bool:
    try:
        from .hmm import decode  # noqa: F401

        return True
    except ImportError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in ("final_model.joblib", "logreg.joblib"):
        path = ARTIFACTS / name
        if path.exists():
            STATE["model"] = joblib.load(path)
            STATE["model_name"] = name.removesuffix(".joblib")
            meta = ARTIFACTS / "final_model.json"
            if name == "final_model.joblib" and meta.exists():
                STATE["model_name"] = json.loads(meta.read_text()).get("name", "final_model")
            log.info("loaded model %s", path)
            break
    else:
        raise RuntimeError(f"no model artifact found in {ARTIFACTS}")
    cfg = ARTIFACTS / "hmm_config.json"
    STATE["hmm_params"] = json.loads(cfg.read_text()) if cfg.exists() else None
    STATE["use_hmm"] = hmm_available()
    if STATE["use_hmm"]:
        log.info("HMM decoder available (params: %s)", cfg if cfg.exists() else "defaults")
    else:
        log.warning("hmm.decode not available -> falling back to classifier-only (argmax)")
    STATE["cache"] = {}
    log.info("eval dir: %s (%d trips)", eval_dir(), len(trip_ids()))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("model_name"),
            "decoder": "hmm" if STATE.get("use_hmm") else "classifier-only",
            "n_trips": len(trip_ids())}


@app.get("/trips")
def trips():
    return {"trips": trip_ids()}


def _polyline(nodes, arcs, arc_id):
    a = arcs.iloc[int(arc_id)]
    f, t = nodes.iloc[int(a.from_node)], nodes.iloc[int(a.to_node)]
    return [[float(f.lat), float(f.lon)], [float(t.lat), float(t.lon)]]


def _run(trip_id: str) -> dict:
    trip = load_trip(eval_dir() / trip_id)  # route stays server-side
    nodes, arcs, track = trip["nodes"], trip["arcs"], trip["track"]
    use_hmm = STATE["use_hmm"]
    log.info("[%s] request: %d GPS points, %d arcs, decoder=%s", trip_id, len(track), len(arcs),
             "hmm" if use_hmm else "classifier-only")
    t0 = time.perf_counter()
    feats = build_features(trip)
    t1 = time.perf_counter()
    log.info("[%s] step 1 candidates+features: %d candidate rows (%.0f ms)", trip_id, len(feats),
             (t1 - t0) * 1e3)
    feats["p"] = STATE["model"].predict_proba(feats[FEATURES].to_numpy("float32"))[:, 1]
    t2 = time.perf_counter()
    log.info("[%s] step 2 model scoring (%s) (%.0f ms)", trip_id, STATE["model_name"],
             (t2 - t1) * 1e3)
    if use_hmm:
        try:
            from .hmm import decode

            pred = decode(trip, feats, STATE["hmm_params"] or {})
        except Exception:
            log.exception("[%s] HMM failed, falling back to classifier-only", trip_id)
            use_hmm = False
    if not use_hmm:
        top = feats.loc[feats.groupby("obs_idx").p.idxmax()]
        pred = top[["obs_idx", "arc_id"]].reset_index(drop=True)
    t3 = time.perf_counter()
    log.info("[%s] step 3 decode (%s) (%.0f ms)", trip_id, "hmm" if use_hmm else "argmax",
             (t3 - t2) * 1e3)

    pred = pred.sort_values("obs_idx")
    seq = pred.arc_id.to_numpy(np.int64)
    route = [int(a) for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]

    # server-side evaluation against the internal reference
    metrics = {"n_obs": int(len(track)), "n_pred_obs": int(len(pred)),
               "n_route_arcs": len(route),
               "route_continuity": route_continuity(route, arcs.from_node, arcs.to_node)}
    if "route" in trip:
        ref = align(trip)
        canon = canonical_arc_ids(arcs, len(nodes))
        ref_arc = canon[ref.arc_id.to_numpy()]
        m = pred.set_index("obs_idx").arc_id.reindex(ref.obs_idx).to_numpy()
        metrics["top1_accuracy"] = float(np.mean(m == ref_arc))
    else:
        metrics["top1_accuracy"] = None
    log.info("[%s] metrics: top1=%s continuity=%.4f n_obs=%d route_arcs=%d", trip_id,
             metrics["top1_accuracy"], metrics["route_continuity"], metrics["n_obs"], len(route))
    timings = {"candidates_features_ms": round((t1 - t0) * 1e3, 1),
               "scoring_ms": round((t2 - t1) * 1e3, 1),
               "hmm_ms": round((t3 - t2) * 1e3, 1) if use_hmm else 0.0,
               "total_ms": round((t3 - t0) * 1e3, 1)}
    return {
        "trip_id": trip_id,
        "gps": [[float(la), float(lo)] for la, lo in zip(track.lat, track.lon)],
        "route": [{"arc_id": a, "polyline": _polyline(nodes, arcs, a)} for a in route],
        "metrics": metrics,
        "timings_ms": timings,
        "model": STATE["model_name"],
        "decoder": "hmm" if use_hmm else "classifier-only (argmax)",
    }


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in trip_ids():
        log.warning("[%s] unknown trip id", trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip id {trip_id!r}")
    cache = STATE["cache"]
    use_cache = os.environ.get("PATHSNAP_CACHE") == "1"  # off by default: every request re-runs
    with _trip_lock(trip_id):  # several laptops may request the same trip at once
        if use_cache and trip_id in cache:
            log.info("[%s] served from cache", trip_id)
            return cache[trip_id]
        result = _run(trip_id)
        if use_cache:
            cache[trip_id] = result
    return result
