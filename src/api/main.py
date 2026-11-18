"""FastAPI backend (runs on Laptop 1). Labels stay server-side."""
from fastapi import FastAPI

app = FastAPI(title="PathSnap API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def list_trips():
    """IDs of the ten evaluation trips. TODO"""
    return {"trips": []}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    """Run full pipeline (candidates -> model -> HMM) and score vs server-side reference. TODO"""
    raise NotImplementedError
