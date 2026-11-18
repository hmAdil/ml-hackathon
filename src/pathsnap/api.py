"""FastAPI backend (Laptop 1). Reference labels stay server-side."""
from fastapi import FastAPI

app = FastAPI(title="PathSnap API")


@app.get("/health")
def health():
    return {"status": "ok"}
