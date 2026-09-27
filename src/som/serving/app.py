"""Serving API. Loads a trained artifact at startup; never trains in-process.

SOM_ARTIFACT_DIR=artifacts uvicorn som.serving.app:app
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from som import __version__
from som.serving.schemas import Health, MapRequest, MapResponse, TrainJobRequest, TrainJobResponse
from som.store import load_artifact

log = logging.getLogger("som.serving")

_state: dict = {"model": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = Path(os.environ.get("SOM_ARTIFACT_DIR", "artifacts"))
    _state["model"], _state["meta"] = None, None
    if (artifact_dir / "weights.npz").exists():
        _state["model"], _state["meta"] = load_artifact(artifact_dir)
        log.info("loaded artifact from %s", artifact_dir)
    else:
        log.warning("no artifact at %s; /map will return 503 until one exists", artifact_dir)
    yield


app = FastAPI(title="som", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=Health)
def health() -> Health:
    meta = _state["meta"] or {}
    return Health(
        status="ok",
        version=__version__,
        model_loaded=_state["model"] is not None,
        model_created_at=meta.get("created_at"),
        quantization_error=(meta.get("metrics") or {}).get("quantization_error"),
    )


@app.post("/map", response_model=MapResponse)
def map_points(req: MapRequest) -> MapResponse:
    """Inference: the grid cell each sample lands on."""
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    try:
        coords = model.transform(np.asarray(req.data, dtype=float))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return MapResponse(bmu=coords.tolist())


@app.post("/train-job", response_model=TrainJobResponse, status_code=202)
def train_job(req: TrainJobRequest) -> TrainJobResponse:
    """Trigger a training run. STUB.

    In production this submits `python -m som.training ...` as a job (Vertex custom
    job, k8s Job, Cloud Run job) and returns immediately. The service itself never
    trains: that's minutes of CPU on a request path, can't be retried, doesn't scale.
    """
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    log.info("would submit training job %s with %s", job_id, req.model_dump())
    return TrainJobResponse(
        job_id=job_id,
        status="accepted",
        detail="stub: no job runner wired up. See README 'How I'd productionise it'.",
    )
