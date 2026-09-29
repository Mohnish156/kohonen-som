"""Serving API. Loads a trained artifact at startup; never trains in-process.

SOM_ARTIFACT_DIR=artifacts uvicorn som.serving.app:app
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from som import __version__
from som.serving.schemas import Health, MapRequest, MapResponse, TrainJobRequest, TrainJobResponse
from som.store import load_artifact

log = logging.getLogger("som.serving")
# uvicorn configures only its own loggers; make ours visible without clobbering a caller's setup
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s %(message)s")

_state: dict = {"model": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = Path(os.environ.get("SOM_ARTIFACT_DIR", "artifacts"))
    _state["model"], _state["meta"] = None, None
    if (artifact_dir / "weights.npz").exists():
        _state["model"], _state["meta"] = load_artifact(artifact_dir)
        m = _state["model"]
        log.info(
            "artifact loaded into memory from %s: grid %dx%d, %d features, qe=%.4f, trained %s",
            artifact_dir,
            m.config.width,
            m.config.height,
            m.weights.shape[-1],
            (_state["meta"].get("metrics") or {}).get("quantization_error", float("nan")),
            _state["meta"].get("created_at"),
        )
    else:
        log.warning("no artifact at %s; /map will return 503 until one exists", artifact_dir)
    yield


app = FastAPI(
    title="som",
    version=__version__,
    description=(
        "Kohonen Self-Organising Map, served. Train with `python -m som.training`, "
        "then use **/predict** to upload an image and get it back repainted with the "
        "map's palette, or **/image** to see the map itself."
    ),
    lifespan=lifespan,
)

PNG = {200: {"content": {"image/png": {}}, "description": "PNG image"}}


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


def _png(arr01: np.ndarray) -> Response:
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/image", response_class=Response, responses=PNG)
def image(scale: int = 8) -> Response:
    """The trained grid as a PNG, one cell per node coloured by its weights. RGB only."""
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    try:
        grid = model.to_image()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    scale = max(1, min(scale, 64))
    return _png(np.kron(grid, np.ones((scale, scale, 1))))


@app.post("/predict", response_class=Response, responses=PNG)
async def predict(file: UploadFile = File(...), max_side: int = 512) -> Response:  # noqa: B008
    """Upload an image; get it back repainted with only the colours the map learned.

    Every pixel is mapped to its BMU and replaced by that node's weights. This is the
    SOM used as a colour quantiser, and it is the quickest way to see what the map knows.
    """
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    if model.weights.shape[-1] != 3:
        raise HTTPException(status_code=422, detail="loaded model is not RGB")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"not an image: {e}") from e
    img.thumbnail((max_side, max_side))
    px = np.asarray(img, dtype=np.float64) / 255.0
    h, w, _ = px.shape
    out = model.quantize(px.reshape(-1, 3)).reshape(h, w, 3)
    log.info("predict: %s %dx%d -> %d pixels quantised", file.filename, w, h, h * w)
    return _png(out)


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
