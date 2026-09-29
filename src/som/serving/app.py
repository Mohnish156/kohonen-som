"""Serving API. Loads a trained artifact at startup; never trains in-process.

SOM_ARTIFACT_DIR=artifacts uvicorn som.serving.app:app
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from som import __version__
from som.serving.schemas import (
    DeployRequest,
    DeployResponse,
    Health,
    MapRequest,
    MapResponse,
    TrainJobRequest,
    TrainJobResponse,
)
from som.store import load_artifact

log = logging.getLogger("som.serving")
# uvicorn configures only its own loggers; make ours visible without clobbering a caller's setup
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s %(message)s")

_state: dict = {"model": None, "meta": None}
_swap_lock = threading.Lock()


def _install(model, meta, source) -> None:
    """Atomically make (model, meta) the served version."""
    with _swap_lock:
        _state["model"], _state["meta"] = model, meta
    log.info(
        "artifact loaded into memory from %s: grid %dx%d, %d features, qe=%.4f, trained %s",
        source,
        model.config.width,
        model.config.height,
        model.weights.shape[-1],
        (meta.get("metrics") or {}).get("quantization_error", float("nan")),
        meta.get("created_at"),
    )


def _load_and_check(artifact_dir: Path):
    """Load an artifact and prove it can answer a request before it is allowed to serve."""
    model, meta = load_artifact(artifact_dir)
    n_features = model.weights.shape[-1]
    probe = np.random.default_rng(0).random((4, n_features))
    model.transform(probe)  # raises if the artifact is broken
    return model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = Path(os.environ.get("SOM_ARTIFACT_DIR", "artifacts"))
    _state["model"], _state["meta"] = None, None
    if (artifact_dir / "weights.npz").exists():
        _install(*_load_and_check(artifact_dir), artifact_dir)
    else:
        log.warning("no artifact at %s; /map will return 503 until one exists", artifact_dir)
    yield


app = FastAPI(
    title="som",
    version=__version__,
    description=(
        "Kohonen Self-Organising Map, served. Train with `som-train`, "
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


@app.post("/train", response_model=TrainJobResponse, status_code=202)
def train(req: TrainJobRequest) -> TrainJobResponse:
    """Trigger a training run. STUB.

    In production this submits `som-train ...` as a job (Vertex custom
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


@app.post("/deploy", response_model=DeployResponse, status_code=202)
async def deploy(req: DeployRequest) -> DeployResponse:
    """Load a new artifact in the background and swap to it once it passes a self-check.

    Returns immediately. The current model keeps serving until the swap. If the new
    artifact is missing or broken, the error is logged and nothing changes.
    """
    artifact_dir = Path(req.artifact_dir)
    if not (artifact_dir / "weights.npz").exists():
        raise HTTPException(status_code=404, detail=f"no weights.npz in {artifact_dir}")
    log.info("deploy: loading artifact from %s", artifact_dir)

    async def _bg() -> None:
        try:
            model, meta = await asyncio.to_thread(_load_and_check, artifact_dir)
            _install(model, meta, artifact_dir)
            log.info("deploy: swapped to model created_at=%s", meta.get("created_at"))
        except Exception:  # noqa: BLE001
            log.exception("deploy: failed to load %s; previous model still serving", artifact_dir)

    asyncio.get_running_loop().create_task(_bg())
    return DeployResponse(
        status="loading",
        artifact_dir=str(artifact_dir),
        detail="loading in the background; /health shows the new version once swapped",
    )
