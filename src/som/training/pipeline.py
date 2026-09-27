"""Training pipeline: load -> validate -> fit -> evaluate -> save.

Each step is a plain function so it can be tested alone and so the whole thing
maps 1:1 onto components in an orchestrator (Vertex, Kubeflow, Airflow).
`run()` is what the orchestrator, a cron, or `python -m som.training` calls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from som.model import SOM, SOMConfig
from som.store import save_artifact

log = logging.getLogger(__name__)


def load(source: str | Path | None, *, n_random: int = 10, seed: int | None = None) -> np.ndarray:
    """Read samples x features from a CSV, or generate random RGB points if no source."""
    if source is None:
        log.info("no data source; generating %d random RGB samples", n_random)
        return np.random.default_rng(seed).random((n_random, 3))
    log.info("loading %s", source)
    return np.loadtxt(source, delimiter=",", ndmin=2)


def validate(X: np.ndarray) -> np.ndarray:
    """Reject anything the model can't train on. Same rules as SOM.fit, surfaced early."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"expected (n_samples, n_features) with n_samples > 0, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("data contains NaN or inf")
    return X


def fit(X: np.ndarray, cfg: SOMConfig) -> SOM:
    return SOM(cfg).fit(X)


def evaluate(som: SOM, X: np.ndarray) -> dict:
    return {
        "quantization_error": som.quantization_error(X),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


def save(som: SOM, metrics: dict, dest: str | Path) -> Path:
    return save_artifact(som, metrics, dest)


def run(cfg: SOMConfig, dest: str | Path, *, source: str | Path | None = None) -> dict:
    """The whole pipeline. Returns the metrics that were written to the artifact."""
    t0 = time.perf_counter()
    X = validate(load(source, seed=cfg.seed))
    som = fit(X, cfg)
    metrics = evaluate(som, X)
    metrics["train_seconds"] = round(time.perf_counter() - t0, 3)
    path = save(som, metrics, dest)
    log.info("artifact written to %s: %s", path, metrics)
    return metrics
