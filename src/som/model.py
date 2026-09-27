"""Kohonen Self-Organising Map.

Same algorithm as the original. The per-node Python loop in the
original is replaced by one numpy broadcast over the whole grid, and the
function is wrapped in a small, testable, serialisable class.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SOMConfig:
    """Hyper-parameters. Defaults match the original.

    width, height:   grid size
    n_epochs:        passes over the data ("n_max_iterations" in the original)
    learning_rate:   alpha_0
    initial_radius:  sigma_0; defaults to max(width, height) / 2
    radius_cutoff:   if True, nodes further than sigma_t from the BMU are not
                     updated at all (what the written spec says). If False, every
                     node gets a Gaussian-weighted update (what the original does).
    seed:            RNG seed for reproducible weight initialisation
    """

    width: int
    height: int
    n_epochs: int = 100
    learning_rate: float = 0.1
    initial_radius: float | None = None
    radius_cutoff: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be >= 1")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.initial_radius is not None and self.initial_radius <= 0:
            raise ValueError("initial_radius must be > 0")
        if self.sigma_0 <= 1:
            # lambda = n / ln(sigma_0). ln(1) = 0 -> division by zero;
            # ln(<1) < 0 -> decay runs backwards.
            raise ValueError(
                "initial_radius (default max(width, height)/2) must be > 1: "
                "use a grid with a side > 2, or set initial_radius explicitly"
            )

    @property
    def sigma_0(self) -> float:
        if self.initial_radius is not None:
            return self.initial_radius
        return max(self.width, self.height) / 2

    @property
    def time_constant(self) -> float:
        """lambda. Chosen so sigma decays to exactly 1 on the final epoch."""
        return self.n_epochs / np.log(self.sigma_0)


@dataclass
class SOM:
    """Usage:

    som = SOM(SOMConfig(width=10, height=10, n_epochs=100, seed=0))
    som.fit(X)                   # X: (n_samples, n_features)
    som.transform(X)             # -> (n_samples, 2) grid coords of each sample's BMU
    som.quantization_error(X)    # -> float, lower is better
    som.save("model.npz"); SOM.load("model.npz")
    """

    config: SOMConfig
    weights: np.ndarray | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- helpers

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None

    def _require_fitted(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self.weights

    @staticmethod
    def _validate(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
        return X

    def _grid_coords(self) -> np.ndarray:
        """(width, height, 2) array holding each node's (x, y). Computed once.

        This is the answer to the spec's "you will need an approach to x() and y()":
        a node's position on the lattice is simply its index in the weight array.
        """
        xs, ys = np.meshgrid(
            np.arange(self.config.width), np.arange(self.config.height), indexing="ij"
        )
        return np.stack([xs, ys], axis=-1).astype(np.float64)

    def _schedule(self, t: int) -> tuple[float, float]:
        """(sigma_t, alpha_t) for epoch t. Both decay with the same exp(-t/lambda)."""
        decay = np.exp(-t / self.config.time_constant)
        return self.config.sigma_0 * decay, self.config.learning_rate * decay

    # -------------------------------------------------------------------- fit

    def fit(self, X, *, initial_weights: np.ndarray | None = None) -> SOM:
        """Train on X of shape (n_samples, n_features). Returns self.

        initial_weights: optional (width, height, n_features) starting grid.
        Used by the equivalence test to start from the same state as the original.
        """
        X = self._validate(X)
        cfg = self.config
        n_features = X.shape[1]
        rng = np.random.default_rng(cfg.seed)

        if initial_weights is None:
            weights = rng.random((cfg.width, cfg.height, n_features))
        else:
            weights = np.array(initial_weights, dtype=np.float64, copy=True)
            expected = (cfg.width, cfg.height, n_features)
            if weights.shape != expected:
                raise ValueError(f"initial_weights must have shape {expected}, got {weights.shape}")

        coords = self._grid_coords()
        logger.info(
            "fit: grid=%dx%d samples=%d features=%d epochs=%d",
            cfg.width,
            cfg.height,
            X.shape[0],
            n_features,
            cfg.n_epochs,
        )

        for t in range(cfg.n_epochs):
            sigma, alpha = self._schedule(t)
            two_sigma_sq = 2.0 * sigma * sigma

            for v in X:
                # 1. BMU: squared feature-space distance to every node, take the min.
                #    argmin of d^2 == argmin of d, so no sqrt.
                d2_feat = np.sum((weights - v) ** 2, axis=-1)
                bmu = np.unravel_index(np.argmin(d2_feat), d2_feat.shape)

                # 2. Grid distance^2 from every node to the BMU, in one broadcast.
                d2_grid = np.sum((coords - np.asarray(bmu, dtype=np.float64)) ** 2, axis=-1)

                # 3. Gaussian influence of the BMU on every node.
                theta = np.exp(-d2_grid / two_sigma_sq)
                if cfg.radius_cutoff:
                    theta[d2_grid > sigma * sigma] = 0.0

                # 4. Move every node toward v. theta[..., None] broadcasts over features.
                weights += alpha * theta[..., None] * (v - weights)

            if cfg.n_epochs >= 10 and (t + 1) % (cfg.n_epochs // 10) == 0:
                logger.debug("epoch %d/%d sigma=%.3f alpha=%.4f", t + 1, cfg.n_epochs, sigma, alpha)

        self.weights = weights
        return self

    # -------------------------------------------------------------- inference

    def transform(self, X) -> np.ndarray:
        """Grid coordinate (x, y) of each sample's BMU. Shape (n_samples, 2)."""
        weights = self._require_fitted()
        X = self._validate(X)
        n_features = weights.shape[-1]
        if X.shape[1] != n_features:
            raise ValueError(f"expected {n_features} features, got {X.shape[1]}")
        flat = weights.reshape(-1, n_features)  # (N, F)
        d2 = np.sum((X[:, None, :] - flat[None, :, :]) ** 2, axis=-1)  # (n, N)
        idx = np.argmin(d2, axis=1)
        return np.stack(np.unravel_index(idx, (self.config.width, self.config.height)), axis=1)

    def quantization_error(self, X) -> float:
        """Mean distance from each sample to its BMU's weights. Lower is better.

        The standard SOM quality metric. Track it across training runs and on
        fresh data in production to detect drift / decide when to retrain.
        """
        weights = self._require_fitted()
        X = self._validate(X)
        bmu = self.transform(X)
        nearest = weights[bmu[:, 0], bmu[:, 1]]
        return float(np.mean(np.linalg.norm(X - nearest, axis=1)))

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        """Weights + config in one .npz so a service can load exactly what was trained."""
        weights = self._require_fitted()
        cfg = {k: ("" if v is None else v) for k, v in asdict(self.config).items()}
        np.savez(Path(path), weights=weights, **{f"cfg_{k}": np.array(v) for k, v in cfg.items()})

    @classmethod
    def load(cls, path: str | Path) -> SOM:
        with np.load(Path(path), allow_pickle=False) as data:
            raw = {k[4:]: data[k].item() for k in data.files if k.startswith("cfg_")}
            cfg = SOMConfig(
                width=int(raw["width"]),
                height=int(raw["height"]),
                n_epochs=int(raw["n_epochs"]),
                learning_rate=float(raw["learning_rate"]),
                initial_radius=(
                    None if raw["initial_radius"] == "" else float(raw["initial_radius"])
                ),
                radius_cutoff=bool(raw["radius_cutoff"]),
                seed=None if raw["seed"] == "" else int(raw["seed"]),
            )
            return cls(cfg, weights=np.array(data["weights"]))

    # ---------------------------------------------------------- visualisation

    def to_image(self) -> np.ndarray:
        """Weights as an RGB image in [0, 1]. Only valid when n_features == 3."""
        weights = self._require_fitted()
        if weights.shape[-1] != 3:
            raise ValueError("to_image() needs exactly 3 features (RGB)")
        return np.clip(weights, 0.0, 1.0)
