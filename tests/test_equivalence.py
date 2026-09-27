"""The vectorised model must produce the SAME weights as the original.

Both start from identical initial weights and see identical data. If any
refactor changes the maths, this test fails. This is what makes the
performance work safe to ship.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from som import SOM, SOMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "original"))
import kohonen as original  # noqa: E402


def _original_from(init, X, n_iter, width, height):
    """Run the original train() but from a chosen starting grid instead of a fresh random one."""
    np.random.seed(0)
    # Monkeypatch np.random.random so the original's initialisation returns our grid.
    real = np.random.random
    np.random.random = lambda shape: init.copy()
    try:
        return original.train(X, n_iter, width, height)
    finally:
        np.random.random = real


@pytest.mark.parametrize(("width", "height", "n_iter"), [(10, 10, 100), (7, 4, 25), (4, 9, 10)])
def test_vectorised_matches_original(width, height, n_iter):
    rng = np.random.default_rng(123)
    X = rng.random((10, 3))
    init = rng.random((width, height, 3))

    expected = _original_from(init, X, n_iter, width, height)
    got = (
        SOM(SOMConfig(width=width, height=height, n_epochs=n_iter))
        .fit(X, initial_weights=init)
        .weights
    )

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
