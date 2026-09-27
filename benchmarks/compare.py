"""Original vs vectorised: same data, same initial weights, wall-clock time and output check.

python benchmarks/compare.py            # quick: both notebook cases, big one extrapolated
python benchmarks/compare.py --full     # actually run the 100x100 x 1000 original (~minutes)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "original"))
import kohonen as original  # noqa: E402

from som import SOM, SOMConfig  # noqa: E402


def run_original(X, n_iter, w, h, init):
    real = np.random.random
    np.random.random = lambda shape: init.copy()
    try:
        t0 = time.perf_counter()
        out = original.train(X, n_iter, w, h)
        return out, time.perf_counter() - t0
    finally:
        np.random.random = real


def run_vectorised(X, n_iter, w, h, init):
    t0 = time.perf_counter()
    out = SOM(SOMConfig(width=w, height=h, n_epochs=n_iter)).fit(X, initial_weights=init).weights
    return out, time.perf_counter() - t0


def case(name, w, h, n_iter, *, original_iters=None):
    rng = np.random.default_rng(0)
    X = rng.random((10, 3))
    init = rng.random((w, h, 3))

    # For the big case the original is too slow to run in full by default;
    # run a few epochs and scale linearly (it is exactly linear in epochs).
    o_iters = original_iters or n_iter
    o_out, o_time = run_original(X, o_iters, w, h, init)
    if o_iters != n_iter:
        o_time = o_time * n_iter / o_iters
        o_out = None

    v_out, v_time = run_vectorised(X, n_iter, w, h, init)

    same = "n/a (extrapolated)" if o_out is None else str(np.allclose(o_out, v_out, atol=1e-12))
    print(
        f"{name:<28} original {o_time:>8.2f}s   vectorised {v_time:>7.3f}s   "
        f"speedup {o_time / v_time:>6.0f}x   identical: {same}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run the big original case in full")
    args = ap.parse_args()

    print()
    case("10x10 grid, 100 epochs", 10, 10, 100)
    case("100x100 grid, 1000 epochs", 100, 100, 1000, original_iters=None if args.full else 5)
    print()
