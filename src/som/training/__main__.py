"""som-train --width 10 --height 10 --epochs 100 --seed 0 --out artifacts/"""

from __future__ import annotations

import argparse
import logging
import sys

from som.model import SOMConfig
from som.training.pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="som-train")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data", default=None, help="CSV or image file; default random RGB points")
    p.add_argument("--pixels", type=int, default=5000, help="pixels to sample from an image")
    p.add_argument("--out", default="artifacts", help="artifact directory")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = SOMConfig(
        width=args.width,
        height=args.height,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    run(cfg, args.out, source=args.data, n_pixels=args.pixels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
