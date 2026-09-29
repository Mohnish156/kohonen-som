"""som-serve: start the API against a trained artifact.

som-serve --artifacts artifacts --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="som-serve")
    p.add_argument("--artifacts", default="artifacts", help="dir with weights.npz + metadata.json")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="dev mode: restart on code changes")
    args = p.parse_args(argv)

    os.environ["SOM_ARTIFACT_DIR"] = args.artifacts
    import uvicorn

    uvicorn.run("som.serving.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
