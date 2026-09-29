"""The model artifact: the one contract between training and serving.

Training writes it, serving reads it. Neither imports the other.
Locally it's a directory; in production it'd be a bucket or a registry.

    <dir>/weights.npz      SOM config + weights (see SOM.save)
    <dir>/metadata.json    what was trained, on what, how well, when
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from som.model import SOM


def save_artifact(som: SOM, metrics: dict, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    som.save(dest / "weights.npz")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(som.config),
        "metrics": metrics,
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    return dest


def load_artifact(src: str | Path) -> tuple[SOM, dict]:
    src = Path(src)
    som = SOM.load(src / "weights.npz")
    meta = json.loads((src / "metadata.json").read_text())
    return som, meta
