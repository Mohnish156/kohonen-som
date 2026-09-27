import json

import numpy as np
import pytest

from som import SOMConfig
from som.store import load_artifact
from som.training import pipeline


def test_load_random_is_seeded():
    a = pipeline.load(None, n_random=5, seed=1)
    b = pipeline.load(None, n_random=5, seed=1)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5, 3)


def test_load_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("0.1,0.2,0.3\n0.4,0.5,0.6\n")
    assert pipeline.load(p).shape == (2, 3)


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((0, 3)), np.array([[1.0, np.nan]])])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        pipeline.validate(bad)


def test_run_writes_artifact(tmp_path):
    cfg = SOMConfig(width=6, height=6, n_epochs=5, seed=0)
    metrics = pipeline.run(cfg, tmp_path / "out")

    assert (tmp_path / "out" / "weights.npz").exists()
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["metrics"] == metrics
    assert meta["config"]["width"] == 6
    assert metrics["quantization_error"] >= 0 and metrics["train_seconds"] > 0

    som, meta2 = load_artifact(tmp_path / "out")
    assert som.config == cfg and meta2 == meta
