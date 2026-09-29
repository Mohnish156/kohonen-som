import numpy as np
import pytest
from fastapi.testclient import TestClient

from som import SOMConfig
from som.serving import app as serving
from som.training.pipeline import run


@pytest.fixture
def client_with_model(tmp_path, monkeypatch):
    run(SOMConfig(width=6, height=6, n_epochs=5, seed=0), tmp_path)
    monkeypatch.setenv("SOM_ARTIFACT_DIR", str(tmp_path))
    with TestClient(serving.app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SOM_ARTIFACT_DIR", str(tmp_path / "empty"))
    with TestClient(serving.app) as c:
        yield c


def test_health_reports_model(client_with_model):
    body = client_with_model.get("/health").json()
    assert body["model_loaded"] is True
    assert body["quantization_error"] is not None


def test_map_returns_grid_coords(client_with_model):
    r = client_with_model.post("/map", json={"data": [[0.1, 0.2, 0.3], [0.9, 0.9, 0.9]]})
    assert r.status_code == 200
    bmu = r.json()["bmu"]
    assert len(bmu) == 2 and all(0 <= x < 6 and 0 <= y < 6 for x, y in bmu)


def test_map_wrong_feature_count_is_422(client_with_model):
    assert client_with_model.post("/map", json={"data": [[0.1, 0.2]]}).status_code == 422


def test_map_without_model_is_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    assert client_without_model.post("/map", json={"data": [[0.1, 0.2, 0.3]]}).status_code == 503


def test_train_job_is_a_stub(client_with_model):
    r = client_with_model.post("/train-job", json={"width": 10, "height": 10})
    assert r.status_code == 202
    assert r.json()["status"] == "accepted" and r.json()["job_id"].startswith("job-")


def test_image_endpoint_returns_png(client_with_model):
    r = client_with_model.get("/image?scale=2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_repaints_image(client_with_model, tmp_path):
    import io

    from PIL import Image

    Image.fromarray(np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)).save(
        tmp_path / "in.png"
    )
    files = {"file": ("in.png", (tmp_path / "in.png").read_bytes(), "image/png")}
    r = client_with_model.post("/predict", files=files)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    out = Image.open(io.BytesIO(r.content))
    assert out.size == (30, 20)
