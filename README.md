# Kohonen SOM review

![ci](https://github.com/mohnish156/som-review/actions/workflows/ci.yml/badge.svg)

A code review of a Self-Organising Map implementation, with recommendations. The
code under review is in `original/kohonen.py`, untouched.

The rest of this repo is that code taken to production: the same algorithm rewritten
to be fast, tested and packaged, split into a training pipeline that produces a model
artifact and a service that loads one, with a Dockerfile and CI around it. Each of the
five recommendations below points at the part of the repo that demonstrates it.

```
original/kohonen.py       the code under review
src/som/
  model.py                SOM, SOMConfig
  store.py                artifact format (weights.npz + metadata.json)
  training/               pipeline + `python -m som.training`
  serving/                FastAPI app + schemas
tests/
benchmarks/compare.py
Dockerfile
.github/workflows/ci.yml
```

## Top 5 recommendations

### 1. Vectorise the update loop

The BMU search is already one numpy call over the grid; the update is a Python loop over
every node. Precompute a `(width, height, 2)` array of node coordinates once, then grid
distance, θ, and the weight update are each one broadcast. Same maths.

```python
coords = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="ij"), axis=-1)  # once

for v in X:
    d2_feat = np.sum((weights - v) ** 2, axis=-1)                   # (W, H)   unchanged
    bmu = np.unravel_index(np.argmin(d2_feat), d2_feat.shape)
    d2_grid = np.sum((coords - bmu) ** 2, axis=-1)                  # (W, H)   was the loop
    theta = np.exp(-d2_grid / (2 * sigma**2))                       # (W, H)
    weights += alpha * theta[..., None] * (v - weights)             # (W, H, F)
```

```
$ python benchmarks/compare.py

10x10 grid, 100 epochs       original     0.21s   vectorised   0.010s   speedup     21x   identical: True
100x100 grid, 1000 epochs    original   204.06s   vectorised   2.140s   speedup     95x   identical: n/a (extrapolated)
```

[`tests/test_equivalence.py`](tests/test_equivalence.py) starts both implementations
from identical weights and asserts the outputs agree to 1e-12.

### 2. Reproducible and validated

[`SOMConfig`](src/som/model.py) rejects invalid grids, learning rates and radii at
construction, including the σ0 ≤ 1 case that divides by zero in the original. `fit()`
rejects non-2-D, empty, or non-finite input. `seed` drives a `default_rng`, and
`n_features` comes from the data instead of being hard-coded to 3.

### 3. A proper module

[`SOM`](src/som/model.py) exposes `fit`, `transform` (BMU coordinates for new data),
`quantization_error`, `save`/`load`, and `to_image`. Training has no dependency on
matplotlib. Installable with `pip install -e .`.

### 4. Tests and a quality metric

[`tests/`](tests/): config validation, reproducibility, arbitrary feature dimension,
convergence, radius cutoff, save/load round-trip, pipeline, serving, and the
equivalence test. Quantisation error is the standard SOM quality number; it's written
into every artifact and reported by `/health`.

### 5. Separate training from serving

Training is a batch job that writes an artifact. Serving is a stateless process that
loads one. They share the model class and the artifact format, and nothing else.

```
python -m som.training ──writes──▶  artifacts/weights.npz + metadata.json  ◀──reads──  uvicorn som.serving.app:app
```

- [`training/pipeline.py`](src/som/training/pipeline.py): each step is a function, so
  the pipeline maps 1:1 onto components in Vertex / Kubeflow / Airflow.
- [`serving/app.py`](src/som/serving/app.py): `/map` is inference, `/health` reports
  which model is loaded and its quantisation error, `/train-job` returns `202` and
  would submit the job in production. The service never trains in-process.
- [`Dockerfile`](Dockerfile): one image; the default command serves, the job runner
  overrides it to train.

## How I'd productionise it

![architecture](docs/architecture.png)

- **Training job**: scheduled or triggered via `/train-job`. Runs the pipeline in the
  same image, writes the artifact to a model store.
- **Model store**: object storage or a registry. Versioned. Locally it's a directory.
- **Serving**: loads the latest artifact at startup. `transform` is one matrix op over
  the grid. Horizontally scalable; a redeploy picks up a new model.
- **Monitoring**: quantisation error on fresh data vs. the value in `metadata.json`.
  Drift past a threshold triggers a retrain.

Next steps, not built here: a real job runner behind `/train-job`, a model store client
instead of a directory, hot reload or polling for new artifacts, structured logging with
request ids, metrics export, and a size limit on `/map`.

## Running it

```bash
pip install -e ".[dev]"
pytest
python benchmarks/compare.py

python -m som.training --width 10 --height 10 --epochs 100 --seed 0 --out artifacts
SOM_ARTIFACT_DIR=artifacts uvicorn som.serving.app:app
curl localhost:8000/health
curl -X POST localhost:8000/map -H 'content-type: application/json' -d '{"data":[[0.1,0.2,0.3]]}'

docker build -t som .
docker run --rm -v ./artifacts:/artifacts som python -m som.training --width 10 --height 10 --out /artifacts
docker run --rm -p 8000:8000 -v ./artifacts:/artifacts som
```
