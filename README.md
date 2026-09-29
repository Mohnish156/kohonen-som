# Kohonen SOM review

![ci](https://github.com/Mohnish156/kohonen-som/actions/workflows/ci.yml/badge.svg)

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
  training/               pipeline; `som-train`
  serving/                FastAPI app + schemas; `som-serve`
tests/
benchmarks/compare.py
Dockerfile
.github/workflows/ci.yml
```

## Top 5 recommendations

### 1. Vectorise the update loop

**Problem:** the BMU lookup is already vectorised (`np.argmin(np.sum(...))` over the whole
grid), but two lines later the weight update drops into a `for x: for y:` loop and touches
every node one at a time. On the 100x100 / 1000 epoch case that's 10k nodes x 10 samples x
1000 epochs = 100M trips through the Python interpreter to do a subtract and a multiply.
I measured it at ~3.5 min on my laptop.

**Fix:** precompute a `(W, H, 2)` array of node coordinates once. Then distance-to-BMU for
every node is a single subtraction, theta is a single `exp`, and the update is a single
broadcast. Same maths as before, just no Python loop.

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

`tests/test_equivalence.py` runs both versions from the same starting weights and checks
the outputs match to 1e-12, so this is a pure speedup with no behaviour change.

Files: `src/som/model.py` (`fit`), `benchmarks/compare.py`, `tests/test_equivalence.py`

### 2. Validate inputs, seed the RNG

**Problem:** no validation anywhere. Pass a 2x2 grid and sigma0 = 1, log(1) = 0, lambda
divides by zero. Pass anything that isn't 3 features and it breaks because the 3 is
hard-coded. Nothing is seeded so you can't reproduce a run, and the `__main__` block
regenerates the data between its two calls, so the two PNGs aren't even trained on the same
inputs.

**Fix:**
- `SOMConfig` validates on construction: grid size, learning rate, radius, and the sigma0 <= 1 case
- `fit()` refuses non-2D, empty or NaN input
- `seed` argument feeds `default_rng`
- feature count comes from `X.shape[1]`, not a literal 3

Files: `src/som/model.py`, `tests/test_model.py`

### 3. Make it a module, not a script

**Problem:** it's one function. You can't import it without also importing matplotlib. Once
you've trained you get a raw array back and that's it - no way to ask "which cell does this
new point land in", no save, no load. And the Greek variable names are cute but nobody can
type or grep them.

**Fix:** a `SOM` class
- `fit` trains, `transform` gives you BMU coords for new samples
- `quantization_error` tells you how well the map fits
- `save` / `load`: one .npz with config + weights, no pickle
- matplotlib is an optional extra, training doesn't need it
- `pip install -e .`, type hints, docstrings

Files: `src/som/model.py`

### 4. Tests, and an actual quality metric

**Problem:** zero tests, so any change is a gamble. And the only way to know if training
worked is to open the PNG and squint.

**Fix:** 44 tests. Config edge cases, reproducibility, non-RGB input, convergence on a single
point, save/load round trip, the pipeline, the API. The one I care most about is the
equivalence test against the original, because it's what let me do #1 without worrying.
Quantisation error (mean distance from each sample to its BMU) gets computed after every
training run and written into the artifact metadata.

Files: `tests/`

### 5. Split training from serving

**Problem:** `__main__` trains, plots, exits. The model doesn't outlive the process. There's
nothing you could deploy, version, or monitor.

**Fix:** two entry points that share the model class and an artifact format, nothing else
- `som-train` is the batch job: load, validate, fit, evaluate, save. Each step is a plain function so it drops into whatever orchestrator you've got
- it writes the artifact: `weights.npz` + `metadata.json`
- the FastAPI service loads that at startup. It never trains
- `/predict` uploads an image and gets it back repainted with the map's palette (colour quantisation)
- `/image` returns the trained grid as a PNG
- `/map` returns raw BMU coordinates for programmatic use
- `/train` is a stub that would submit the job in a real deployment
- `/deploy` loads a new artifact in the background, self-checks it, then swaps; the old model serves until then
- one Docker image: default command serves, override it to train
- CI builds the image, runs a training job in it, starts the server on the output and smoke-tests the endpoints

Files: `src/som/training/`, `src/som/serving/`, `src/som/store.py`, `Dockerfile`, `.github/workflows/ci.yml`

## How I'd productionise it

![architecture](docs/architecture.png)

- **Data unload** (purple): Snowflake + dbt land the feature table in a bucket on a
  schedule and call `/train` with the URI.
- **Training + deploy** (blue): `/train` submits a Cloud Run Job that runs the pipeline
  (load, validate, fit, evaluate, save) and writes the artifact to a versioned model store.
  It then calls `/deploy`, which loads the new artifact in the background, self-checks it,
  and swaps. The old model serves until the swap.
- **Realtime inference** (green): clients call `/predict` (upload an image, get it
  repainted), `/image` (see the grid), or `/map` (raw BMU coordinates). `transform` is one
  matrix op over the grid, so the service is stateless and scales horizontally.

Next steps, not built here: a real job runner behind `/train`, a model store client
instead of a directory, structured logging with request ids, metrics export, and a size
limit on `/predict`.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python benchmarks/compare.py
```

Train on the original's data (10 random RGB points), serve it, hit the API:

```bash
som-train --width 10 --height 10 --epochs 100 --seed 0 --out artifacts
som-serve --artifacts artifacts
open http://localhost:8000/docs
```

Train on a photo instead, then upload any image to `/predict` in the Swagger UI and get it
back repainted with the map's palette:

```bash
som-train --data data/sample.png --width 24 --height 24 --epochs 40 --seed 0 --out artifacts
som-serve --artifacts artifacts
```

`data/` is gitignored; drop any photo in there.

Docker:

```bash
docker build -t som .
docker run --rm -v ./artifacts:/artifacts -v ./data:/data som som-train --data /data/sample.png --width 24 --height 24 --epochs 40 --out /artifacts
docker run --rm -p 8000:8000 -v ./artifacts:/artifacts som
```
