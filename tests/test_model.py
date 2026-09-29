import numpy as np
import pytest

from som import SOM, SOMConfig


@pytest.fixture
def rgb():
    return np.random.default_rng(0).random((10, 3))


# ---------------------------------------------------------------- config


def test_defaults_match_spec():
    cfg = SOMConfig(width=10, height=4, n_epochs=100)
    assert cfg.sigma_0 == 5.0
    assert cfg.learning_rate == 0.1
    assert cfg.time_constant == pytest.approx(100 / np.log(5.0))


def test_radius_decays_to_one_on_last_epoch():
    cfg = SOMConfig(width=10, height=10, n_epochs=100)
    sigma_last, _ = SOM(cfg)._schedule(cfg.n_epochs)
    assert sigma_last == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=0, height=5),
        dict(width=5, height=5, n_epochs=0),
        dict(width=5, height=5, learning_rate=0),
        dict(width=5, height=5, learning_rate=1.5),
        dict(width=2, height=2),  # sigma_0 = 1 -> ln(1) = 0 -> original divides by zero
        dict(width=5, height=5, initial_radius=-1),
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SOMConfig(**kwargs)


# ------------------------------------------------------------------- fit


def test_fit_sets_weights_and_returns_self(rgb):
    som = SOM(SOMConfig(width=6, height=5, n_epochs=5, seed=0))
    assert som.fit(rgb) is som
    assert som.weights.shape == (6, 5, 3)
    assert np.all(np.isfinite(som.weights))


def test_seed_makes_fit_reproducible(rgb):
    a = SOM(SOMConfig(width=5, height=5, n_epochs=10, seed=42)).fit(rgb)
    b = SOM(SOMConfig(width=5, height=5, n_epochs=10, seed=42)).fit(rgb)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_any_feature_dimension():
    X = np.random.default_rng(1).random((20, 7))
    som = SOM(SOMConfig(width=4, height=4, n_epochs=3, seed=0)).fit(X)
    assert som.weights.shape == (4, 4, 7)


def test_training_lowers_quantization_error(rgb):
    cfg = SOMConfig(width=8, height=8, n_epochs=50, seed=0)
    before = SOM(cfg, weights=np.random.default_rng(99).random((8, 8, 3)))
    after = SOM(cfg).fit(rgb)
    assert after.quantization_error(rgb) < before.quantization_error(rgb)


def test_single_input_converges_to_it():
    X = np.array([[0.2, 0.7, 0.9]])
    som = SOM(SOMConfig(width=5, height=5, n_epochs=200, seed=0)).fit(X)
    x, y = som.transform(X)[0]
    np.testing.assert_allclose(som.weights[x, y], X[0], atol=1e-3)


def test_radius_cutoff_skips_far_nodes():
    X = np.array([[0.5, 0.5, 0.5]])
    init = np.zeros((21, 21, 3))
    init[10, 10] = 0.5  # plant the BMU in the centre
    cfg = SOMConfig(width=21, height=21, n_epochs=1, initial_radius=2.0, radius_cutoff=True)
    som = SOM(cfg).fit(X, initial_weights=init)
    assert np.all(som.weights[0, 0] == 0)  # far corner untouched
    assert som.weights[10, 9, 0] > 0  # neighbour inside radius moved


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((0, 3)), np.array([[1.0, np.nan]])])
def test_bad_input_rejected(bad):
    with pytest.raises(ValueError):
        SOM(SOMConfig(width=5, height=5)).fit(bad)


def test_initial_weights_shape_checked(rgb):
    with pytest.raises(ValueError):
        SOM(SOMConfig(width=5, height=5)).fit(rgb, initial_weights=np.zeros((4, 4, 3)))


# ------------------------------------------------------------- inference


def test_transform_matches_bruteforce(rgb):
    som = SOM(SOMConfig(width=6, height=4, n_epochs=5, seed=0)).fit(rgb)
    got = som.transform(rgb)
    assert got.shape == (10, 2)
    for x, (gx, gy) in zip(rgb, got, strict=True):
        d = np.sum((som.weights - x) ** 2, axis=-1)
        assert (gx, gy) == np.unravel_index(np.argmin(d), d.shape)


def test_unfitted_model_raises(rgb):
    som = SOM(SOMConfig(width=5, height=5))
    with pytest.raises(RuntimeError):
        som.transform(rgb)
    with pytest.raises(RuntimeError):
        som.quantization_error(rgb)


def test_feature_mismatch_on_transform(rgb):
    som = SOM(SOMConfig(width=5, height=5, n_epochs=2, seed=0)).fit(rgb)
    with pytest.raises(ValueError):
        som.transform(np.ones((2, 4)))


# ----------------------------------------------------------- persistence


def test_save_load_roundtrip(tmp_path, rgb):
    cfg = SOMConfig(
        width=5,
        height=4,
        n_epochs=7,
        learning_rate=0.2,
        initial_radius=3.0,
        radius_cutoff=True,
        seed=3,
    )
    som = SOM(cfg).fit(rgb)
    path = tmp_path / "m.npz"
    som.save(path)
    loaded = SOM.load(path)
    assert loaded.config == cfg
    np.testing.assert_array_equal(loaded.weights, som.weights)


def test_to_image(rgb):
    som = SOM(SOMConfig(width=6, height=4, n_epochs=2, seed=0)).fit(rgb)
    img = som.to_image()
    assert img.shape == (4, 6, 3) and img.min() >= 0 and img.max() <= 1


def test_quantize_returns_bmu_weights(rgb):
    som = SOM(SOMConfig(width=6, height=4, n_epochs=5, seed=0)).fit(rgb)
    q = som.quantize(rgb)
    bmu = som.transform(rgb)
    np.testing.assert_array_equal(q, som.weights[bmu[:, 0], bmu[:, 1]])
    with pytest.raises(ValueError):
        SOM(SOMConfig(width=4, height=4, n_epochs=1, seed=0)).fit(np.ones((3, 2))).to_image()
