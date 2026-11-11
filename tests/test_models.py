"""Model registry, reducers, nested tuning and model files."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from voice_severity_bench import (
    MODEL_NAMES,
    Config,
    ConfigError,
    ModelFileError,
    build_model,
    fit_model,
    load_model,
    save_model,
)
from voice_severity_bench.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    BlockSelector,
    model_filename,
    search_space,
)


@pytest.fixture
def regression_data(rng):
    X = rng.normal(size=(60, 5))
    return X, X @ np.array([2.0, -1.0, 0.0, 0.5, 0.0]) * 10 + 50 + rng.normal(0, 2, 60)


@pytest.fixture
def class_data(rng):
    X = rng.normal(size=(60, 4))
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=60) > 0, "dysphonic", "healthy")
    return X, y


@pytest.mark.parametrize("name", [n for n in REGRESSION_MODELS if n != "torch_mlp"])
def test_every_regressor_fits_and_predicts(name, regression_data):
    X, y = regression_data
    config = Config().override({"model.name": name, "model.n_estimators": 20})
    model = fit_model(X, y, config)
    assert [step for step, _ in model.steps] == ["scale", "reduce", "model"]
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.8


@pytest.mark.parametrize("name", [n for n in CLASSIFICATION_MODELS if n != "torch_mlp"])
def test_every_classifier_fits_and_scores(name, class_data):
    X, y = class_data
    config = Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": name,
            "ablation.models": [],
            "model.n_estimators": 20,
        }
    )
    model = fit_model(X, y, config)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (60, 2) and np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) > 0.8


def test_names_cover_both_tasks():
    assert set(MODEL_NAMES) == set(REGRESSION_MODELS) | set(CLASSIFICATION_MODELS)
    with pytest.raises(ConfigError, match=r"data\.task"):
        build_model(Config().override({"model.name": "logistic"}))


def test_reducers_are_fitted_inside_the_pipeline(regression_data, class_data):
    X, y = regression_data
    pca = fit_model(X, y, Config().override({"model.reducer": "pca", "model.n_components": 2}))
    assert pca.named_steps["reduce"].n_components_ == 2
    Xc, yc = class_data
    lda_config = Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": "logistic",
            "model.reducer": "lda",
            "ablation.models": [],
        }
    )
    lda = fit_model(Xc, yc, lda_config)
    assert lda.named_steps["reduce"].transform(Xc).shape == (60, 1)
    with pytest.raises(ConfigError, match="discriminant directions"):
        fit_model(Xc, yc, lda_config.override({"model.n_components": 3}))


def test_tuning_searches_the_grid_on_inner_folds(regression_data):
    X, y = regression_data
    config = Config().override({"model.tune": True})
    assert search_space(config) == {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    groups = np.repeat([f"s{i}" for i in range(12)], 5)
    model = fit_model(X, y, config, groups=groups)
    assert model.named_steps["model"].alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
    untuned = fit_model(X, y, Config().override({"model.name": "linear", "model.tune": True}))
    assert untuned.named_steps["model"].__class__.__name__ == "LinearRegression"


def test_model_file_round_trip(tmp_path, regression_data):
    X, y = regression_data
    config = Config()
    model = fit_model(X, y, config)
    path = save_model(tmp_path / model_filename(config), model, config)
    bundle = load_model(path)
    assert bundle["format"] == "voice_severity_bench/model-v1"
    assert Config.from_dict(bundle["config"]) == config
    assert np.allclose(bundle["estimator"].predict(X), model.predict(X))


def test_foreign_and_missing_model_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_model(tmp_path / "absent.joblib")
    joblib.dump({"format": "something-else"}, tmp_path / "other.joblib")
    with pytest.raises(ModelFileError, match="not written by"):
        load_model(tmp_path / "other.joblib")
    (tmp_path / "junk.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ModelFileError, match="not a readable"):
        load_model(tmp_path / "junk.joblib")
    assert model_filename(Config().override({"model.name": "torch_mlp"})) == "model.pt"


def test_the_inner_loop_picks_the_informative_block(rng):
    signal = rng.normal(size=(80, 3))
    y = signal @ np.array([3.0, -2.0, 1.0]) * 5 + rng.normal(0, 1, 80)
    noise = rng.normal(size=(80, 6))
    X = np.hstack([noise[:, :3], signal, noise[:, 3:]])
    blocks = {"layer0": (0, 3), "layer1": (3, 6), "layer2": (6, 9)}
    config = Config().override(
        {"model.tune": True, "model.block_selection": True, "model.ridge_alphas": [0.1, 10.0]}
    )
    model = fit_model(X, y, config, blocks=blocks)
    assert model.get_params()["select__block"] == "layer1"
    assert model.named_steps["select"].columns_.tolist() == [3, 4, 5]
    paired = fit_model(X, y, config, blocks=blocks, block_candidates=[("layer0", "layer2")])
    assert paired.get_params()["select__block"] == ("layer0", "layer2")


def test_median_imputation_keeps_rows_with_undefined_measures(rng):
    X = rng.normal(size=(40, 3))
    y = X[:, 0] * 10 + rng.normal(0, 1, 40)
    X[::4, 1] = np.nan
    X[:, 2] = np.nan  # a measure that is undefined for every training item
    model = fit_model(X, y, Config().override({"model.impute": "median"}))
    assert np.isfinite(model.predict(X)).all()
    # Two indicator columns join the three features: one per column that had gaps.
    assert model.named_steps["impute"].transform(X).shape == (40, 5)


def test_block_selector_rejects_unknown_and_misfit_blocks():
    X = np.zeros((2, 4))
    with pytest.raises(ConfigError, match="unknown feature block"):
        BlockSelector({"a": (0, 2)}, "b").fit(X)
    with pytest.raises(ConfigError, match="do not fit"):
        BlockSelector({"a": (2, 6)}, "a").fit(X)
    with pytest.raises(ValueError, match="columns"):
        BlockSelector({"a": (0, 2)}, "a").fit(X).transform(np.zeros((2, 3)))


@pytest.mark.parametrize("impute", ["none", "median"])
def test_the_ridge_path_chooses_what_grid_search_chooses(rng, impute, monkeypatch):
    from voice_severity_bench import models

    X = rng.normal(size=(90, 40))
    y = X[:, 12] * 5 + X[:, 13] * 3 + rng.normal(0, 2, 90)
    if impute == "median":
        X[rng.random(X.shape) < 0.05] = np.nan
    blocks = {f"b{i}": (10 * i, 10 * i + 10) for i in range(4)}
    config = Config().override(
        {
            "model.tune": True,
            "model.block_selection": True,
            "model.impute": impute,
            "model.ridge_alphas": np.logspace(-3, 7, 21).tolist(),
        }
    )
    groups = np.repeat([f"s{i}" for i in range(30)], 3)
    fast = fit_model(X, y, config, groups=groups, blocks=blocks)
    monkeypatch.setattr(models, "_ridge_path_applies", lambda *args: False)
    slow = fit_model(X, y, config, groups=groups, blocks=blocks)
    for key in ("model__alpha", "select__block"):
        assert fast.get_params()[key] == slow.get_params()[key]
    X_new = np.nan_to_num(rng.normal(size=(10, 40)))
    assert np.allclose(fast.predict(X_new), slow.predict(X_new), atol=1e-8)


@pytest.mark.parametrize("name", ["logistic", "svc", "random_forest"])
def test_class_weight_reaches_the_classifier(name):
    settings = {"data.task": "classification", "model.name": name}
    assert build_model(Config().override(settings))[-1].class_weight is None
    balanced = Config().override(settings | {"model.class_weight": "balanced"})
    assert build_model(balanced)[-1].class_weight == "balanced"


@pytest.mark.parametrize(
    ("task", "name"),
    [
        ("classification", "mlp"),
        ("classification", "torch_mlp"),
        ("regression", "ridge"),
        ("regression", "random_forest"),
    ],
)
def test_class_weight_is_refused_where_it_cannot_work(task, name):
    config = Config().override(
        {"data.task": task, "model.name": name, "model.class_weight": "balanced"}
    )
    with pytest.raises(ConfigError, match=r"model\.class_weight"):
        build_model(config)


def test_unknown_class_weights_are_refused():
    with pytest.raises(ConfigError, match=r"model\.class_weight"):
        Config().override({"model.class_weight": "inverse"})


def test_balanced_weights_recover_the_rare_class(rng):
    X = np.concatenate([rng.normal(0, 1, (180, 2)), rng.normal(1, 1, (20, 2))])
    y = np.array(["common"] * 180 + ["rare"] * 20)
    settings = {"data.task": "classification", "model.name": "logistic"}
    plain = fit_model(X, y, Config().override(settings)).predict(X)
    balanced = fit_model(
        X, y, Config().override(settings | {"model.class_weight": "balanced"})
    ).predict(X)
    assert np.count_nonzero(balanced[180:] == "rare") > np.count_nonzero(plain[180:] == "rare")
