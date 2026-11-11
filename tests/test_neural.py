"""PyTorch perceptrons (dl extra).

The training tests skip without PyTorch. The parameter test runs everywhere,
and the missing-dependency message is checked only where PyTorch is absent.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from voice_severity_bench import (
    Config,
    MissingDependencyError,
    TorchMLPClassifier,
    TorchMLPRegressor,
    fit_model,
    load_model,
    save_model,
)


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="PyTorch is installed")
def test_missing_torch_names_the_dl_extra():
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        TorchMLPRegressor(n_epochs=1).fit(np.zeros((4, 2)), np.zeros(4))


def test_estimators_are_scikit_learn_compatible_without_torch():
    model = TorchMLPClassifier(hidden_units=(8,), n_epochs=3)
    assert model.get_params()["hidden_units"] == (8,)
    assert TorchMLPRegressor().set_params(n_epochs=5).n_epochs == 5


def test_regressor_learns_a_linear_map():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) * 10 + 40
    model = TorchMLPRegressor(hidden_units=(16,), activation="tanh", n_epochs=150, random_state=0)
    model.fit(X, y)
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.95


def test_classifier_probabilities_and_model_file(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 2))
    y = np.where(X[:, 0] > 0, "dysphonic", "healthy")
    config = Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": "torch_mlp",
            "ablation.models": [],
            # 60 epochs of 4 mini-batches at the default 1e-3 leave the logistic
            # units barely trained (about 61 % here); 1e-2 separates the classes.
            "model.n_epochs": 60,
            "model.learning_rate": 1e-2,
            "model.hidden_units": [8],
        }
    )
    pipeline = fit_model(X, y, config)
    probabilities = pipeline.predict_proba(X)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(pipeline.predict(X) == y) > 0.9
    path = save_model(tmp_path / "model.pt", pipeline, config, ["dysphonic", "healthy"])
    restored = load_model(path)["estimator"]
    assert np.allclose(restored.predict_proba(X), probabilities, atol=1e-6)
