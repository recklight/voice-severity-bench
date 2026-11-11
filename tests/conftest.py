"""Shared fixtures. Every fixture is synthesized in code; no recorded data is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_severity_bench import Config, prepare_features, write_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def fast_config():
    return Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")


@pytest.fixture
def synthetic_dataset(tmp_path, fast_config):
    return write_synthetic_dataset(tmp_path / "data", fast_config.synthetic, rng=0)


@pytest.fixture
def synthetic_features(synthetic_dataset, fast_config):
    return prepare_features(synthetic_dataset, fast_config)


@pytest.fixture
def classification_config(fast_config):
    return fast_config.override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": "logistic",
            "ablation.models": ["logistic"],
            "evaluation.n_splits": 3,
        }
    )
