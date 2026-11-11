"""Utterance pooling: statistics against SciPy, layout, names and degenerate combinations."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from voice_severity_bench import FrontendConfig, PoolingConfig, pool_frames, pooled_feature_names
from voice_severity_bench.pooling import statistic_labels


def test_statistics_match_scipy(rng):
    frames = rng.normal(size=(50, 3)) * [1.0, 2.0, 0.5] + [0.0, 5.0, -1.0]
    pooled = pool_frames(
        frames,
        ["mean", "std", "min", "max", "percentile", "skew", "kurtosis"],
        percentiles=[25.0, 75.0],
    )
    blocks = pooled.reshape(-1, 3)
    assert np.allclose(blocks[0], frames.mean(axis=0))
    assert np.allclose(blocks[1], frames.std(axis=0))
    assert np.allclose(blocks[2], frames.min(axis=0)) and np.allclose(blocks[3], frames.max(axis=0))
    assert np.allclose(blocks[4:6], np.percentile(frames, [25.0, 75.0], axis=0))
    assert np.allclose(blocks[6], stats.skew(frames, axis=0))
    assert np.allclose(blocks[7], stats.kurtosis(frames, axis=0))


def test_constant_columns_have_zero_higher_moments():
    pooled = pool_frames(np.ones((10, 2)), ["skew", "kurtosis"])
    assert np.all(pooled == 0.0)


def test_normalization_before_pooling(rng):
    frames = rng.normal(3.0, 2.0, size=(40, 2))
    centered = pool_frames(frames, ["std", "max"], normalization="cmn")
    assert np.allclose(centered[:2], frames.std(axis=0))
    assert np.allclose(centered[2:], (frames - frames.mean(axis=0)).max(axis=0))
    with pytest.raises(ValueError, match="constant"):
        pool_frames(frames, ["mean"], normalization="cmn")
    with pytest.raises(ValueError, match="constant"):
        pool_frames(frames, ["std"], normalization="cmvn")


def test_names_follow_the_vector_layout():
    frontend = FrontendConfig(n_ceps=2, delta_order=1)
    pooling = PoolingConfig(statistics=("mean", "percentile"), percentiles=(10.0, 90.0))
    names = pooled_feature_names(frontend, pooling)
    assert names[:6] == ("mean_c1", "mean_c2", "mean_c0", "mean_d_c1", "mean_d_c2", "mean_d_c0")
    assert names[6] == "p10_c1" and names[-1] == "p90_d_c0"
    assert len(names) == 3 * frontend.n_dims
    assert statistic_labels(["percentile"], [2.5]) == ("p2.5",)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        pool_frames(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="unknown pooling statistic"):
        pool_frames(np.zeros((4, 3)), ["median"])
    with pytest.raises(ValueError, match=r"pooling\.normalization"):
        pool_frames(np.zeros((4, 3)), ["std"], normalization="zscore")
