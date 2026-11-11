"""Stepwise selection: partial F tests, recorded steps and collinear synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from voice_severity_bench import make_collinear_data, stepwise_select
from voice_severity_bench.stepwise import partial_f_test


def test_partial_f_matches_its_definition():
    f_stat, p_value = partial_f_test(rss_reduced=50.0, rss_full=40.0, df_full=16, n_restrictions=2)
    assert f_stat == pytest.approx((10.0 / 2) / (40.0 / 16))
    assert p_value == pytest.approx(stats.f.sf(f_stat, 2, 16))


def test_forward_selection_finds_the_informative_predictors(rng):
    X = rng.normal(size=(80, 6))
    y = 3.0 * X[:, 1] - 2.0 * X[:, 4] + rng.normal(0, 0.5, 80)
    result = stepwise_select(X, y, [f"v{i}" for i in range(6)])
    assert result.selected == ("v1", "v4")
    assert result.steps["action"].tolist() == ["start", "enter", "enter"]
    assert result.steps["term"].tolist()[1:] == ["v1", "v4"]
    assert result.final.names == ("const", "v1", "v4")
    assert result.steps["r2"].is_monotonic_increasing


def test_backward_start_removes_the_noise(rng):
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + rng.normal(0, 0.3, 60)
    result = stepwise_select(X, y, start="full", p_remove=0.05, p_enter=0.01)
    assert result.selected == ("x1",)
    assert set(result.steps["action"].iloc[1:]) == {"remove"}


def test_collinear_mixture_still_terminates():
    data = make_collinear_data(40, rng=3)
    X = data[["p1", "p2", "p3", "p4"]]
    result = stepwise_select(X, data["y"], X.columns, start="full")
    assert 1 <= len(result.selected) <= 3
    assert len(result.steps) <= 2 * 4 + 11


def test_threshold_and_shape_validation(rng):
    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="p_enter <= p_remove"):
        stepwise_select(X, X[:, 0], p_enter=0.2, p_remove=0.1)
    with pytest.raises(ValueError, match="unique"):
        stepwise_select(X, X[:, 0], ["a", "a"])
    with pytest.raises(ValueError, match="empty model"):
        stepwise_select(rng.normal(size=(5, 5)), np.arange(5.0), start="full")
