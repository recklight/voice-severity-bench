"""Gradient descent: agreement with the closed form, scaling and stopping reasons."""

from __future__ import annotations

import numpy as np
import pytest

from voice_severity_bench import fit_linear_gd, fit_ols


def test_converged_descent_matches_least_squares_to_1e_6(rng):
    X = rng.normal(size=(200, 3)) * [1.0, 10.0, 0.1] + [5.0, -20.0, 0.0]
    y = 4.0 + X @ np.array([0.5, -0.2, 3.0]) + rng.normal(0, 0.2, 200)
    result = fit_linear_gd(X, y)
    reference = fit_ols(X, y)
    assert result.converged and not result.diverged
    assert np.allclose([result.intercept, *result.coef], reference.coef, rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(result.cost_history) <= 1e-15)


def test_unscaled_years_stall_without_standardization():
    years = np.arange(1998.0, 2018.0, 2.0)
    population = 21.8 + 0.08 * (years - 1998.0)
    scaled = fit_linear_gd(years, population)
    raw = fit_linear_gd(years, population, standardize=False, learning_rate=1e-7, max_iter=2000)
    reference = fit_ols(years, population)
    assert scaled.converged
    assert scaled.coef[0] == pytest.approx(reference.coef[1], rel=1e-6)
    assert not raw.converged and abs(raw.coef[0] - reference.coef[1]) > 1e-3


def test_too_large_a_learning_rate_is_detected(rng):
    X = rng.normal(size=(50, 2))
    result = fit_linear_gd(X, X.sum(axis=1), learning_rate=5.0)
    assert result.diverged and not result.converged


def test_without_intercept(rng):
    X = rng.normal(size=(100, 2)) + 3.0
    y = X @ np.array([1.5, -0.5])
    result = fit_linear_gd(X, y, fit_intercept=False)
    assert result.intercept == 0.0
    assert np.allclose(result.coef, [1.5, -0.5], atol=1e-6)
    assert np.allclose(result.predict(X), y, atol=1e-5)


def test_input_validation():
    with pytest.raises(ValueError, match="learning_rate"):
        fit_linear_gd(np.ones((3, 1)), np.ones(3), learning_rate=0.0)
    with pytest.raises(ValueError, match="NaN"):
        fit_linear_gd(np.array([1.0, np.nan]), np.ones(2))
