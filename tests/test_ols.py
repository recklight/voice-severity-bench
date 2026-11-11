"""Least squares: closed-form references, interval coverage and non-estimable designs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from voice_severity_bench import DesignMatrixError, fit_ols


def test_coefficients_match_scikit_learn(rng):
    X = rng.normal(size=(80, 3))
    y = 2.0 + X @ np.array([1.0, -0.5, 0.25]) + rng.normal(0, 0.3, 80)
    result = fit_ols(X, y)
    reference = LinearRegression().fit(X, y)
    assert np.allclose(result.coef[1:], reference.coef_, rtol=1e-10)
    assert result.coef[0] == pytest.approx(reference.intercept_, rel=1e-10)
    assert result.r2 == pytest.approx(reference.score(X, y), rel=1e-10)


def test_simple_regression_matches_linregress(rng):
    x = rng.uniform(0, 10, 40)
    y = 1.0 + 0.3 * x + rng.normal(0, 1, 40)
    result = fit_ols(x, y, ["x"])
    reference = stats.linregress(x, y)
    assert result.se[1] == pytest.approx(reference.stderr, rel=1e-10)
    assert result.p_values[1] == pytest.approx(reference.pvalue, rel=1e-8)
    # with one predictor the overall F test is the squared slope t test
    assert result.f_stat == pytest.approx(result.t[1] ** 2, rel=1e-10)
    assert result.f_pvalue == pytest.approx(reference.pvalue, rel=1e-8)


def test_sums_of_squares_and_information_criteria(rng):
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + rng.normal(size=30)
    result = fit_ols(X, y)
    assert result.tss == pytest.approx(result.ess + result.rss)
    assert result.adj_r2 == pytest.approx(1 - (1 - result.r2) * 29 / 27)
    n, p = 30, 3
    loglik = -n / 2 * (math.log(2 * math.pi) + math.log(result.rss / n) + 1)
    assert result.aic == pytest.approx(-2 * loglik + 2 * p)
    assert result.bic == pytest.approx(-2 * loglik + p * math.log(n))


def test_confidence_intervals_cover_about_95_percent():
    rng = np.random.default_rng(2024)
    x = np.linspace(0.0, 1.0, 20)
    covered = 0
    trials = 2000
    for _ in range(trials):
        y = 1.0 + 2.0 * x + rng.normal(0, 0.5, x.size)
        result = fit_ols(x, y)
        covered += int(result.ci_low[1] <= 2.0 <= result.ci_high[1])
    assert 0.93 < covered / trials < 0.97


def test_interpolating_fit_reports_nan_instead_of_fake_inference():
    x = np.arange(7.0)
    design = np.column_stack([x**k for k in range(1, 7)])
    result = fit_ols(design, x**2 + np.sin(x))
    assert result.df_resid == 0 and not result.estimable
    for value in (result.r2, result.adj_r2, result.f_stat, result.f_pvalue, result.aic):
        assert math.isnan(value)
    assert np.all(np.isnan(result.se)) and np.all(np.isnan(result.p_values))
    assert "not estimable" in result.summary()


def test_underdetermined_and_collinear_designs_are_rejected():
    with pytest.raises(DesignMatrixError, match="cannot be estimated"):
        fit_ols(np.ones((3, 5)), np.arange(3.0))
    x = np.arange(10.0)
    with pytest.raises(DesignMatrixError, match="rank deficient"):
        fit_ols(np.column_stack([x, 2 * x]), x)


def test_table_summary_and_prediction(rng):
    X = rng.normal(size=(25, 2))
    result = fit_ols(X, X.sum(axis=1) + rng.normal(0, 0.1, 25), ["a", "b"])
    table = result.coefficient_table()
    assert table["term"].tolist() == ["const", "a", "b"]
    assert np.allclose(result.predict(X), result.fitted)
    assert "R^2" in result.summary() and "condition number" in result.summary()
    assert set(result.as_dict()) >= {"r2", "adj_r2", "f_stat", "aic", "bic", "estimable"}
    without = fit_ols(X, X.sum(axis=1), add_intercept=False)
    assert without.names == ("x1", "x2") and np.allclose(without.coef, [1.0, 1.0])
