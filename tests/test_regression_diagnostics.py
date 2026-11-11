"""Regression diagnostics against their defining formulas and SciPy."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from voice_severity_bench import (
    breusch_pagan,
    condition_number,
    cooks_distance,
    durbin_watson,
    fit_ols,
    jarque_bera,
    leverage,
    make_collinear_data,
    residual_diagnostics,
    shapiro_wilk,
    vif,
)
from voice_severity_bench.regression_diagnostics import influence_table, studentized_residuals


def test_normality_tests_match_scipy(rng):
    sample = rng.standard_t(5, size=150)
    assert shapiro_wilk(sample).statistic == pytest.approx(stats.shapiro(sample)[0])
    reference = stats.jarque_bera(sample)
    result = jarque_bera(sample)
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-10)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-8)


def test_breusch_pagan_flags_heteroscedasticity(rng):
    x = rng.uniform(1, 10, 400)
    design = np.column_stack([np.ones_like(x), x])
    constant = breusch_pagan(rng.normal(0, 1, 400), design)
    growing = breusch_pagan(rng.normal(0, 1, 400) * x, design)
    assert constant.p_value > 0.01 and growing.p_value < 1e-6
    assert growing.statistic > constant.statistic


def test_durbin_watson_formula(rng):
    residuals = rng.normal(size=100)
    expected = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
    assert durbin_watson(residuals) == pytest.approx(expected)
    walk = np.cumsum(rng.normal(size=200))
    assert durbin_watson(walk - walk.mean()) < 0.5


def test_leverage_is_the_hat_diagonal(rng):
    X = np.column_stack([np.ones(20), rng.normal(size=(20, 2))])
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    assert np.allclose(leverage(X), np.diag(hat))
    assert leverage(X).sum() == pytest.approx(3.0)


def test_cooks_distance_matches_leave_one_out_refits(rng):
    x = rng.uniform(0, 5, 15)
    y = 1.0 + x + rng.normal(0, 0.5, 15)
    y[3] += 4.0
    result = fit_ols(x, y)
    distances = cooks_distance(result)
    design = result.design
    for i in (0, 3, 7):
        keep = np.arange(15) != i
        coef = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        shift = design @ (coef - result.coef)
        expected = shift @ shift / (result.n_params * result.sigma2)
        assert distances[i] == pytest.approx(expected, rel=1e-8)
    assert int(np.argmax(distances)) == 3
    student = studentized_residuals(result)
    hat = leverage(design)
    assert np.allclose(student, result.residuals / np.sqrt(result.sigma2 * (1 - hat)))


def test_vif_equals_the_inverse_correlation_diagonal(rng):
    a = rng.normal(size=60)
    X = np.column_stack([a, a + rng.normal(0, 0.5, 60), rng.normal(size=60)])
    expected = np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))
    assert np.allclose(vif(X)["vif"], expected, rtol=1e-8)


def test_near_exact_collinearity_is_flagged():
    data = make_collinear_data(40, noise_sd=1.0, rng=0)
    table = vif(data[["p1", "p2", "p3", "p4"]].to_numpy(), ["p1", "p2", "p3", "p4"])
    assert (table["vif"] > 10).all()
    exact = np.column_stack([np.arange(10.0), 2 * np.arange(10.0)])
    assert math.isinf(vif(exact)["vif"].iloc[0])


def test_condition_number_and_report(rng):
    assert condition_number(np.diag([4.0, 2.0, 0.5])) == pytest.approx(8.0)
    x = rng.uniform(0, 10, 50)
    result = fit_ols(x, 2 * x + rng.normal(size=50))
    report = residual_diagnostics(result)
    assert report["condition_number"] == pytest.approx(result.condition_number)
    assert 0.0 <= report["shapiro_p"] <= 1.0 and report["max_leverage"] < 1.0
    table = influence_table(result)
    assert table.columns.tolist() == [
        "observed", "fitted", "residual", "studentized", "leverage", "cooks_distance",
        "normal_quantile",
    ]  # fmt: skip
