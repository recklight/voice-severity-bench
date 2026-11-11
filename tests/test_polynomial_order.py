"""Polynomial order sweep: in-sample monotonicity, PRESS identity and non-estimable degrees."""

from __future__ import annotations

import numpy as np
import pytest

from voice_severity_bench import (
    fit_ols,
    make_polynomial_data,
    polynomial_features,
    sweep_polynomial_order,
)


def test_in_sample_fit_grows_while_cross_validation_picks_the_true_degree():
    data = make_polynomial_data(60, noise_sd=0.5, rng=1)
    sweep = sweep_polynomial_order(data["x"], data["y"], range(1, 9), rng=0)
    assert sweep.table["r2"].is_monotonic_increasing
    assert sweep.recommended_degree == 3
    by_bic = sweep_polynomial_order(data["x"], data["y"], range(1, 9), criterion="bic", rng=0)
    assert by_bic.recommended_degree == 3


def test_loocv_equals_explicit_leave_one_out(rng):
    x = np.sort(rng.uniform(-2, 2, 15))
    y = x**2 + rng.normal(0, 0.3, 15)
    sweep = sweep_polynomial_order(x, y, [2], basis="centered", rng=0)
    errors = []
    for i in range(15):
        keep = np.arange(15) != i
        design, _ = polynomial_features(x[keep], 2, basis="raw")
        result = fit_ols(design, y[keep])
        errors.append(y[i] - result.predict(np.array([[x[i], x[i] ** 2]]))[0])
    expected = float(np.sqrt(np.mean(np.square(errors))))
    assert sweep.table["loocv_rmse"].iloc[0] == pytest.approx(expected, rel=1e-8)


def test_degrees_without_residual_freedom_are_not_estimable():
    x = np.arange(7.0)
    sweep = sweep_polynomial_order(x, x**3 - x, [1, 6, 7], rng=0)
    table = sweep.table.set_index("degree")
    assert table.loc[1, "estimable"] and not table.loc[6, "estimable"]
    assert table.loc[7, "df_resid"] == -1
    for column in ("r2", "adj_r2", "f_pvalue", "aic", "kfold_rmse"):
        assert np.isnan(table.loc[6, column]) and np.isnan(table.loc[7, column])
    assert sweep.recommended_degree == 1


def test_bases_give_the_same_fit(rng):
    x = rng.uniform(1950, 2010, 20)
    y = 0.001 * (x - 1980) ** 2 + rng.normal(0, 0.1, 20)
    tables = [
        sweep_polynomial_order(x, y, [1, 2, 3], basis=basis, rng=0).table
        for basis in ("orthogonal", "centered")
    ]
    assert np.allclose(tables[0]["sse"], tables[1]["sse"], rtol=1e-8)


def test_input_validation():
    with pytest.raises(ValueError, match="criterion"):
        sweep_polynomial_order(np.arange(5.0), np.arange(5.0), criterion="r2")
    with pytest.raises(ValueError, match="length"):
        sweep_polynomial_order(np.arange(5.0), np.arange(4.0))
