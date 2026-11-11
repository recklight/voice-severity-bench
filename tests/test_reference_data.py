"""The published cement hardening data reproduce their textbook regression results."""

from __future__ import annotations

import numpy as np
import pytest

from voice_severity_bench import cement_hardening_heat, fit_ols, stepwise_select, vif


def test_full_model_coefficients_and_fit():
    data = cement_hardening_heat()
    result = fit_ols(data[["x1", "x2", "x3", "x4"]], data["heat"])
    assert np.allclose(result.coef, [62.4054, 1.5511, 0.5102, 0.1019, -0.1441], atol=1e-4)
    assert result.r2 == pytest.approx(0.9824, abs=1e-4)


def test_the_compounds_are_nearly_collinear():
    data = cement_hardening_heat()
    totals = data[["x1", "x2", "x3", "x4"]].sum(axis=1)
    assert totals.between(95, 99).all()
    assert (vif(data[["x1", "x2", "x3", "x4"]].to_numpy())["vif"] > 10).all()


def test_stepwise_with_a_looser_entry_threshold_drops_x4_again():
    data = cement_hardening_heat()
    X = data[["x1", "x2", "x3", "x4"]]
    result = stepwise_select(X, data["heat"], X.columns, p_enter=0.08, p_remove=0.10)
    assert result.steps["term"].tolist()[1:] == ["x4", "x1", "x2", "x4"]
    assert result.steps["action"].tolist()[-1] == "remove"
    assert result.selected == ("x1", "x2")
