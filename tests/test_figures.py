"""Figures: every plot returns a Figure, files keep dotted stems, tables have stable columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from voice_severity_bench import (
    fit_ols,
    make_quadratic_surface_data,
    sweep_polynomial_order,
)
from voice_severity_bench.figures import (
    ablation_table,
    bland_altman_table,
    confusion_matrix_table,
    correlations_table,
    cost_history_table,
    new_figure,
    plot_ablation,
    plot_bland_altman,
    plot_confusion_matrix,
    plot_correlations,
    plot_cost_history,
    plot_order_sweep,
    plot_regression_diagnostics,
    plot_residuals,
    plot_response_surface,
    plot_roc_curve,
    regression_diagnostics_table,
    residuals_table,
    response_surface_table,
    roc_curve_table,
    save_figure,
)


def regression_predictions(rng):
    truth = rng.uniform(0, 100, 30)
    return pd.DataFrame(
        {
            "path": [f"x{i}.wav" for i in range(30)],
            "group": "",
            "fold": np.arange(30) % 3,
            "severity": truth,
            "predicted": truth + rng.normal(0, 8, 30),
        }
    )


def test_regression_result_figures(rng):
    predictions = regression_predictions(rng)
    assert isinstance(plot_residuals(predictions, "severity"), Figure)
    assert isinstance(plot_bland_altman(predictions, "severity"), Figure)
    assert residuals_table(predictions, "severity").columns.tolist() == [
        "path", "fold", "observed", "predicted", "residual",
    ]  # fmt: skip
    assert bland_altman_table(predictions, "severity").columns.tolist() == [
        "path", "mean", "difference", "bias", "lower_loa", "upper_loa",
    ]  # fmt: skip


def test_classification_figures(rng):
    truth = rng.choice(["dysphonic", "healthy"], 40)
    score = np.where(truth == "dysphonic", 0.7, 0.3) + rng.normal(0, 0.2, 40)
    predictions = pd.DataFrame({"label": truth, "score_dysphonic": np.clip(score, 0, 1)})
    assert isinstance(plot_roc_curve(predictions, "label", "dysphonic"), Figure)
    roc = roc_curve_table(predictions, "label", "dysphonic")
    assert roc.columns.tolist() == ["false_positive_rate", "true_positive_rate", "threshold"]
    assert roc["true_positive_rate"].is_monotonic_increasing
    matrix = [[10, 2], [3, 25]]
    assert isinstance(plot_confusion_matrix(["dysphonic", "healthy"], matrix), Figure)
    table = confusion_matrix_table(["dysphonic", "healthy"], matrix)
    assert table.columns.tolist() == ["true", "predicted_dysphonic", "predicted_healthy"]


def test_analysis_figures(rng):
    summary = pd.DataFrame(
        {
            "config_id": ["a", "b", "c"],
            "model": ["ridge", "ridge", "svr"],
            "n_features": [26, 78, 26],
            "rmse_mean": [8.0, 9.0, 10.0],
            "rmse_std": [1.0, 1.5, np.nan],
            "rank": [1, 2, 3],
        }
    )
    assert isinstance(plot_ablation(summary, "rmse"), Figure)
    assert ablation_table(summary, "rmse").columns.tolist() == [
        "config_id", "model", "n_features", "mean", "std", "rank",
    ]  # fmt: skip
    correlations = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "pearson_r": [0.8, -0.4, np.nan],
            "ci_low": [0.6, -0.7, np.nan],
            "ci_high": [0.9, -0.1, np.nan],
            "pearson_p_adjusted": [0.001, 0.04, np.nan],
        }
    )
    assert isinstance(plot_correlations(correlations, 2), Figure)
    assert len(correlations_table(correlations, 5)) == 2

    data = make_quadratic_surface_data(25, rng=0)
    result = fit_ols(data[["x1", "x2"]], data["y"], ["x1", "x2"])
    assert isinstance(plot_regression_diagnostics(result), Figure)
    assert "cooks_distance" in regression_diagnostics_table(result).columns

    sweep = sweep_polynomial_order(data["x1"], data["y"], [1, 2, 3], rng=0)
    assert isinstance(plot_order_sweep(sweep), Figure)

    grid = response_surface_table(
        lambda points: points.sum(axis=1), (-1.0, 1.0), (0.0, 2.0), ("x1", "x2"), resolution=5
    )
    assert grid.shape == (25, 3) and grid["fitted"].max() == 3.0
    assert isinstance(plot_response_surface(grid, data, ("x1", "x2"), "y"), Figure)

    histories = {"lr=0.1": np.geomspace(10, 1, 20), "lr=0.01": np.geomspace(10, 5, 40)}
    assert isinstance(plot_cost_history(histories), Figure)
    assert cost_history_table(histories).columns.tolist() == ["run", "iteration", "cost"]


def test_save_figure_keeps_dots_in_the_stem(tmp_path):
    path = save_figure(new_figure(2, 2, 50), tmp_path / "n_ceps=12.v2", "png")
    assert path.name == "n_ceps=12.v2.png" and path.stat().st_size > 0
