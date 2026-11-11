"""Every example script runs end to end on synthetic data (with --quick)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_quick(name: str, out: Path, expected: tuple[str, ...]) -> None:
    assert load(name).main(["--quick", "--out", str(out)]) == 0
    for filename in expected:
        assert (out / filename).is_file(), filename


def test_every_example_has_a_test():
    tested = {
        "end_to_end_synthetic",
        "compare_feature_settings",
        "compare_feature_families",
        "compare_severity_regressors",
        "diagnose_regression_models",
        "select_polynomial_order",
        "compare_gradient_descent_with_ols",
    }
    assert {path.stem for path in EXAMPLES.glob("*.py")} == tested


def test_end_to_end_synthetic(tmp_path):
    run_quick(
        "end_to_end_synthetic",
        tmp_path / "end_to_end_synthetic",
        ("metrics.json", "predictions.csv", "model.joblib", "new_predictions.csv"),
    )


def test_compare_feature_settings(tmp_path):
    run_quick(
        "compare_feature_settings",
        tmp_path / "compare_feature_settings",
        (
            "ablation_summary.csv",
            "ablation_comparison.csv",
            "ablation_metric_by_dimension.png",
            "ablation_metric_by_dimension.csv",
        ),
    )


def test_compare_feature_families(tmp_path):
    run_quick(
        "compare_feature_families",
        tmp_path / "compare_feature_families",
        (
            "comparison_summary.csv",
            "comparison_contrasts.csv",
            "comparison_metric_by_feature_file.png",
            "comparison.json",
        ),
    )


def test_compare_severity_regressors(tmp_path):
    run_quick(
        "compare_severity_regressors",
        tmp_path / "compare_severity_regressors",
        (
            "model_comparison.csv",
            "correlations.csv",
            "bland_altman_ridge.png",
            "bland_altman_ridge.csv",
        ),
    )


def test_diagnose_regression_models(tmp_path):
    run_quick(
        "diagnose_regression_models",
        tmp_path / "diagnose_regression_models",
        (
            "surface_coefficients.csv",
            "response_surface.png",
            "cement_vif.csv",
            "cement_stepwise_full.csv",
            "cement_diagnostics.json",
            "cement_residuals.png",
            "cement_residuals.csv",
        ),
    )


def test_select_polynomial_order(tmp_path):
    run_quick(
        "select_polynomial_order",
        tmp_path / "select_polynomial_order",
        ("order_sweep_ten_points.csv", "order_sweep_sixty_points.png"),
    )


def test_compare_gradient_descent_with_ols(tmp_path):
    run_quick(
        "compare_gradient_descent_with_ols",
        tmp_path / "compare_gradient_descent_with_ols",
        ("gradient_descent_runs.csv", "cost_history.png", "cost_history.csv"),
    )
