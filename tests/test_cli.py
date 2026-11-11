"""Command line, end to end on synthetic data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from voice_severity_bench.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
QUICK = PROJECT_ROOT / "configs" / "quick_demo.yaml"
COMMANDS = (
    "synthesize",
    "index",
    "prepare",
    "evaluate",
    "train",
    "predict",
    "extract",
    "ablate",
    "compare",
    "correlate",
    "regress",
    "stepwise",
    "order-sweep",
)
MANIFEST_KEYS = [
    "command",
    "created_utc",
    "package_version",
    "git_commit",
    "seed",
    "versions",
    "inputs",
    "config",
    "details",
]


def run(*args):
    return RUNNER.invoke(app, [str(a) for a in args])


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    assert run("synthesize", "--config", QUICK, "--out", data).exit_code == 0
    result = run(
        "prepare", "--config", QUICK, "--dataset", data / "dataset.csv", "--out", tmp_path / "prep"
    )
    assert result.exit_code == 0, result.output
    return tmp_path


def test_full_workflow_writes_every_fixed_file(prepared):
    root = prepared
    for name in ("dataset.csv", "config.yaml", "manifest.json"):
        assert (root / "data" / name).is_file(), name
    assert (root / "data" / "items" / "item_000.wav").is_file()
    for name in ("features.npz", "items.csv", "config.yaml", "manifest.json"):
        assert (root / "prep" / name).is_file(), name

    result = run(
        "evaluate", root / "prep" / "features.npz", "--config", QUICK, "--out", root / "ev"
    )
    assert result.exit_code == 0, result.output
    assert "RMSE" in result.output and "CCC" in result.output
    for name in (
        "metrics.json",
        "folds.csv",
        "predictions.csv",
        "residuals.csv",
        "residuals.png",
        "bland_altman.csv",
        "bland_altman.png",
    ):
        assert (root / "ev" / name).is_file(), name
    metrics = json.loads((root / "ev" / "metrics.json").read_text(encoding="utf-8"))
    assert list(metrics) == ["task", "n_items", "n_splits", "seed", "folds", "summary", "pooled"]
    manifest = json.loads((root / "ev" / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest) == MANIFEST_KEYS
    assert manifest["command"] == "evaluate"
    columns = pd.read_csv(root / "ev" / "predictions.csv").columns.tolist()
    assert columns == ["path", "group", "fold", "severity", "predicted"]

    result = run("train", root / "prep" / "features.npz", "--config", QUICK, "--out", root / "tr")
    assert result.exit_code == 0, result.output
    assert (root / "tr" / "model.joblib").is_file()

    wav = root / "data" / "items" / "item_003.wav"
    result = run("predict", wav, "--model", root / "tr" / "model.joblib", "--out", root / "pr")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(root / "pr" / "predictions.csv")
    assert table.columns.tolist() == ["path", "predicted"] and len(table) == 1


def test_classification_workflow_writes_confusion_matrix_and_roc(prepared):
    root = prepared
    overrides = [
        "--set", "data.task=classification", "--set", "data.target=label",
        "--set", "model.name=logistic", "--set", 'ablation.models=["logistic"]',
        "--set", "evaluation.n_splits=3",
    ]  # fmt: skip
    data = root / "data" / "dataset.csv"
    result = run("prepare", "--config", QUICK, *overrides, "--dataset", data, "--out", root / "cp")
    assert result.exit_code == 0, result.output
    features = root / "cp" / "features.npz"
    result = run("evaluate", features, "--config", QUICK, *overrides, "--out", root / "ce")
    assert result.exit_code == 0, result.output
    assert "UAR" in result.output and "ROC-AUC" in result.output
    for name in ("confusion_matrix.csv", "confusion_matrix.png", "roc_curve.csv", "roc_curve.png"):
        assert (root / "ce" / name).is_file(), name
    columns = pd.read_csv(root / "ce" / "predictions.csv").columns.tolist()
    assert columns == [
        "path", "group", "fold", "label", "predicted", "score_dysphonic", "score_healthy",
    ]  # fmt: skip
    metrics = json.loads((root / "ce" / "metrics.json").read_text(encoding="utf-8"))
    assert list(metrics["summary"]) == [
        "uar", "accuracy", "f1_macro", "sensitivity", "specificity", "roc_auc",
    ]  # fmt: skip
    assert metrics["pooled"]["classes"] == ["dysphonic", "healthy"]
    result = run("train", features, "--config", QUICK, *overrides, "--out", root / "ct")
    assert result.exit_code == 0, result.output
    wav = root / "data" / "items" / "item_000.wav"
    result = run("predict", wav, "--model", root / "ct" / "model.joblib", "--out", root / "cpr")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(root / "cpr" / "predictions.csv")
    assert table.columns.tolist() == ["path", "predicted", "score_dysphonic", "score_healthy"]


def test_evaluate_is_reproducible_byte_for_byte(prepared):
    features = prepared / "prep" / "features.npz"
    for name in ("a", "b"):
        result = run("evaluate", features, "--config", QUICK, "--seed", 3, "--no-figures",
                     "--out", prepared / name)  # fmt: skip
        assert result.exit_code == 0, result.output
    for name in ("predictions.csv", "metrics.json", "folds.csv"):
        assert (prepared / "a" / name).read_bytes() == (prepared / "b" / name).read_bytes(), name


def test_prepare_and_synthesize_are_reproducible(tmp_path):
    for name in ("one", "two"):
        assert run("synthesize", "--config", QUICK, "--out", tmp_path / name).exit_code == 0
        result = run("prepare", "--config", QUICK, "--dataset", tmp_path / name / "dataset.csv",
                     "--out", tmp_path / f"{name}_prep")  # fmt: skip
        assert result.exit_code == 0, result.output
    one, two = tmp_path / "one", tmp_path / "two"
    assert (one / "dataset.csv").read_bytes() == (two / "dataset.csv").read_bytes()
    assert (one / "items" / "item_007.wav").read_bytes() == (
        two / "items" / "item_007.wav"
    ).read_bytes()
    first = (tmp_path / "one_prep" / "items.csv").read_text(encoding="utf-8")
    second = (tmp_path / "two_prep" / "items.csv").read_text(encoding="utf-8")
    assert first.replace("one", "two") == second


def test_without_out_a_time_stamped_run_folder_is_created(prepared):
    features = prepared / "prep" / "features.npz"
    root = prepared / "runs"
    result = run("evaluate", features, "--config", QUICK, "--no-figures", "--set",
                 f"output.root={root.as_posix()}")  # fmt: skip
    assert result.exit_code == 0, result.output
    folders = [path.name for path in root.iterdir()]
    assert len(folders) == 1 and folders[0].startswith("evaluate_") and folders[0].endswith("Z")


def test_help_version_and_bare_invocation():
    result = run("--help")
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output
    result = run("--version")
    assert result.exit_code == 0 and result.output.strip() == "voice-severity-bench 0.1.0"
    assert "Usage" in run().output


def test_analysis_commands_on_synthetic_features(prepared):
    features = prepared / "prep" / "features.npz"
    result = run("correlate", features, "--out", prepared / "co", "--top", 5)
    assert result.exit_code == 0, result.output
    assert (prepared / "co" / "correlations.csv").is_file()
    assert (prepared / "co" / "feature_correlations.png").is_file()
    assert len(pd.read_csv(prepared / "co" / "feature_correlations.csv")) == 5

    result = run("stepwise", features, "--out", prepared / "sw")
    assert result.exit_code == 0, result.output
    for name in ("stepwise_steps.csv", "coefficients.csv", "regression_summary.json"):
        assert (prepared / "sw" / name).is_file(), name

    result = run("regress", features, "--predictors", "std_c10,mean_c10", "--out", prepared / "rg")
    assert result.exit_code == 0, result.output
    summary = json.loads((prepared / "rg" / "regression_summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"model", "diagnostics"}
    for name in ("coefficients.csv", "vif.csv", "regression_diagnostics.png"):
        assert (prepared / "rg" / name).is_file(), name


def test_regress_and_order_sweep_on_a_csv_table(tmp_path):
    from voice_severity_bench import make_polynomial_data, make_quadratic_surface_data

    surface = tmp_path / "surface.csv"
    make_quadratic_surface_data(30, rng=0).to_csv(surface, index=False)
    result = run("regress", surface, "--target", "y", "--design", "quadratic", "--out",
                 tmp_path / "rs")  # fmt: skip
    assert result.exit_code == 0, result.output
    terms = pd.read_csv(tmp_path / "rs" / "coefficients.csv")["term"].tolist()
    assert terms == ["const", "x1", "x2", "x1^2", "x2^2", "x1*x2"]

    curve = tmp_path / "curve.csv"
    make_polynomial_data(40, rng=0).to_csv(curve, index=False)
    result = run("order-sweep", curve, "--x", "x", "--target", "y", "--out", tmp_path / "os")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "os" / "order_sweep.csv").is_file()
    assert (tmp_path / "os" / "order_sweep.png").is_file()
    assert "chosen" in result.output


def test_extract_and_ablate(prepared):
    data = prepared / "data"
    result = run("extract", data, "--config", QUICK, "--format", "htk", "--out", prepared / "ex")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(prepared / "ex" / "extraction.csv")
    assert (table["status"] == "ok").all()
    assert (prepared / "ex" / "features" / "items" / "item_000.htk").is_file()

    result = run("ablate", "--config", QUICK, "--dataset", data / "dataset.csv", "--out",
                 prepared / "ab", "--cache-dir", prepared / "cache")  # fmt: skip
    assert result.exit_code == 0, result.output
    for name in (
        "ablation_folds.csv",
        "ablation_summary.csv",
        "ablation_comparison.csv",
        "ablation.json",
        "ablation_metric_by_dimension.png",
        "ablation_metric_by_dimension.csv",
    ):
        assert (prepared / "ab" / name).is_file(), name
    assert any((prepared / "cache").glob("frames_*.npz"))


def test_index_matches_files_to_a_ratings_table(tmp_path):
    from speechdsp import write_wav

    from voice_severity_bench import make_vowel

    audio = tmp_path / "audio"
    for name in ("p01", "p02", "p03"):
        write_wav(audio / "visit1" / f"{name}.wav", make_vowel(30.0, rng=1), 16_000)
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("id,severity,group\np01,10,a\np02,50,b\np03,80,c\n", encoding="utf-8")
    result = run("index", audio, "--targets", ratings)
    assert result.exit_code == 0, result.output
    index = pd.read_csv(audio / "dataset.csv")
    assert index["path"].tolist() == ["visit1/p01.wav", "visit1/p02.wav", "visit1/p03.wav"]


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("evaluate", "missing.npz"), "not found"),
        (("prepare", "--set", "frontend.n_cepz=3"), "unknown configuration key"),
        (("prepare", "--set", "frontend.n_ceps=40"), "frontend.n_ceps"),
        (("prepare", "--set", "frontend.n_ceps"), "section.key=value"),
        (("prepare", "--config", "BROKEN"), "not valid YAML"),
        (("--log-level", "LOUD", "prepare"), "Usage"),
    ],
)
def test_expected_failures_exit_with_code_2(tmp_path, args, fragment):
    broken = tmp_path / "broken.yaml"
    broken.write_text("data: [unclosed", encoding="utf-8")
    resolved = [str(broken) if a == "BROKEN" else a for a in args]
    result = run(*resolved)
    assert result.exit_code == 2, result.output
    assert "error:" in result.output or "Usage" in result.output
    assert fragment in result.output
    assert isinstance(result.exception, SystemExit)


def test_mismatched_feature_settings_are_rejected(prepared):
    features = prepared / "prep" / "features.npz"
    result = run("evaluate", features, "--config", QUICK, "--set", "frontend.delta_order=1")
    assert result.exit_code == 2
    assert "frontend.delta_order" in result.output and "prepare" in result.output
    assert isinstance(result.exception, SystemExit)


def test_module_entry_point_lists_every_command():
    completed = subprocess.run(
        [sys.executable, "-m", "voice_severity_bench", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8"),
        check=True,
    )
    for command in COMMANDS:
        assert command in completed.stdout
