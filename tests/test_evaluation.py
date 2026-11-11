"""Cross-validation: folds, out-of-fold predictions, and metrics checked against
scikit-learn, SciPy and speechdsp."""

from __future__ import annotations

import math

import numpy as np
import pytest
import speechdsp
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from voice_severity_bench import (
    ConfigError,
    FeatureSet,
    classification_metrics,
    concordance_ccc,
    cross_validate,
    regression_metrics,
)
from voice_severity_bench.evaluation import summarize


def test_every_item_is_predicted_once_and_groups_never_cross_folds(synthetic_features, fast_config):
    result = cross_validate(synthetic_features, fast_config)
    predictions = result.predictions
    assert len(predictions) == len(synthetic_features.y)
    assert predictions["path"].tolist() == synthetic_features.paths.tolist()
    assert sorted(predictions["fold"].unique().tolist()) == [0, 1, 2, 3]
    assert (predictions.groupby("group")["fold"].nunique() == 1).all()
    assert result.folds["n_test"].sum() == len(synthetic_features.y)


def test_fold_metrics_match_reference_implementations(synthetic_features, fast_config):
    result = cross_validate(synthetic_features, fast_config)
    predictions = result.predictions
    for row in result.folds.itertuples():
        part = predictions[predictions["fold"] == row.fold]
        truth, guess = part["severity"].to_numpy(), part["predicted"].to_numpy()
        assert math.isclose(row.mae, mean_absolute_error(truth, guess), rel_tol=1e-10)
        assert math.isclose(row.rmse, np.sqrt(mean_squared_error(truth, guess)), rel_tol=1e-10)
        assert math.isclose(row.r2, r2_score(truth, guess), rel_tol=1e-10)
        assert math.isclose(row.pearson_r, stats.pearsonr(truth, guess)[0], rel_tol=1e-10)
        assert math.isclose(row.spearman_rho, stats.spearmanr(truth, guess)[0], rel_tol=1e-10)
        assert math.isclose(row.ccc, concordance_ccc(truth, guess), rel_tol=1e-10)
    summary = result.metrics["summary"]["rmse"]
    assert math.isclose(summary["std"], float(np.std(result.folds["rmse"], ddof=1)), rel_tol=1e-9)
    assert "bland_altman" in result.metrics["pooled"]


def test_classification_metrics_match_scikit_learn(rng):
    classes = ["dysphonic", "healthy"]
    truth = rng.choice(classes, 60)
    scores = rng.uniform(size=60)
    guess = np.where(scores > 0.5, "dysphonic", "healthy")
    probabilities = np.column_stack([scores, 1.0 - scores])
    metrics = classification_metrics(
        truth, guess, probabilities, classes=classes, positive_class="dysphonic"
    )
    assert math.isclose(metrics["uar"], balanced_accuracy_score(truth, guess))
    assert math.isclose(metrics["f1_macro"], f1_score(truth, guess, average="macro"))
    assert math.isclose(metrics["roc_auc"], roc_auc_score(truth == "dysphonic", scores))
    positive = truth == "dysphonic"
    assert math.isclose(metrics["sensitivity"], np.mean(guess[positive] == "dysphonic"))
    assert math.isclose(metrics["specificity"], np.mean(guess[~positive] == "healthy"))


def test_single_class_set_averages_only_the_class_that_occurs():
    classes = ["dysphonic", "healthy"]
    truth = ["dysphonic"] * 4
    perfect = classification_metrics(truth, truth, classes=classes, positive_class="dysphonic")
    assert perfect["uar"] == 1.0
    # No healthy item: the sensitivity is still defined, the specificity and ROC-AUC are not.
    assert perfect["sensitivity"] == 1.0
    assert math.isnan(perfect["specificity"]) and math.isnan(perfect["roc_auc"])
    guess = ["dysphonic", "dysphonic", "dysphonic", "healthy"]
    one_miss = classification_metrics(truth, guess, classes=classes, positive_class="dysphonic")
    # 'healthy' is only predicted, so the UAR is the recall of 'dysphonic' alone.
    assert one_miss["uar"] == pytest.approx(0.75)
    assert one_miss["uar"] == pytest.approx(speechdsp.uar(truth, guess))
    assert one_miss["sensitivity"] == pytest.approx(0.75)


def test_negative_only_set_has_a_specificity_but_no_sensitivity():
    truth = ["healthy"] * 4
    guess = ["healthy", "healthy", "dysphonic", "healthy"]
    metrics = classification_metrics(
        truth, guess, classes=["dysphonic", "healthy"], positive_class="dysphonic"
    )
    assert metrics["specificity"] == pytest.approx(0.75)
    assert math.isnan(metrics["sensitivity"]) and math.isnan(metrics["roc_auc"])


def test_three_classes_have_no_binary_metrics():
    metrics = classification_metrics(["a", "b", "c"], ["a", "b", "b"], classes=["a", "b", "c"])
    assert list(metrics) == ["uar", "accuracy", "f1_macro"]


def test_classification_cross_validation(synthetic_features, classification_config):
    features = FeatureSet(
        X=synthetic_features.X,
        y=np.where(synthetic_features.y >= 35.0, "dysphonic", "healthy"),
        groups=synthetic_features.groups,
        paths=synthetic_features.paths,
        config=synthetic_features.config,
    )
    result = cross_validate(features, classification_config)
    pooled = result.metrics["pooled"]
    assert pooled["classes"] == ["dysphonic", "healthy"]
    assert int(np.sum(pooled["confusion_matrix"])) == len(features.y)
    assert set(pooled["per_class"]["healthy"]) == {"recall", "precision", "f1", "support"}
    scores = result.predictions[["score_dysphonic", "score_healthy"]].to_numpy()
    assert np.allclose(scores.sum(axis=1), 1.0)
    assert (result.predictions.groupby("group")["fold"].nunique() == 1).all()


def test_unknown_positive_class_is_a_configuration_error(synthetic_features, classification_config):
    features = FeatureSet(
        X=synthetic_features.X,
        y=np.where(synthetic_features.y >= 35.0, "dysphonic", "healthy"),
        groups=synthetic_features.groups,
        paths=synthetic_features.paths,
        config=synthetic_features.config,
    )
    config = classification_config.override({"evaluation.positive_class": "sick"})
    with pytest.raises(ConfigError, match=r"evaluation\.positive_class"):
        cross_validate(features, config)


def test_regression_metrics_handle_constant_predictions():
    metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert math.isnan(metrics["pearson_r"]) and math.isnan(metrics["spearman_rho"])
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)


def test_summary_uses_the_sample_standard_deviation():
    assert summarize([1.0, 2.0, 3.0]) == {"mean": 2.0, "std": 1.0}
    assert math.isnan(summarize([4.0])["std"])
