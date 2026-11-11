"""Group bootstrap: weights, weighted metrics and paired intervals."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from voice_severity_bench import classification_metrics, regression_metrics
from voice_severity_bench.bootstrap import (
    BOOTSTRAP_METRICS,
    CLASSIFICATION_BOOTSTRAP_METRICS,
    bootstrap_metric,
    group_codes,
    group_weights,
    paired_bootstrap,
    weighted_metric,
)


@pytest.fixture
def ratings(rng):
    truth = rng.uniform(0, 100, 50)
    return truth, truth + rng.normal(0, 8, 50)


@pytest.mark.parametrize("metric", BOOTSTRAP_METRICS)
def test_unit_weights_reproduce_the_regression_metrics(ratings, metric):
    truth, predicted = ratings
    assert weighted_metric(truth, predicted, metric=metric) == pytest.approx(
        regression_metrics(truth, predicted)[metric], rel=1e-10
    )


@pytest.mark.parametrize("metric", BOOTSTRAP_METRICS)
def test_integer_weights_equal_repeating_the_items(ratings, metric, rng):
    truth, predicted = ratings
    counts = rng.integers(0, 3, truth.size)
    counts[0] = 1
    repeated = np.repeat(np.arange(truth.size), counts)
    assert weighted_metric(truth, predicted, counts, metric=metric) == pytest.approx(
        regression_metrics(truth[repeated], predicted[repeated])[metric], rel=1e-9
    )


def test_prediction_columns_are_scored_separately_then_averaged(ratings, rng):
    truth, predicted = ratings
    second = truth + rng.normal(0, 3, truth.size)
    both = np.column_stack([predicted, second])
    expected = (weighted_metric(truth, predicted) + weighted_metric(truth, second)) / 2
    assert weighted_metric(truth, both) == pytest.approx(expected)
    # Not the score of the averaged prediction, which would be lower.
    assert weighted_metric(truth, both.mean(axis=1)) < expected


def test_groups_are_drawn_whole_and_empty_labels_stand_alone():
    groups = np.array(["s1", "s1", "s1", "s2", "", ""])
    weights = group_weights(groups, 6, 300, seed=3)
    assert np.all(weights[:, 0] == weights[:, 1]) and np.all(weights[:, 1] == weights[:, 2])
    assert not np.all(weights[:, 4] == weights[:, 5])
    # Four groups drawn four times per resample.
    assert np.all(weights[:, [0, 3, 4, 5]].sum(axis=1) == 4)
    assert np.array_equal(weights, group_weights(groups, 6, 300, seed=3))


def test_paired_interval_is_narrower_than_the_separate_ones(rng):
    truth = rng.uniform(0, 100, 120)
    shared = rng.normal(0, 10, 120)
    first = truth + shared + rng.normal(0, 1, 120)
    second = truth + shared + rng.normal(0, 2, 120)
    paired = paired_bootstrap(truth, first, second, n_resamples=1000, seed=1)
    a = bootstrap_metric(truth, first, n_resamples=1000, seed=1)
    b = bootstrap_metric(truth, second, n_resamples=1000, seed=1)
    assert paired.estimate == pytest.approx(a.estimate - b.estimate)
    assert paired.high - paired.low < 0.5 * ((a.high - a.low) + (b.high - b.low))
    assert np.allclose(paired.samples, a.samples - b.samples)


def test_identical_systems_differ_by_exactly_zero(ratings):
    truth, predicted = ratings
    result = paired_bootstrap(truth, predicted, predicted, n_resamples=200)
    assert result.estimate == 0.0 and result.low == 0.0 and result.high == 0.0


def test_intervals_cover_the_truth_in_most_samples():
    # 200 samples of 40 speakers x 3 recordings. Errors share a speaker part
    # (sd 6) besides their own (sd 8), so the population RMSE is 10.
    hits = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        speaker = rng.normal(0, 6, 40).repeat(3)
        truth = rng.uniform(0, 100, 120)
        predicted = truth + speaker + rng.normal(0, 8, 120)
        groups = np.repeat([f"s{i}" for i in range(40)], 3)
        result = bootstrap_metric(truth, predicted, groups, n_resamples=400, seed=seed)
        hits += result.low <= 10.0 <= result.high
    assert hits >= 170


def test_bad_inputs_are_explained():
    with pytest.raises(ValueError, match="unknown metric"):
        weighted_metric([1.0, 2.0], [1.0, 2.0], metric="f1_macro")
    with pytest.raises(ValueError, match="predictions"):
        weighted_metric([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="groups"):
        group_weights(["a"], 2, 10, seed=0)


@pytest.fixture
def labels(rng):
    truth = np.repeat(["healthy", "dysphonic"], [30, 20])
    # Scores rounded to one decimal, so the ROC-AUC has ties to deal with.
    score = np.round(np.clip((truth == "dysphonic") * 0.3 + rng.uniform(0.1, 0.7, 50), 0, 1), 1)
    return truth, np.where(score >= 0.5, "dysphonic", "healthy"), score


def _scored(truth, predicted, score):
    return classification_metrics(
        truth,
        predicted,
        np.column_stack([score, 1.0 - score]),
        classes=["dysphonic", "healthy"],
        positive_class="dysphonic",
    )


@pytest.mark.parametrize("metric", CLASSIFICATION_BOOTSTRAP_METRICS)
def test_unit_weights_reproduce_the_classification_metrics(labels, metric):
    truth, predicted, score = labels
    prediction = score if metric == "roc_auc" else predicted
    assert weighted_metric(
        truth, prediction, metric=metric, positive_class="dysphonic"
    ) == pytest.approx(_scored(truth, predicted, score)[metric], rel=1e-12)


@pytest.mark.parametrize("metric", CLASSIFICATION_BOOTSTRAP_METRICS)
def test_integer_weights_equal_repeating_the_recordings(labels, metric, rng):
    truth, predicted, score = labels
    counts = rng.integers(0, 3, truth.size)
    counts[[0, -1]] = 1
    repeated = np.repeat(np.arange(truth.size), counts)
    prediction = score if metric == "roc_auc" else predicted
    assert weighted_metric(
        truth, prediction, counts, metric=metric, positive_class="dysphonic"
    ) == pytest.approx(
        _scored(truth[repeated], predicted[repeated], score[repeated])[metric], rel=1e-12
    )


def test_the_weighted_auc_agrees_with_scikit_learn_on_tied_scores(labels, rng):
    truth, _, score = labels
    weights = rng.integers(0, 5, (6, truth.size)).astype(float)
    weights[:, [0, -1]] = 1.0
    expected = [roc_auc_score(truth == "dysphonic", score, sample_weight=row) for row in weights]
    computed = weighted_metric(truth, score, weights, metric="roc_auc", positive_class="dysphonic")
    assert computed == pytest.approx(expected, rel=1e-12)


def test_several_repetitions_are_averaged(labels, rng):
    truth, predicted, score = labels
    other = rng.permutation(predicted)
    matrix = np.column_stack([predicted, other])
    assert weighted_metric(truth, matrix, metric="uar") == pytest.approx(
        np.mean(
            [
                weighted_metric(truth, predicted, metric="uar"),
                weighted_metric(truth, other, metric="uar"),
            ]
        )
    )
    scores = np.column_stack([score, rng.permutation(score)])
    assert weighted_metric(truth, scores, metric="roc_auc", positive_class="dysphonic") == (
        pytest.approx(
            np.mean(
                [
                    weighted_metric(truth, column, metric="roc_auc", positive_class="dysphonic")
                    for column in scores.T
                ]
            )
        )
    )


def test_more_than_two_classes_have_a_uar_and_an_accuracy():
    truth = np.array(["a", "a", "b", "b", "c", "c"])
    guess = np.array(["a", "b", "b", "b", "c", "a"])
    assert weighted_metric(truth, guess, metric="uar") == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert weighted_metric(truth, guess, metric="accuracy") == pytest.approx(4 / 6)


@pytest.mark.parametrize(
    ("metric", "kept"),
    [("uar", [0, 1]), ("sensitivity", [2, 3]), ("specificity", [0, 1]), ("roc_auc", [0, 1])],
)
def test_a_resample_that_loses_a_needed_class_is_undefined(metric, kept):
    truth = ["sick", "sick", "well", "well"]
    weights = np.zeros((2, 4))
    weights[0] = 1.0
    weights[1, kept] = 1.0
    prediction = [0.9, 0.2, 0.3, 0.1] if metric == "roc_auc" else ["sick", "well", "well", "well"]
    values = weighted_metric(truth, prediction, weights, metric=metric, positive_class="sick")
    assert np.isfinite(values[0]) and np.isnan(values[1])


def test_accuracy_survives_a_resample_with_one_class():
    assert weighted_metric(["a", "b"], ["a", "a"], [1.0, 0.0], metric="accuracy") == 1.0


def test_resamples_without_a_class_are_counted_and_left_out_of_the_interval(rng):
    # Two speakers of the rare class: drawn out together now and then.
    truth = np.array(["sick"] * 2 + ["well"] * 10)
    groups = np.array(["a", "b"] + [f"w{i}" for i in range(10)])
    guess = np.where(rng.random(12) < 0.8, truth, "well")
    result = bootstrap_metric(truth, guess, groups, metric="uar", n_resamples=2000, seed=3)
    assert result.n_undefined == int(np.count_nonzero(np.isnan(result.samples)))
    assert 0 < result.n_undefined < 2000
    assert np.isfinite(result.low) and np.isfinite(result.high)
    assert result.as_dict()["n_undefined"] == result.n_undefined


def test_nothing_undefined_with_well_filled_classes(labels):
    truth, predicted, _ = labels
    result = bootstrap_metric(truth, predicted, metric="uar", n_resamples=300, seed=1)
    assert result.n_undefined == 0
    assert result.low < result.estimate < result.high


def test_regression_resamples_are_counted_too():
    result = bootstrap_metric(
        np.arange(6.0), np.ones(6), metric="pearson_r", n_resamples=50, seed=0
    )
    assert result.n_undefined == 50 and np.isnan(result.low)


def test_paired_classification_intervals_use_the_same_resamples(labels, rng):
    truth, predicted, score = labels
    weaker = np.where(rng.random(truth.size) < 0.4, rng.permutation(predicted), predicted)
    result = paired_bootstrap(
        truth, predicted, weaker, metric="uar", n_resamples=400, seed=2, positive_class="dysphonic"
    )
    separately = [
        bootstrap_metric(truth, candidate, metric="uar", n_resamples=400, seed=2).samples
        for candidate in (predicted, weaker)
    ]
    assert result.samples == pytest.approx(separately[0] - separately[1])
    worse = paired_bootstrap(
        truth, score, 1.0 - score, metric="roc_auc", n_resamples=200, positive_class="dysphonic"
    )
    assert worse.estimate > 0 and worse.low > 0


@pytest.mark.parametrize("metric", ["sensitivity", "specificity", "roc_auc"])
def test_one_against_the_rest_metrics_need_two_classes_and_a_positive_one(metric):
    with pytest.raises(ValueError, match="positive_class"):
        weighted_metric(["a", "b"], ["a", "b"], metric=metric)
    with pytest.raises(ValueError, match="positive_class"):
        weighted_metric(["a", "b"], ["a", "b"], metric=metric, positive_class="c")
    with pytest.raises(ValueError, match="two classes"):
        weighted_metric(["a", "b", "c"], ["a", "b", "c"], metric=metric, positive_class="a")


def test_classification_shapes_are_checked():
    with pytest.raises(ValueError, match="predictions"):
        weighted_metric(["a", "b"], ["a", "b", "a"], metric="uar")
    with pytest.raises(ValueError, match="weights"):
        weighted_metric(["a", "b"], ["a", "b"], [1.0, 1.0, 1.0], metric="uar")


def test_group_codes_treat_an_empty_label_as_a_speaker_of_its_own():
    codes = group_codes(["a", "", "a", ""], 4)
    assert codes[0] == codes[2] and codes[1] != codes[3]
    assert group_codes(None, 3).tolist() == [0, 1, 2]
    assert group_codes(["", ""], 2).tolist() == [0, 1]
    with pytest.raises(ValueError, match="groups"):
        group_codes(["a"], 2)
