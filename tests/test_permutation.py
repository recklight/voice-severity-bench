"""Speaker-level label permutation and the permutation test of a classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from voice_severity_bench import (
    Config,
    ConfigError,
    DatasetError,
    compare_feature_sets,
    feature_set_from_table,
    permutation_test,
    permute_group_labels,
)


def _table(signal: float, n_speakers: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.repeat(["healthy", "dysphonic"], n_speakers // 2)
    rows = {
        "path": [f"r{i}.wav" for i in range(2 * n_speakers)],
        "group": np.repeat([f"s{i}" for i in range(n_speakers)], 2),
        "label": np.repeat(labels, 2),
    }
    table = pd.DataFrame(rows)
    table["x"] = (table["label"] == "dysphonic") * signal + rng.normal(0, 1, 2 * n_speakers)
    return table


def _features(table: pd.DataFrame):
    return feature_set_from_table(
        table, target="label", feature_columns=["x"], task="classification"
    )


def _config(**extra) -> Config:
    return Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": "logistic",
            "evaluation.n_splits": 3,
            "evaluation.n_repeats": 1,
            "analysis.n_resamples": 100,
        }
        | extra
    )


def test_every_recording_of_a_speaker_keeps_one_label():
    table = _table(0.0)
    permuted = permute_group_labels(table["label"], table["group"], rng=1)
    per_speaker = pd.Series(permuted).groupby(table["group"].to_numpy()).nunique()
    assert per_speaker.max() == 1
    assert not np.array_equal(permuted, table["label"].to_numpy())


def test_speakers_keep_the_label_counts():
    table = _table(0.0)
    permuted = permute_group_labels(table["label"], table["group"], rng=2)
    assert np.unique(permuted, return_counts=True)[1].tolist() == [24, 24]


def test_unequal_speakers_change_recording_counts_not_speaker_counts():
    y = ["sick"] * 4 + ["well"] * 2
    groups = ["a", "a", "a", "a", "b", "c"]
    results = {tuple(permute_group_labels(y, groups, rng=seed).tolist()) for seed in range(30)}
    assert ("well",) * 4 + ("sick", "well") in results
    assert all(result[0] == result[3] for result in results)


def test_labels_only_move_inside_a_stratum():
    table = _table(0.0)
    strata = np.where(
        np.arange(48) < 12, "corpus_a", np.where(np.arange(48) < 24, "corpus_b", "corpus_c")
    )
    original = pd.DataFrame({"stratum": strata, "label": table["label"]})
    for seed in range(5):
        permuted = pd.DataFrame(
            {
                "stratum": strata,
                "label": permute_group_labels(table["label"], table["group"], strata, seed),
            }
        )
        pd.testing.assert_series_equal(
            original.groupby("stratum")["label"].value_counts().sort_index(),
            permuted.groupby("stratum")["label"].value_counts().sort_index(),
        )


def test_the_same_seed_gives_the_same_permutation():
    table = _table(0.0)
    first = permute_group_labels(table["label"], table["group"], rng=7)
    second = permute_group_labels(table["label"], table["group"], rng=np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_items_without_a_group_are_speakers_of_their_own():
    y = ["sick", "well", "sick", "well"]
    results = {
        tuple(permute_group_labels(y, ["", "", "", ""], rng=seed).tolist()) for seed in range(30)
    }
    assert len(results) > 1
    assert len({tuple(permute_group_labels(y, None, rng=seed).tolist()) for seed in range(30)}) > 1


def test_a_speaker_with_two_labels_is_refused():
    with pytest.raises(DatasetError, match="more than one label"):
        permute_group_labels(["sick", "well"], ["a", "a"], rng=0)


def test_a_speaker_in_two_strata_is_refused():
    with pytest.raises(DatasetError, match="more than one stratum"):
        permute_group_labels(["sick", "sick"], ["a", "a"], ["x", "y"], 0)


def test_a_clear_signal_reaches_the_smallest_possible_p_value():
    result = permutation_test(_features(_table(4.0)), _config(), n_permutations=19, seed=1)
    assert result.p_value == pytest.approx(1 / 20)
    assert result.observed > 0.9
    assert result.null.shape == (19,)
    assert result.n_undefined == 0
    assert result.as_dict()["n_permutations"] == 19


def test_no_signal_leaves_the_observed_score_inside_the_null():
    result = permutation_test(_features(_table(0.0, seed=3)), _config(), n_permutations=29, seed=2)
    assert result.p_value > 0.05
    assert abs(float(np.mean(result.null)) - 0.5) < 0.1


def test_the_p_value_counts_the_null_scores_at_or_above_the_observed_one():
    result = permutation_test(_features(_table(0.5, seed=4)), _config(), n_permutations=29, seed=5)
    expected = (1 + np.count_nonzero(result.null >= result.observed)) / 30
    assert result.p_value == pytest.approx(expected)


def test_the_observed_score_is_the_one_the_comparison_reports():
    features = _features(_table(1.5, seed=6))
    config = _config(**{"evaluation.n_repeats": 2, "model.tune": True})
    result = permutation_test(features, config, n_permutations=1, seed=0)
    summary = compare_feature_sets({"x": features}, config).summary
    assert result.observed == pytest.approx(float(summary["uar"].iloc[0]))


def test_every_permutation_tunes_again(monkeypatch):
    searches = []

    class CountedSearch(GridSearchCV):
        def fit(self, *args, **kwargs):
            searches.append(1)
            return super().fit(*args, **kwargs)

    monkeypatch.setattr("voice_severity_bench.models.GridSearchCV", CountedSearch)
    features = _features(_table(1.0, seed=7))
    permutation_test(features, _config(**{"model.tune": True}), n_permutations=5, seed=3)
    # The real labels and five permutations, each over three outer folds.
    assert len(searches) == (1 + 5) * 3


def test_results_do_not_depend_on_the_number_of_jobs():
    features = _features(_table(1.0, seed=8))
    one = permutation_test(features, _config(), n_permutations=6, seed=4, n_jobs=1)
    two = permutation_test(features, _config(), n_permutations=6, seed=4, n_jobs=2)
    assert np.array_equal(one.null, two.null)


def test_other_metrics_use_the_positive_class():
    result = permutation_test(
        _features(_table(4.0)),
        _config(**{"evaluation.positive_class": "dysphonic"}),
        n_permutations=9,
        metric="roc_auc",
        seed=1,
    )
    assert result.metric == "roc_auc"
    assert result.observed > 0.9
    assert result.p_value == pytest.approx(0.1)


def test_permuting_inside_strata_keeps_their_share_in_the_null():
    table = _table(0.0, n_speakers=40, seed=9)
    # The stratum alone predicts the label, so the signal survives this null.
    speakers = np.repeat(np.arange(40), 2)
    strata = np.where(speakers % 40 < 18, "a", "b")
    table["label"] = np.where(
        (strata == "a") == (np.repeat(np.arange(40) % 10 != 0, 2)), "dysphonic", "healthy"
    )
    table["x"] = (strata == "a") * 4.0 + np.random.default_rng(0).normal(0, 1, 80)
    features = _features(table)
    free = permutation_test(features, _config(), n_permutations=19, seed=1)
    inside = permutation_test(features, _config(), n_permutations=19, strata=strata, seed=1)
    assert free.observed == inside.observed
    assert float(np.mean(inside.null)) > float(np.mean(free.null)) + 0.15
    assert inside.p_value > free.p_value


def test_folds_that_a_permutation_cannot_fill_are_counted_not_fatal():
    table = _table(1.0, n_speakers=24, seed=10)
    speakers = np.repeat(np.arange(24), 2)
    # Three speakers of each label per fold in "rare"; permuting freely can empty a cell.
    split_strata = np.where(np.isin(speakers, [0, 1, 2, 12, 13, 14]), "rare", "common")
    result = permutation_test(
        _features(table), _config(), n_permutations=15, split_strata=split_strata, seed=2
    )
    assert result.n_undefined + np.count_nonzero(np.isfinite(result.null)) == 15
    assert result.n_undefined > 0
    defined = result.null[np.isfinite(result.null)]
    expected = (1 + np.count_nonzero(defined >= result.observed)) / (1 + defined.size)
    assert result.p_value == pytest.approx(expected)


def test_regression_is_refused():
    table = _table(1.0)
    table["severity"] = np.arange(48.0)
    features = feature_set_from_table(table, target="severity", feature_columns=["x"])
    with pytest.raises(ConfigError, match="classification"):
        permutation_test(features, Config(), n_permutations=2)


def test_a_binary_metric_with_three_classes_is_refused():
    table = _table(1.0)
    table.loc[table.index[:16], "label"] = "other"
    with pytest.raises(ConfigError, match="two classes"):
        permutation_test(_features(table), _config(), n_permutations=2, metric="roc_auc")


def test_zero_permutations_are_refused():
    with pytest.raises(ValueError, match="at least 1"):
        permutation_test(_features(_table(1.0)), _config(), n_permutations=0)


def test_a_speaker_with_two_labels_stops_the_test_before_any_permutation():
    table = _table(1.0)
    table.loc[0, "label"] = "dysphonic"
    with pytest.raises(DatasetError, match="more than one label"):
        permutation_test(_features(table), _config(), n_permutations=2)
