"""Cross-validation splits: grouping, stratification, repeats and clear failures."""

from __future__ import annotations

import numpy as np
import pytest

from voice_severity_bench import DatasetError, split_indices
from voice_severity_bench.splitting import has_groups, repeat_seeds, repeated_split_indices


def test_grouped_regression_folds_never_split_a_speaker(rng):
    groups = np.repeat([f"s{i}" for i in range(10)], 3)
    y = rng.uniform(0, 100, 30)
    folds = split_indices(y, groups, task="regression", n_splits=5, random_state=0)
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(30))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def _speaker_folds(groups, seed):
    folds = split_indices(np.zeros(groups.size), groups, task="regression", n_splits=5,
                          random_state=seed)  # fmt: skip
    return [sorted({int(name[1:]) for name in groups[test]}) for _, test in folds]


def test_equal_sized_speakers_get_the_same_folds_everywhere():
    # 20 speakers x 3 recordings is the default synthetic dataset, where every
    # group ties with every other one on size.
    groups = np.repeat([f"s{i:02d}" for i in range(20)], 3)
    assert _speaker_folds(groups, 0) == [
        [0, 1, 16, 17],
        [4, 6, 9, 14],
        [3, 11, 12, 15],
        [2, 5, 7, 18],
        [8, 10, 13, 19],
    ]


def test_grouped_folds_do_not_depend_on_the_tie_order_of_argsort(monkeypatch):
    groups = np.repeat([f"s{i:02d}" for i in range(20)], 3)
    expected = _speaker_folds(groups, 0)
    real_argsort = np.argsort

    def ties_last_first(a, *args, **kwargs):
        values = np.asarray(a)
        if values.ndim != 1:
            return real_argsort(a, *args, **kwargs)
        return values.size - 1 - real_argsort(values[::-1], kind="stable")

    monkeypatch.setattr(np, "argsort", ties_last_first)
    assert _speaker_folds(groups, 0) == expected


def test_grouped_folds_put_large_speakers_first_and_balance_the_folds():
    sizes = [5, 1, 3, 3, 2, 4, 1]
    groups = np.repeat([f"g{i}" for i in range(len(sizes))], sizes)
    folds = split_indices(np.zeros(groups.size), groups, task="regression", n_splits=3,
                          random_state=4)  # fmt: skip
    assert [sorted(set(groups[test])) for _, test in folds] == [
        ["g0", "g1", "g6"],
        ["g4", "g5"],
        ["g2", "g3"],
    ]
    assert [test.size for _, test in folds] == [7, 6, 6]


def test_repeats_with_groups_differ(rng):
    groups = np.repeat([f"s{i}" for i in range(12)], 2)
    y = rng.uniform(size=24)
    first, second = (
        split_indices(y, groups, task="regression", n_splits=4, random_state=seed)
        for seed in repeat_seeds(0, 2)
    )
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(first, second, strict=True))


def test_stratified_folds_keep_class_proportions():
    y = np.array(["a"] * 20 + ["b"] * 10)
    folds = split_indices(y, None, task="classification", n_splits=5, random_state=1)
    for _, test in folds:
        assert (y[test] == "b").sum() == 2


def test_impossible_splits_are_explained():
    with pytest.raises(DatasetError, match="groups"):
        split_indices(np.arange(6.0), ["a", "a", "b", "b", "c", "c"], task="regression",
                      n_splits=4, random_state=0)  # fmt: skip
    with pytest.raises(DatasetError, match="smallest class"):
        split_indices(np.array(["a"] * 8 + ["b"] * 2), None, task="classification",
                      n_splits=3, random_state=0)  # fmt: skip
    with pytest.raises(DatasetError, match="folds from"):
        split_indices(np.arange(3.0), None, task="regression", n_splits=4, random_state=0)


def test_group_detection():
    assert not has_groups(None) and not has_groups(np.array(["", ""]))
    assert has_groups(["a", ""])


def test_repeated_splits_match_one_split_per_repeat_seed(rng):
    # run_ablation and compare_feature_sets both rely on this to share folds.
    groups = np.repeat([f"s{i}" for i in range(9)], 2)
    y = rng.uniform(0, 100, 18)
    repeated = repeated_split_indices(y, groups, task="regression", n_splits=3, n_repeats=4, seed=7)
    for folds, seed in zip(repeated, repeat_seeds(7, 4), strict=True):
        single = split_indices(y, groups, task="regression", n_splits=3, random_state=seed)
        assert [test.tolist() for _, test in folds] == [test.tolist() for _, test in single]


def test_repeated_splits_ignore_empty_groups(rng):
    y = rng.uniform(0, 100, 10)
    empty = repeated_split_indices(y, [""] * 10, task="regression", n_splits=2, n_repeats=1, seed=0)
    none = repeated_split_indices(y, None, task="regression", n_splits=2, n_repeats=1, seed=0)
    assert [t.tolist() for _, t in empty[0]] == [t.tolist() for _, t in none[0]]


def _classes_and_speakers(per_class):
    labels, speakers = [], []
    for name, recordings_per_speaker in per_class.items():
        for index, count in enumerate(recordings_per_speaker):
            labels += [name] * count
            speakers += [f"{name}{index}"] * count
    return np.array(labels), np.array(speakers)


def test_grouped_classes_are_counted_in_speakers_not_recordings():
    y, groups = _classes_and_speakers({"sick": [8, 8], "well": [1] * 10})
    with pytest.raises(DatasetError, match="smallest class has 2 speakers"):
        split_indices(y, groups, task="classification", n_splits=3, random_state=0)
    # The same recordings without speakers can fill the folds, one by one.
    assert len(split_indices(y, None, task="classification", n_splits=3, random_state=0)) == 3


def test_enough_speakers_in_every_class_fill_the_folds():
    y, groups = _classes_and_speakers({"sick": [8, 8, 8], "well": [1] * 10})
    folds = split_indices(y, groups, task="classification", n_splits=3, random_state=0)
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert {"sick", "well"} == set(y[test])


def test_a_speaker_in_two_classes_counts_once_in_each():
    y = np.array(["sick", "well"] * 3)
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    assert len(split_indices(y, groups, task="classification", n_splits=3, random_state=0)) == 3


def test_strata_keep_the_balance_of_the_combinations():
    y = np.tile(np.repeat(["sick", "well"], 15), 2)
    strata = np.repeat(["corpus_a", "corpus_b"], 30)
    folds = split_indices(y, None, task="classification", n_splits=5, random_state=2, strata=strata)
    for _, test in folds:
        for label in ("sick", "well"):
            for stratum in ("corpus_a", "corpus_b"):
                assert np.count_nonzero((y[test] == label) & (strata[test] == stratum)) == 3


def test_strata_work_with_speakers():
    y = np.repeat(np.tile(["sick", "well"], 20), 2)
    groups = np.repeat([f"s{i}" for i in range(40)], 2)
    strata = np.repeat(np.tile(["a", "a", "b", "b"], 10), 2)
    folds = split_indices(
        y, groups, task="classification", n_splits=4, random_state=1, strata=strata
    )
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert {str(cell) for cell in np.char.add(y[test], strata[test])} == {
            "sicka", "sickb", "wella", "wellb"
        }  # fmt: skip


def test_an_empty_combination_is_explained_with_its_name():
    y = np.array(["sick"] * 12 + ["well"] * 12)
    strata = np.array(["a"] * 2 + ["b"] * 10 + ["a"] * 6 + ["b"] * 6)
    with pytest.raises(DatasetError, match=r"cell \(sick x a\) has 2 items.*merge strata"):
        split_indices(y, None, task="classification", n_splits=3, random_state=0, strata=strata)
    groups = np.array([f"s{i}" for i in range(24)])
    with pytest.raises(DatasetError, match="has 2 speakers"):
        split_indices(y, groups, task="classification", n_splits=3, random_state=0, strata=strata)


def test_strata_are_for_classification_and_must_cover_the_items():
    with pytest.raises(ValueError, match="regression"):
        split_indices(np.arange(10.0), None, task="regression", n_splits=2, random_state=0,
                      strata=["a"] * 10)  # fmt: skip
    with pytest.raises(ValueError, match="strata must have shape"):
        split_indices(np.array(["a", "b"] * 5), None, task="classification", n_splits=2,
                      random_state=0, strata=["x"] * 3)  # fmt: skip


def test_repeated_splits_pass_the_strata_on():
    y = np.tile(np.repeat(["sick", "well"], 10), 2)
    strata = np.repeat(["a", "b"], 20)
    splits = repeated_split_indices(y, None, task="classification", n_splits=5, n_repeats=2, seed=0,
                                    strata=strata)  # fmt: skip
    for folds in splits:
        for _, test in folds:
            assert np.count_nonzero((y[test] == "sick") & (strata[test] == "a")) == 2


@pytest.mark.parametrize("seed", range(4))
def test_grouped_folds_keep_the_class_balance(seed):
    y = np.repeat(np.tile(["sick", "well"], 20), 2)
    groups = np.repeat([f"s{i}" for i in range(40)], 2)
    folds = split_indices(y, groups, task="classification", n_splits=5, random_state=seed)
    assert [int(np.count_nonzero(y[test] == "sick")) for _, test in folds] == [8] * 5


def test_grouped_classification_folds_differ_between_seeds():
    y = np.repeat(np.tile(["sick", "well"], 20), 2)
    groups = np.repeat([f"s{i}" for i in range(40)], 2)
    first = split_indices(y, groups, task="classification", n_splits=5, random_state=0)
    second = split_indices(y, groups, task="classification", n_splits=5, random_state=1)
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(first, second, strict=True))
