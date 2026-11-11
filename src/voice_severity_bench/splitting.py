"""Cross-validation splits that respect speakers and class balance.

* Classification uses stratified folds, so every fold keeps the class
  proportions; with a ``group`` column the folds are also grouped
  (``StratifiedGroupKFold``), so no speaker contributes to both sides. An
  optional stratum (corpus, sex, age band) is crossed with the label, so the
  folds keep the balance of the combinations as well.
* Regression uses shuffled k-fold, or, with groups, puts whole groups on the
  folds largest first, each on the fold with the fewest items so far (the
  ``GroupKFold`` rule). Groups of equal size are taken in an order drawn from
  the seed, so repeated splits differ and do not depend on how a NumPy release
  orders ties when sorting.

The splits depend only on the targets, the groups and the seed, not on the
features, so every feature setting of an ablation study is evaluated on the
same folds. That is what allows the settings to be compared fold by fold in a
paired test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from .errors import DatasetError

#: Learning tasks accepted by ``data.task``.
TASKS: tuple[str, ...] = ("regression", "classification")

Split = tuple[NDArray[np.intp], NDArray[np.intp]]


def has_groups(groups: ArrayLike | None) -> bool:
    """Whether a group vector carries information (not missing, not all empty).

    Examples
    --------
    >>> has_groups(None), has_groups(["", ""]), has_groups(["s1", "s2"])
    (False, False, True)
    """
    if groups is None:
        return False
    values = np.asarray(groups).astype(str)
    return bool(values.size) and bool(np.any(values != ""))


def split_indices(
    y: ArrayLike,
    groups: ArrayLike | None = None,
    *,
    task: str,
    n_splits: int,
    random_state: int,
    strata: ArrayLike | None = None,
) -> list[Split]:
    """Train and test indices of every fold.

    Parameters
    ----------
    y : array_like, shape (n_items,)
        Targets (class names or ratings).
    groups : array_like, shape (n_items,), optional
        Speaker or session of every item; empty strings mean "no grouping".
    task : {"regression", "classification"}
        Chooses stratification.
    n_splits : int
        Number of folds.
    random_state : int
        Seed of the shuffle.
    strata : array_like, shape (n_items,), optional
        Classification only: a stratum per item; the folds are stratified on
        label x stratum instead of the label.

    Returns
    -------
    list of tuple of numpy.ndarray
        ``(train, test)`` index arrays; every item is tested exactly once.

    Raises
    ------
    DatasetError
        If there are fewer items or groups than folds, or a class (or label x
        stratum cell) has fewer members than folds. With groups the members
        are counted in speakers: ten recordings of three speakers cannot fill
        five folds.

    Examples
    --------
    >>> y = np.repeat(["healthy", "dysphonic"], 6)
    >>> groups = np.repeat([f"s{i}" for i in range(6)], 2)
    >>> folds = split_indices(y, groups, task="classification", n_splits=3, random_state=0)
    >>> sorted(int(i) for _, test in folds for i in test) == list(range(12))
    True
    >>> all(not set(groups[a]) & set(groups[b]) for a, b in folds)
    True
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose one of {list(TASKS)}")
    targets = np.asarray(y)
    n_items = targets.shape[0]
    if n_splits > n_items:
        raise DatasetError(f"cannot make {n_splits} folds from {n_items} items")
    grouped = has_groups(groups)
    group_codes = None
    if grouped:
        labels = np.asarray(groups).astype(str)
        unique, group_codes = np.unique(labels, return_inverse=True)
        if unique.size < n_splits:
            raise DatasetError(
                f"cannot make {n_splits} grouped folds from {unique.size} groups; "
                "lower evaluation.n_splits or add speakers"
            )
    placeholder = np.zeros((n_items, 1))
    if strata is not None and task != "classification":
        raise ValueError("strata stratify classification folds; regression has none")
    if task == "classification":
        labels = _stratification_labels(targets, strata)
        _check_members(labels, group_codes, n_splits, crossed=strata is not None)
        if grouped:
            assert group_codes is not None
            # shuffle=True in scikit-learn 1.7 shuffles the table of class counts per group
            # without the group labels, so the folds lose the stratification. Shuffling the
            # labels here varies the folds the same way and keeps the counts right.
            order = np.random.default_rng(random_state).permutation(group_codes.max() + 1)
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
            folds = splitter.split(placeholder, labels, order[group_codes])
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = splitter.split(placeholder, labels)
    elif grouped:
        assert group_codes is not None
        folds = _grouped_folds(group_codes, n_splits, random_state)
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(placeholder)
    return [(np.asarray(train), np.asarray(test)) for train, test in folds]


def _stratification_labels(targets: NDArray[Any], strata: ArrayLike | None) -> NDArray[np.str_]:
    labels = targets.astype(str)
    if strata is None:
        return labels
    extra = np.asarray(strata).astype(str)
    if extra.shape != labels.shape:
        raise ValueError(f"strata must have shape {labels.shape}, got {extra.shape}")
    return np.char.add(np.char.add(labels, " x "), extra)


def _check_members(
    labels: NDArray[np.str_], group_codes: NDArray[np.intp] | None, n_splits: int, *, crossed: bool
) -> None:
    """Every class or cell needs at least one member per fold, a member being a speaker if known."""
    cells, cell_codes = np.unique(labels, return_inverse=True)
    if group_codes is None:
        counts = np.bincount(cell_codes)
    else:
        pairs = np.unique(np.column_stack([cell_codes, group_codes]), axis=0)
        counts = np.bincount(pairs[:, 0], minlength=cells.size)
    smallest = int(counts.argmin())
    if counts[smallest] < n_splits:
        kind = f"label x stratum cell ({cells[smallest]})" if crossed else "class"
        members = "items" if group_codes is None else "speakers"
        advice = (
            "merge strata or lower the number of folds" if crossed else "lower the number of folds"
        )
        raise DatasetError(
            f"the smallest {kind} has {int(counts[smallest])} {members}, fewer than the "
            f"{n_splits} folds; {advice}"
        )


def _grouped_folds(codes: NDArray[np.intp], n_splits: int, random_state: int) -> list[Split]:
    """Folds that keep every group whole, balanced by item count.

    Same rule as ``GroupKFold``: largest group first, each onto the fold with
    the fewest items so far (the first such fold on a tie). ``GroupKFold``
    breaks ties between equal-sized groups by the order ``np.argsort`` leaves
    them in, which differs between NumPy releases, so the same seed would give
    different folds on different installations. Here a seeded rank of each
    group breaks the tie.
    """
    sizes = np.bincount(codes)
    rank = np.random.default_rng(random_state).permutation(sizes.size)
    # Ranks are unique, so this order has no ties left to break.
    order = np.lexsort((-rank, -sizes))
    load = np.zeros(n_splits, dtype=np.int64)
    fold_of_group = np.empty(sizes.size, dtype=np.intp)
    for group in order:
        fold = int(np.argmin(load))
        fold_of_group[group] = fold
        load[fold] += sizes[group]
    fold_of_item = fold_of_group[codes]
    items = np.arange(codes.size)
    return [(items[fold_of_item != fold], items[fold_of_item == fold]) for fold in range(n_splits)]


def repeated_split_indices(
    y: ArrayLike,
    groups: ArrayLike | None = None,
    *,
    task: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
    strata: ArrayLike | None = None,
) -> list[list[Split]]:
    """Folds of repeated cross-validation, one list of folds per repeat.

    Every repeat is :func:`split_indices` with its own seed from
    :func:`repeat_seeds`, so two analyses with the same targets, groups and
    master seed get identical folds and their per-fold scores can be paired.

    Parameters
    ----------
    y, groups, task, n_splits
        As in :func:`split_indices`; groups that carry no information are ignored.
    n_repeats : int
        Number of repetitions.
    seed : int
        Master seed.
    strata : array_like, shape (n_items,), optional
        As in :func:`split_indices`.

    Returns
    -------
    list of list of tuple of numpy.ndarray
        ``n_repeats`` lists of ``n_splits`` ``(train, test)`` pairs.

    Examples
    --------
    >>> y = np.arange(12.0)
    >>> groups = np.repeat([f"s{i}" for i in range(6)], 2)
    >>> splits = repeated_split_indices(
    ...     y, groups, task="regression", n_splits=3, n_repeats=2, seed=0
    ... )
    >>> len(splits), [len(folds) for folds in splits]
    (2, [3, 3])
    >>> tested = [sorted(np.concatenate([test for _, test in folds])) for folds in splits]
    >>> all(items == list(range(12)) for items in tested)
    True
    """
    usable = groups if has_groups(groups) else None
    return [
        split_indices(y, usable, task=task, n_splits=n_splits, random_state=child, strata=strata)
        for child in repeat_seeds(seed, n_repeats)
    ]


def repeat_seeds(seed: int, n_repeats: int) -> Sequence[int]:
    """Independent split seeds for repeated cross-validation, derived from one master seed.

    Examples
    --------
    >>> seeds = repeat_seeds(0, 3)
    >>> len(seeds), len(set(seeds)), list(repeat_seeds(0, 3)) == list(seeds)
    (3, 3, True)
    """
    children = np.random.SeedSequence(seed).spawn(n_repeats)
    return [int(child.generate_state(1)[0] % (2**31 - 1)) for child in children]
