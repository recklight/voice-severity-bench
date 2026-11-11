"""Speaker-level permutation tests of a cross-validated classifier.

A permutation test asks what score the whole procedure reaches when the labels
carry no information (Ojala & Garriga, 2010). Two things make it honest here:

* the label moves with the speaker. All recordings of a speaker keep one label,
  and speakers swap labels. Shuffling recordings one by one would break up each
  speaker's recordings and would test a different null hypothesis;
* every permutation repeats the full cross-validation, inner tuning and block
  choice included, because that selection is part of the procedure being tested.

Swapping only inside a stratum (corpus, sex, age band) keeps what the stratum
explains in the null, so the test asks whether the score goes beyond it.

The p-value counts the observed labelling as one of the permutations,
:math:`p = (1 + \\#\\{null \\ge observed\\}) / (1 + n)`, which cannot be zero
(Phipson & Smyth, 2010).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from .bootstrap import group_codes
from .comparison import BlockCandidates, _check_metric, _class_names, _fit_fold
from .config import Config
from .dataset import FeatureSet
from .errors import ConfigError, DatasetError
from .evaluation import classification_metrics, default_metric
from .splitting import has_groups, repeated_split_indices


@dataclass(frozen=True, eq=False)
class PermutationResult:
    """Observed score, null distribution and p-value of a permutation test.

    Attributes
    ----------
    metric : str
        The metric tested; higher is better for all of them.
    observed : float
        Score with the real labels.
    null : numpy.ndarray, shape (n_permutations,)
        Score of every permutation; ``nan`` where the permuted labels could
        not fill the folds.
    p_value : float
        :math:`(1 + \\#\\{null \\ge observed\\}) / (1 + n)`, with ``n`` the
        defined permutations.
    n_undefined : int
        Permutations left out of the p-value.
    """

    metric: str
    observed: float
    null: NDArray[np.float64]
    p_value: float
    n_undefined: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Headline numbers for JSON output (the null itself is not included)."""
        return {
            "metric": self.metric,
            "observed": self.observed,
            "p_value": self.p_value,
            "n_permutations": int(self.null.size),
            "n_undefined": self.n_undefined,
            "null_mean": float(np.nanmean(self.null))
            if self.n_undefined < self.null.size
            else None,
        }


def permute_group_labels(
    y: ArrayLike,
    groups: ArrayLike | None = None,
    strata: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
) -> NDArray[Any]:
    """Swap labels between speakers, optionally only inside each stratum.

    Parameters
    ----------
    y : array_like, shape (n_items,)
        Labels; all items of a speaker must share one.
    groups : array_like, shape (n_items,), optional
        Speakers. An empty label, or no groups at all, makes an item a speaker of its own.
    strata : array_like, shape (n_items,), optional
        Stratum of each item; all items of a speaker must share one.
    rng : numpy.random.Generator or int, optional
        Random source, or a seed for one.

    Returns
    -------
    numpy.ndarray, shape (n_items,)
        The permuted labels; each stratum keeps its count of speakers per label.

    Raises
    ------
    DatasetError
        If a speaker has two labels or two strata.

    Examples
    --------
    >>> y = ["sick", "sick", "well", "well", "sick", "well"]
    >>> groups = ["a", "a", "b", "b", "c", "d"]
    >>> permuted = permute_group_labels(y, groups, rng=3)
    >>> bool(permuted[0] == permuted[1]), bool(permuted[2] == permuted[3])
    (True, True)
    >>> sorted(permuted[[0, 2, 4, 5]].tolist())
    ['sick', 'sick', 'well', 'well']
    """
    labels = np.asarray(y)
    n_items = labels.shape[0]
    codes = group_codes(groups, n_items)
    _, first_item = np.unique(codes, return_index=True)
    _check_one_value(labels, codes, first_item, "label")
    speaker_labels = labels[first_item].copy()
    generator = np.random.default_rng(rng)
    if strata is None:
        return generator.permutation(speaker_labels)[codes]
    stratum = np.asarray(strata).astype(str)
    if stratum.shape != (n_items,):
        raise ValueError(f"strata must have shape ({n_items},), got {stratum.shape}")
    _check_one_value(stratum, codes, first_item, "stratum")
    speaker_strata = stratum[first_item]
    for name in np.unique(speaker_strata):
        inside = speaker_strata == name
        speaker_labels[inside] = generator.permutation(speaker_labels[inside])
    return speaker_labels[codes]


def _check_one_value(
    values: NDArray[Any], codes: NDArray[np.intp], first_item: NDArray[np.intp], what: str
) -> None:
    mixed = np.flatnonzero(values != values[first_item][codes])
    if mixed.size:
        raise DatasetError(
            f"{np.unique(codes[mixed]).size} speaker(s) have more than one {what}, for "
            f"example at item {int(mixed[0])}; a permutation moves one {what} per speaker"
        )


def _cross_validated_score(
    X: NDArray[np.float64],
    y: NDArray[np.str_],
    groups: NDArray[np.str_] | None,
    config: Config,
    metric: str,
    classes: Sequence[str],
    positive: str | None,
    blocks: Any,
    candidates: BlockCandidates | None,
    split_strata: ArrayLike | None,
) -> float:
    """The metric pooled per repetition and averaged, as in ``compare_feature_sets``."""
    splits = repeated_split_indices(
        y,
        groups,
        task="classification",
        n_splits=config.evaluation.n_splits,
        n_repeats=config.evaluation.n_repeats,
        seed=config.seed,
        strata=split_strata,
    )
    per_repeat = []
    for folds in splits:
        predicted = np.empty(y.size, dtype=object)
        scores = np.zeros((y.size, len(classes)))
        for train, test in folds:
            labels, scored, _, _ = _fit_fold(
                X, y, groups, train, test, config, blocks, candidates, classes
            )
            predicted[test] = labels
            scores[test] = scored
        pooled = classification_metrics(
            y, predicted.astype(str), scores, classes=classes, positive_class=positive
        )
        per_repeat.append(pooled[metric])
    return float(np.mean(per_repeat))


def _permuted_score(
    child: np.random.SeedSequence,
    X: NDArray[np.float64],
    y: NDArray[np.str_],
    groups: NDArray[np.str_] | None,
    strata: ArrayLike | None,
    **options: Any,
) -> float:
    permuted = permute_group_labels(y, groups, strata, np.random.default_rng(child))
    try:
        return _cross_validated_score(X, permuted, groups, **options)
    except DatasetError:
        # Permuting inside one stratum can empty a cell of another split stratum.
        return float("nan")


def permutation_test(
    features: FeatureSet,
    config: Config,
    *,
    n_permutations: int = 1000,
    metric: str | None = None,
    strata: ArrayLike | None = None,
    split_strata: ArrayLike | None = None,
    block_candidates: BlockCandidates | None = None,
    seed: int = 0,
    n_jobs: int = 1,
) -> PermutationResult:
    """Null distribution of a classifier's cross-validated score under speaker-level permutation.

    Parameters
    ----------
    features : FeatureSet
        One feature file with class names as targets.
    config : Config
        As in :func:`~voice_severity_bench.comparison.compare_feature_sets`. Every
        permutation costs one full run of ``evaluation.n_splits`` x
        ``evaluation.n_repeats`` fits, inner tuning included.
    n_permutations : int, optional
        Number of permutations.
    metric : str, optional
        ``uar`` (default), ``accuracy``, and for two classes ``sensitivity``,
        ``specificity``, ``roc_auc``.
    strata : array_like, shape (n_items,), optional
        Speakers swap labels only inside their stratum.
    split_strata : array_like, shape (n_items,), optional
        Stratify the folds on label x this; give what the comparison itself used.
    block_candidates : sequence, optional
        The blocks the inner loop chooses from.
    seed : int, optional
        Seed of the permutations; the folds come from ``config.seed``.
    n_jobs : int, optional
        Permutations run in parallel (joblib). Each worker gets ``cpu_count // n_jobs``
        BLAS threads, which can move the last bits of the probabilities; a p-value changes
        only if that flips a predicted class.

    Returns
    -------
    PermutationResult

    Raises
    ------
    ConfigError
        For regression, or a metric that does not fit the classes.
    DatasetError
        If a speaker has two labels or two strata, or the real labels cannot fill the folds.

    Examples
    --------
    >>> from voice_severity_bench.dataset import feature_set_from_table
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> table = pd.DataFrame({"path": [f"r{i}" for i in range(40)],
    ...                       "label": np.repeat(["sick", "well"], 20)})
    >>> table["signal"] = (table["label"] == "sick") * 3.0 + rng.normal(0, 1, 40)
    >>> features = feature_set_from_table(
    ...     table, target="label", feature_columns=["signal"], task="classification"
    ... )
    >>> config = Config().override({"data.task": "classification", "data.target": "label",
    ...                             "model.name": "logistic", "evaluation.positive_class": "sick",
    ...                             "evaluation.n_repeats": 1})
    >>> result = permutation_test(features, config, n_permutations=19, seed=1)
    >>> result.p_value
    0.05
    """
    if config.data.task != "classification":
        raise ConfigError("permutation_test works on classification targets")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")
    chosen = default_metric("classification") if metric is None else metric
    y = np.asarray(features.y).astype(str)
    classes, positive = _class_names(y, config)
    _check_metric(chosen, "classification", len(classes))
    X = np.asarray(features.X, dtype=np.float64)
    groups = np.asarray(features.groups).astype(str)
    usable = groups if has_groups(groups) else None
    options = {
        "config": config,
        "metric": chosen,
        "classes": classes,
        "positive": positive,
        "blocks": features.blocks,
        "candidates": block_candidates,
        "split_strata": split_strata,
    }
    # A speaker with two labels should stop the test here, not inside the workers.
    permute_group_labels(y, usable, strata, 0)
    observed = _cross_validated_score(X, y, usable, **options)
    children = np.random.SeedSequence(seed).spawn(n_permutations)
    null = np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(_permuted_score)(child, X, y, usable, strata, **options) for child in children
        ),
        dtype=np.float64,
    )
    finite = null[np.isfinite(null)]
    p_value = float((1 + np.count_nonzero(finite >= observed)) / (1 + finite.size))
    return PermutationResult(
        metric=chosen,
        observed=observed,
        null=null,
        p_value=p_value,
        n_undefined=int(null.size - finite.size),
    )
