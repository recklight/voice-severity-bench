"""Feature families compared on the same folds, with paired tests and bootstrap intervals.

:func:`compare_feature_sets` evaluates the configured model on
several feature files of the same items, for example MFCCs from ``prepare``,
clinical voice measures and the layers of a speech representation network.
Every file sees the same repeated folds, so fold scores pair across files and
every item gets one out-of-fold prediction per file and repetition.
Imputation, scaling, the hyperparameters and, for blocked files, the block
are all chosen inside each training fold. Classification works the same way
with class names as targets: the folds are stratified and grouped, each
repetition keeps the predicted class and the score of every class, and the
headline metric is the UAR.

:func:`pairwise_contrasts` then tests chosen pairs of files, as one family:

* the corrected resampled t test (Nadeau & Bengio, 2003) and the Wilcoxon
  signed-rank test on the paired fold scores, both adjusted over the family
  (Holm by default), as in the ablation study;
* the group bootstrap of the difference in the pooled metric
  (:func:`~voice_severity_bench.bootstrap.paired_bootstrap`);
* with an equivalence margin, whether the interval at
  ``equivalence_ci_level`` lies inside it. A 90 % interval inside
  ``(-margin, margin)`` is the interval form of two one-sided tests at 5 %
  (Schuirmann, 1987).

The point estimate of a metric is computed on all out-of-fold predictions of
one repetition and averaged over the repetitions. The fold scores feed only
the t and Wilcoxon tests.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from .ablation import corrected_resampled_ttest, wilcoxon_signed_rank
from .association import adjust_pvalues
from .bootstrap import (
    BINARY_BOOTSTRAP_METRICS,
    BOOTSTRAP_METRICS,
    CLASSIFICATION_BOOTSTRAP_METRICS,
    bootstrap_metric,
    paired_bootstrap,
)
from .config import Config
from .dataset import FeatureSet
from .errors import ConfigError, DatasetError
from .evaluation import (
    LOWER_IS_BETTER,
    REGRESSION_METRICS,
    class_scores,
    classification_metrics,
    default_metric,
    metric_names,
    regression_metrics,
)
from .models import Blocks, fit_model, search_space
from .splitting import has_groups, repeated_split_indices

_LOGGER = logging.getLogger(__name__)

#: Items an error message names before "... and N more".
_LISTED = 5

BlockCandidates = Sequence[str | tuple[str, ...]]


def _key_label(key: tuple[str, float | None, float | None]) -> str:
    path, start, end = key
    if start is None and end is None:
        return path
    return f"{path} [{'' if start is None else f'{start:g}'}-{'' if end is None else f'{end:g}'} s]"


def _listing(keys: Sequence[tuple[str, float | None, float | None]]) -> str:
    shown = ", ".join(_key_label(key) for key in keys[:_LISTED])
    extra = len(keys) - _LISTED
    return shown + (f" ... and {extra} more" if extra > 0 else "")


def _same_targets(first: NDArray[Any], second: NDArray[Any]) -> bool:
    try:
        return bool(np.allclose(first.astype(np.float64), second.astype(np.float64)))
    except ValueError:
        return bool(np.array_equal(first.astype(str), second.astype(str)))


def align_feature_sets(feature_sets: Mapping[str, FeatureSet]) -> dict[str, FeatureSet]:
    """Put every feature file in the row order of the first one.

    Rows are matched by ``(path, start_s, end_s)``
    (:meth:`~voice_severity_bench.dataset.FeatureSet.keys`).

    Parameters
    ----------
    feature_sets : mapping of str to FeatureSet
        Named feature files of the same items.

    Returns
    -------
    dict of str to FeatureSet
        The same names, in the same order, every file reordered like the first.

    Raises
    ------
    DatasetError
        If a file repeats an item, the files do not hold the same items, or
        their targets or groups disagree.

    Examples
    --------
    >>> from voice_severity_bench.dataset import feature_set_from_table
    >>> a = feature_set_from_table(
    ...     pd.DataFrame({"path": ["x", "y"], "s": [1.0, 2.0], "f": [0.1, 0.2]}), target="s"
    ... )
    >>> b = feature_set_from_table(
    ...     pd.DataFrame({"path": ["y", "x"], "s": [2.0, 1.0], "g": [5.0, 4.0]}), target="s"
    ... )
    >>> align_feature_sets({"a": a, "b": b})["b"].X.ravel().tolist()
    [4.0, 5.0]
    """
    names = list(feature_sets)
    if not names:
        raise ValueError("no feature files given")
    reference = feature_sets[names[0]]
    keys = reference.keys()
    key_set = set(keys)
    aligned = {names[0]: reference}
    for name in names:
        other_keys = feature_sets[name].keys()
        position = {key: row for row, key in enumerate(other_keys)}
        if len(position) != len(other_keys):
            repeated = [key for key, count in Counter(other_keys).items() if count > 1]
            raise DatasetError(f"feature file {name!r} lists items twice: {_listing(repeated)}")
        if name == names[0]:
            continue
        missing = [key for key in keys if key not in position]
        extra = [key for key in other_keys if key not in key_set]
        if missing or extra:
            raise DatasetError(
                f"feature files {names[0]!r} and {name!r} do not hold the same items: "
                f"{len(missing)} only in {names[0]!r} ({_listing(missing)}), "
                f"{len(extra)} only in {name!r} ({_listing(extra)})"
            )
        taken = feature_sets[name].take([position[key] for key in keys])
        if not _same_targets(np.asarray(taken.y), np.asarray(reference.y)):
            raise DatasetError(f"feature files {names[0]!r} and {name!r} hold different targets")
        if not np.array_equal(np.asarray(taken.groups, str), np.asarray(reference.groups, str)):
            raise DatasetError(f"feature files {names[0]!r} and {name!r} hold different groups")
        aligned[name] = taken
    return aligned


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """Fold scores, out-of-fold predictions and the summary of a comparison.

    Attributes
    ----------
    folds : pandas.DataFrame
        One row per feature file, repetition and fold: ``n_train``, ``n_test``,
        ``n_features`` (columns the fitted model used), ``params`` (the values
        the inner loop chose, as JSON) and every regression metric.
    predictions : pandas.DataFrame
        One row per feature file, repetition and item: ``fold``, ``item`` and
        ``y_pred``. Classification adds ``score_<class>`` for every class.
    items : pandas.DataFrame
        ``item``, ``path``, ``start_s``, ``end_s``, ``group`` and ``y_true``.
    summary : pandas.DataFrame
        One row per feature file, best first: every metric pooled per
        repetition and averaged, the bootstrap interval of ``metric``
        (``<metric>_low``, ``<metric>_high``) and its fold mean and standard deviation.
    metric : str
        Ranking metric.
    names : tuple of str
        Feature files in the order given; the first is the default baseline.
    n_repeats : int
        Repetitions of the k-fold split.
    task : str
        ``regression`` or ``classification``.
    classes : tuple of str
        Class names, sorted; empty for regression.
    positive_class : str or None
        The class ``sensitivity`` and ``roc_auc`` refer to; two classes only.
    """

    folds: pd.DataFrame
    predictions: pd.DataFrame
    items: pd.DataFrame
    summary: pd.DataFrame
    metric: str
    names: tuple[str, ...]
    n_repeats: int
    task: str = "regression"
    classes: tuple[str, ...] = ()
    positive_class: str | None = None

    @property
    def y_true(self) -> NDArray[Any]:
        """Targets in item order: floats, or class names for classification."""
        if self.task == "classification":
            return self.items["y_true"].to_numpy().astype(str)
        return self.items["y_true"].to_numpy(dtype=np.float64)

    @property
    def groups(self) -> NDArray[np.str_]:
        """Groups in item order."""
        return self.items["group"].to_numpy().astype(str)

    def _pivot(self, name: str, column: str) -> pd.DataFrame:
        if name not in self.names:
            raise KeyError(f"unknown feature file {name!r}; available: {list(self.names)}")
        part = self.predictions[self.predictions["feature_set"] == name]
        return part.pivot(index="item", columns="repeat", values=column).sort_index()

    def prediction_matrix(self, name: str) -> NDArray[Any]:
        """``(n_items, n_repeats)`` out-of-fold predictions of one feature file.

        Floats for regression, class names for classification.
        """
        table = self._pivot(name, "y_pred")
        if self.task == "classification":
            return table.to_numpy().astype(str)
        return table.to_numpy(dtype=np.float64)

    def score_matrix(self, name: str, class_name: str | None = None) -> NDArray[np.float64]:
        """``(n_items, n_repeats)`` out-of-fold scores of one class, the positive class by default.

        Raises
        ------
        ConfigError
            For regression, or when no class is named and none is positive.
        """
        chosen = self.positive_class if class_name is None else class_name
        if self.task != "classification" or chosen is None:
            raise ConfigError("class scores need classification and a class to read them for")
        if chosen not in self.classes:
            raise ConfigError(f"unknown class {chosen!r}; available: {list(self.classes)}")
        return self._pivot(name, f"score_{chosen}").to_numpy(dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        """Headline numbers for JSON output."""
        best = self.summary.iloc[0]
        headline = {
            "metric": self.metric,
            "n_items": len(self.items),
            "n_repeats": self.n_repeats,
            "feature_sets": list(self.names),
            "best": str(best["feature_set"]),
            "best_value": float(best[self.metric]),
        }
        if self.task == "classification":
            headline |= {
                "task": self.task,
                "classes": list(self.classes),
                "positive_class": self.positive_class,
            }
        return headline


def _chosen(model: Any, keys: Sequence[str]) -> dict[str, Any]:
    params = model.get_params()
    chosen = {}
    for key in keys:
        value = params[key]
        chosen[key] = "+".join(value) if isinstance(value, tuple) else value
    return chosen


def _fit_fold(
    X: NDArray[np.float64],
    y: NDArray[Any],
    groups: NDArray[np.str_] | None,
    train: NDArray[np.intp],
    test: NDArray[np.intp],
    config: Config,
    blocks: Blocks | None,
    candidates: BlockCandidates | None,
    classes: Sequence[str] = (),
) -> tuple[NDArray[Any], NDArray[np.float64] | None, dict[str, Any], int]:
    model = fit_model(
        X[train],
        y[train],
        config,
        groups=None if groups is None else groups[train],
        blocks=blocks,
        block_candidates=candidates,
    )
    keys = list(search_space(config, blocks=blocks, block_candidates=candidates))
    selector = model.named_steps.get("select")
    used = X.shape[1] if selector is None else int(selector.columns_.size)
    if config.data.task == "classification":
        predicted = np.asarray(model.predict(X[test])).astype(str)
        return predicted, class_scores(model, X[test], classes), _chosen(model, keys), used
    return np.asarray(model.predict(X[test]), dtype=np.float64), None, _chosen(model, keys), used


def _check_metric(metric: str, task: str, n_classes: int = 2) -> None:
    known = BOOTSTRAP_METRICS if task == "regression" else CLASSIFICATION_BOOTSTRAP_METRICS
    if metric not in known:
        raise ConfigError(f"metric {metric!r} is not available; choose one of {known}")
    if metric in BINARY_BOOTSTRAP_METRICS and n_classes != 2:
        raise ConfigError(f"metric {metric!r} needs two classes, the data have {n_classes}")


def _class_names(y: NDArray[np.str_], config: Config) -> tuple[tuple[str, ...], str | None]:
    classes = tuple(sorted(np.unique(y).tolist()))
    if len(classes) != 2:
        return classes, None
    positive = config.evaluation.positive_class
    if positive not in classes:
        raise ConfigError(
            f"evaluation.positive_class={positive!r} is not one of the classes {list(classes)}"
        )
    return classes, positive


def compare_feature_sets(
    feature_sets: Mapping[str, FeatureSet],
    config: Config,
    *,
    metric: str | None = None,
    block_candidates: Mapping[str, BlockCandidates] | None = None,
    strata: ArrayLike | None = None,
    n_jobs: int = 1,
) -> ComparisonResult:
    """Evaluate the configured model on every feature file with the same repeated folds.

    Parameters
    ----------
    feature_sets : mapping of str to FeatureSet
        Named feature files of the same items (aligned with
        :func:`align_feature_sets`); targets and groups come from the first.
    config : Config
        ``data.task``, ``model``, ``evaluation.n_splits``, ``evaluation.n_repeats``,
        ``evaluation.positive_class``, ``analysis.n_resamples``,
        ``analysis.ci_level`` and ``seed``.
    metric : str, optional
        Ranking metric. Regression: ``rmse`` (default), ``mae``, ``r2``,
        ``pearson_r``, ``ccc``. Classification: ``uar`` (default), ``accuracy``,
        and for two classes ``sensitivity``, ``specificity``, ``roc_auc``.
    block_candidates : mapping, optional
        For blocked files, the blocks (or tuples of blocks kept together) the
        inner loop chooses from; every single block when a file is not listed.
    strata : array_like, shape (n_items,), optional
        Classification only: a stratum per item, in the row order of the first
        file; the folds are stratified on label x stratum.
    n_jobs : int, optional
        Folds fitted in parallel (joblib). The chosen settings do not depend on it, but
        each worker gets ``cpu_count // n_jobs`` BLAS threads, which can move the last bits
        of the numbers; ``joblib.parallel_config(inner_max_num_threads=1)`` pins them.

    Returns
    -------
    ComparisonResult

    Raises
    ------
    ConfigError
        For a metric that does not fit the task, a positive class that is not
        among the classes, or strata with regression.
    DatasetError
        If the files do not describe the same items.

    Examples
    --------
    >>> from voice_severity_bench.dataset import feature_set_from_table
    >>> rng = np.random.default_rng(0)
    >>> table = pd.DataFrame({"path": [f"r{i}.wav" for i in range(40)],
    ...                       "severity": rng.uniform(0, 100, 40)})
    >>> table["signal"] = table["severity"] / 10 + rng.normal(0, 1, 40)
    >>> table["noise"] = rng.normal(0, 1, 40)
    >>> sets = {
    ...     name: feature_set_from_table(table, target="severity", feature_columns=[name])
    ...     for name in ("noise", "signal")
    ... }
    >>> config = Config().override({"evaluation.n_repeats": 2, "analysis.n_resamples": 200})
    >>> result = compare_feature_sets(sets, config)
    >>> result.summary["feature_set"].tolist(), result.prediction_matrix("signal").shape
    (['signal', 'noise'], (40, 2))
    """
    task = config.data.task
    metric = default_metric(task) if metric is None else metric
    _check_metric(metric, task)
    if strata is not None and task != "classification":
        raise ConfigError("strata stratify classification folds; regression has none")
    aligned = align_feature_sets(feature_sets)
    names = tuple(aligned)
    first = aligned[names[0]]
    classes: tuple[str, ...] = ()
    positive: str | None = None
    if task == "classification":
        y = np.asarray(first.y).astype(str)
        classes, positive = _class_names(y, config)
        _check_metric(metric, task, len(classes))
    else:
        y = np.asarray(first.y, dtype=np.float64)
    groups = np.asarray(first.groups).astype(str)
    usable = groups if has_groups(groups) else None
    candidates = dict(block_candidates or {})
    unknown = sorted(set(candidates) - set(names))
    if unknown:
        raise ConfigError(f"block candidates given for unknown feature file(s) {unknown}")
    splits = repeated_split_indices(
        y,
        usable,
        task=task,
        n_splits=config.evaluation.n_splits,
        n_repeats=config.evaluation.n_repeats,
        seed=config.seed,
        strata=strata,
    )
    jobs = [
        (name, repeat, fold, train, test)
        for name in names
        for repeat, folds in enumerate(splits)
        for fold, (train, test) in enumerate(folds)
    ]
    _LOGGER.info(
        "%d feature files x %d repeats x %d folds on %d items",
        len(names),
        len(splits),
        config.evaluation.n_splits,
        y.size,
    )
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(
            aligned[name].X,
            y,
            usable,
            train,
            test,
            config,
            aligned[name].blocks,
            candidates.get(name),
            classes,
        )
        for name, _, _, train, test in jobs
    )
    fold_rows, prediction_parts = [], []
    for (name, repeat, fold, train, test), (predicted, scores, chosen, used) in zip(
        jobs, outputs, strict=True
    ):
        fold_scores = (
            regression_metrics(y[test], predicted)
            if task == "regression"
            else classification_metrics(
                y[test], predicted, scores, classes=classes, positive_class=positive
            )
        )
        fold_rows.append(
            {
                "feature_set": name,
                "repeat": repeat,
                "fold": fold,
                "n_train": int(train.size),
                "n_test": int(test.size),
                "n_features": used,
                "params": json.dumps(chosen, sort_keys=True),
            }
            | fold_scores
        )
        part = pd.DataFrame(
            {
                "feature_set": name,
                "repeat": repeat,
                "fold": fold,
                "item": test,
                "y_pred": predicted,
            }
        )
        for column, class_name in enumerate(classes):
            part[f"score_{class_name}"] = scores[:, column]
        prediction_parts.append(part)
    folds_table = pd.DataFrame(fold_rows)
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["feature_set", "repeat", "item"], kind="mergesort")
        .reset_index(drop=True)
    )
    intervals = (
        np.full((y.size, 2), np.nan) if first.intervals is None else np.asarray(first.intervals)
    )
    items = pd.DataFrame(
        {
            "item": np.arange(y.size),
            "path": np.asarray(first.paths).astype(str),
            "start_s": intervals[:, 0],
            "end_s": intervals[:, 1],
            "group": groups,
            "y_true": y,
        }
    )
    result = ComparisonResult(
        folds=folds_table,
        predictions=predictions,
        items=items,
        summary=pd.DataFrame(),
        metric=metric,
        names=names,
        n_repeats=len(splits),
        task=task,
        classes=classes,
        positive_class=positive,
    )
    return replace(result, summary=_summary(result, aligned, config))


def _bootstrap_input(result: ComparisonResult, name: str, metric: str) -> NDArray[Any]:
    """What the bootstrap scores: the positive-class score for ROC-AUC, else the predictions."""
    return result.score_matrix(name) if metric == "roc_auc" else result.prediction_matrix(name)


def _per_repeat_scores(result: ComparisonResult, name: str) -> list[dict[str, float]]:
    """Every metric pooled over the out-of-fold predictions of each repetition."""
    matrix = result.prediction_matrix(name)
    if result.task == "regression":
        return [regression_metrics(result.y_true, column) for column in matrix.T]
    scores = np.stack([result.score_matrix(name, c) for c in result.classes], axis=-1)
    return [
        classification_metrics(
            result.y_true,
            matrix[:, repeat],
            scores[:, repeat, :],
            classes=result.classes,
            positive_class=result.positive_class,
        )
        for repeat in range(matrix.shape[1])
    ]


def _summary(
    result: ComparisonResult, aligned: Mapping[str, FeatureSet], config: Config
) -> pd.DataFrame:
    metric = result.metric
    classification = result.task == "classification"
    keys = (
        metric_names(result.task, binary=len(result.classes) == 2)
        if classification
        else REGRESSION_METRICS
    )
    rows = []
    for name in result.names:
        per_repeat = _per_repeat_scores(result, name)
        interval = bootstrap_metric(
            result.y_true,
            _bootstrap_input(result, name, metric),
            result.groups,
            metric=metric,
            n_resamples=config.analysis.n_resamples,
            ci_level=config.analysis.ci_level,
            seed=config.seed,
            positive_class=result.positive_class,
        )
        part = result.folds[result.folds["feature_set"] == name]
        row: dict[str, Any] = {
            "feature_set": name,
            "kind": aligned[name].kind,
            "n_columns": int(np.shape(aligned[name].X)[1]),
            "n_features_used": float(part["n_features"].mean()),
        }
        for key in keys:
            row[key] = float(np.mean([scores[key] for scores in per_repeat]))
        row[f"{metric}_low"], row[f"{metric}_high"] = interval.low, interval.high
        if classification:
            row[f"{metric}_n_undefined"] = interval.n_undefined
        row[f"{metric}_fold_mean"] = float(part[metric].mean())
        row[f"{metric}_fold_std"] = float(part[metric].std(ddof=1))
        rows.append(row)
    table = pd.DataFrame(rows)
    ascending = metric in LOWER_IS_BETTER
    table["rank"] = table[metric].rank(ascending=ascending, method="min").astype(int)
    return table.sort_values(["rank"], kind="mergesort").reset_index(drop=True)


def repeat_averaged_predictions(result: ComparisonResult) -> pd.DataFrame:
    """The item table with each feature file's prediction averaged over the repetitions.

    Meant for residual plots and residual analyses; the metrics of the
    comparison are not computed from these averages (see the module notes).
    For two classes the average is of the positive-class score; class names
    have no average, so more than two classes raise a :class:`ConfigError`.
    """
    table = result.items.copy()
    for name in result.names:
        matrix = result.score_matrix(name) if result.task == "classification" else None
        table[name] = (result.prediction_matrix(name) if matrix is None else matrix).mean(axis=1)
    return table


def pairwise_contrasts(
    result: ComparisonResult,
    contrasts: Sequence[tuple[str, str]] | None = None,
    *,
    metric: str | None = None,
    correction: str = "holm",
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    equivalence_margin: float | None = None,
    equivalence_ci_level: float = 0.90,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired tests and bootstrap intervals for chosen pairs of feature files.

    Parameters
    ----------
    result : ComparisonResult
        Output of :func:`compare_feature_sets`.
    contrasts : sequence of (str, str), optional
        ``(first, second)`` pairs; the difference is first minus second.
        Default: every file against the first one.
    metric : str, optional
        Compared metric; ``result.metric`` when omitted.
    correction : {"holm", "fdr_bh"}, optional
        Adjustment over the family of contrasts.
    n_resamples, ci_level, seed : optional
        Group bootstrap settings; the same resamples serve every contrast.
    equivalence_margin : float, optional
        Margin in metric units; adds ``equivalence_low``, ``equivalence_high``
        and ``equivalent`` (that interval lies inside ``(-margin, margin)``).
    equivalence_ci_level : float, optional
        Coverage of the equivalence interval (0.90 for tests at 5 %).

    Returns
    -------
    pandas.DataFrame
        One row per contrast: ``first``, ``second``, ``difference`` with
        ``ci_low``/``ci_high``, ``better`` (``first`` is better), the fold-level
        ``fold_mean_difference``, ``n_pairs``, ``t_statistic``,
        ``p_corrected_t`` and ``p_wilcoxon`` with their adjusted values.
        Classification adds ``n_undefined``, the resamples that lost a class.

    Examples
    --------
    >>> from voice_severity_bench.dataset import feature_set_from_table
    >>> rng = np.random.default_rng(1)
    >>> table = pd.DataFrame({"path": [f"r{i}" for i in range(60)], "y": rng.uniform(0, 100, 60)})
    >>> table["good"] = table["y"] / 10 + rng.normal(0, 1, 60)
    >>> table["poor"] = table["y"] / 10 + rng.normal(0, 4, 60)
    >>> sets = {c: feature_set_from_table(table, target="y", feature_columns=[c])
    ...         for c in ("poor", "good")}
    >>> config = Config().override({"evaluation.n_repeats": 3, "analysis.n_resamples": 200})
    >>> contrasts = pairwise_contrasts(compare_feature_sets(sets, config), n_resamples=500)
    >>> contrasts[["first", "second", "better"]].values.tolist()
    [['good', 'poor', True]]
    >>> bool(contrasts["ci_high"].iloc[0] < 0)
    True
    """
    chosen = metric or result.metric
    _check_metric(chosen, result.task, len(result.classes) or 2)
    pairs = (
        [(name, result.names[0]) for name in result.names[1:]]
        if contrasts is None
        else [tuple(pair) for pair in contrasts]
    )
    unknown = sorted({name for pair in pairs for name in pair} - set(result.names))
    if unknown:
        raise ConfigError(f"contrasts name unknown feature file(s) {unknown}")
    pivot = result.folds.pivot_table(index=["repeat", "fold"], columns="feature_set", values=chosen)
    ratio = float((result.folds["n_test"] / result.folds["n_train"]).mean())
    rows = []
    for first, second in pairs:
        differences = (pivot[first] - pivot[second]).dropna().to_numpy()
        t_statistic, p_t = corrected_resampled_ttest(differences, ratio)
        interval = paired_bootstrap(
            result.y_true,
            _bootstrap_input(result, first, chosen),
            _bootstrap_input(result, second, chosen),
            result.groups,
            metric=chosen,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
            positive_class=result.positive_class,
        )
        row: dict[str, Any] = {
            "first": first,
            "second": second,
            "difference": interval.estimate,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "better": bool(
                interval.estimate < 0.0 if chosen in LOWER_IS_BETTER else interval.estimate > 0.0
            ),
            "fold_mean_difference": float(differences.mean()) if differences.size else math.nan,
            "n_pairs": int(differences.size),
            "t_statistic": t_statistic,
            "p_corrected_t": p_t,
            "p_wilcoxon": wilcoxon_signed_rank(differences),
        }
        if result.task == "classification":
            row["n_undefined"] = interval.n_undefined
        if equivalence_margin is not None:
            low, high = interval.interval(equivalence_ci_level)
            row["equivalence_low"], row["equivalence_high"] = low, high
            row["equivalent"] = bool(-equivalence_margin < low and high < equivalence_margin)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_corrected_t_adjusted"] = adjust_pvalues(table["p_corrected_t"].to_numpy(), correction)
    table["p_wilcoxon_adjusted"] = adjust_pvalues(table["p_wilcoxon"].to_numpy(), correction)
    order = [
        "first",
        "second",
        "difference",
        "ci_low",
        "ci_high",
        "better",
        "fold_mean_difference",
        "n_pairs",
        "t_statistic",
        "p_corrected_t",
        "p_corrected_t_adjusted",
        "p_wilcoxon",
        "p_wilcoxon_adjusted",
    ]
    extra = [column for column in table.columns if column not in order]
    return table[order + extra]
