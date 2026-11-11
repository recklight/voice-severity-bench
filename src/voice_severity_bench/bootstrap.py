"""Bootstrap intervals that resample speakers, not recordings.

Recordings of one speaker share a voice, so they are not independent draws;
resampling them one by one would make every interval too narrow. Each
resample here draws whole groups with replacement and weights every item by
how often its group was drawn, the cluster bootstrap of Davison & Hinkley
(1997, section 3.8). Items without a group are their own group. Intervals are
percentile intervals (Efron & Tibshirani, 1993).

Predictions may be a matrix with one column per repetition of the
cross-validation. The metric is then computed on each column and averaged,
which is the point estimate ``compare`` reports, so the interval belongs to
the same number. Averaging the predictions first would score an ensemble of
the repetitions instead, which is a better model than any one of them.

:func:`paired_bootstrap` uses the same resamples for both systems, so the
interval of the difference keeps the pairing of the items.

Classification metrics work the same way, with weighted counts. A resample can
lose a whole class when there are few speakers; its UAR, sensitivity,
specificity or ROC-AUC is then undefined, not a number computed on what is
left. Those resamples are left out of the interval and counted in
``n_undefined``: a large count means the interval rests on the resamples that
happened to keep every class and should not be read at face value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Metrics :func:`weighted_metric` computes, a subset of the regression metrics.
BOOTSTRAP_METRICS: tuple[str, ...] = ("rmse", "mae", "r2", "pearson_r", "ccc")
#: Classification metrics it computes; ``roc_auc`` takes the positive-class score as prediction.
CLASSIFICATION_BOOTSTRAP_METRICS: tuple[str, ...] = (
    "uar",
    "accuracy",
    "sensitivity",
    "specificity",
    "roc_auc",
)
#: Metrics that compare one class with the rest, so they need ``positive_class`` and two classes.
BINARY_BOOTSTRAP_METRICS: frozenset[str] = frozenset({"sensitivity", "specificity", "roc_auc"})


@dataclass(frozen=True, eq=False)
class BootstrapInterval:
    """Point estimate, percentile interval and the resampled values.

    Attributes
    ----------
    estimate : float
        The statistic on the original items.
    low, high : float
        Percentile interval at ``ci_level``.
    ci_level : float
        Coverage of ``low`` and ``high``.
    samples : numpy.ndarray, shape (n_resamples,)
        The statistic on every resample, for intervals at other levels.
    n_undefined : int
        Resamples whose statistic could not be computed (a class or all the
        spread gone); the interval comes from the others.
    """

    estimate: float
    low: float
    high: float
    ci_level: float
    samples: NDArray[np.float64]
    n_undefined: int = 0

    def interval(self, ci_level: float) -> tuple[float, float]:
        """Percentile interval at another level from the same resamples.

        Examples
        --------
        >>> result = BootstrapInterval(0.0, -1.0, 1.0, 0.95, np.linspace(-1.0, 1.0, 101))
        >>> result.interval(0.5)
        (-0.5, 0.5)
        """
        return _percentiles(self.samples, ci_level)

    def as_dict(self) -> dict[str, float]:
        """``estimate``, ``low``, ``high``, ``ci_level`` and ``n_undefined`` for tables and JSON."""
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "ci_level": self.ci_level,
            "n_undefined": self.n_undefined,
        }


def _percentiles(samples: NDArray[np.float64], ci_level: float) -> tuple[float, float]:
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie inside (0, 1), got {ci_level!r}")
    tail = (1.0 - ci_level) / 2.0
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return np.nan, np.nan
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high)


def group_codes(groups: ArrayLike | None, n_items: int) -> NDArray[np.intp]:
    """Integer speaker of every item; an empty label or ``None`` is a speaker of its own.

    Examples
    --------
    >>> group_codes(["b", "b", "a"], 3).tolist()
    [1, 1, 0]
    >>> group_codes(None, 2).tolist()
    [0, 1]
    >>> len(set(group_codes(["a", "", ""], 3).tolist()))
    3
    """
    if groups is None:
        return np.arange(n_items)
    labels = np.asarray(groups).astype(str)
    if labels.shape != (n_items,):
        raise ValueError(f"groups must have shape ({n_items},), got {labels.shape}")
    if not np.any(labels != ""):
        return np.arange(n_items)
    labels = np.where(labels == "", [f"\0item{i}" for i in range(n_items)], labels)
    return np.unique(labels, return_inverse=True)[1].ravel()


def group_weights(
    groups: ArrayLike | None, n_items: int, n_resamples: int, *, seed: int
) -> NDArray[np.float64]:
    """How often each item enters each resample when whole groups are drawn.

    Parameters
    ----------
    groups : array_like, shape (n_items,), optional
        Speaker of each item; empty strings or ``None`` make every item its own group.
    n_items : int
        Number of items.
    n_resamples : int
        Number of resamples.
    seed : int
        Seed of the draws.

    Returns
    -------
    numpy.ndarray, shape (n_resamples, n_items)
        Item weights; the items of one group share their weight.

    Examples
    --------
    >>> weights = group_weights(["a", "a", "b"], 3, 4, seed=0)
    >>> weights.shape, bool(np.all(weights[:, 0] == weights[:, 1]))
    ((4, 3), True)
    >>> bool(np.all(weights[:, [0, 2]].sum(axis=1) == 2))
    True
    """
    codes = group_codes(groups, n_items)
    n_groups = int(codes.max()) + 1 if codes.size else 0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    counts = np.stack([np.bincount(row, minlength=n_groups) for row in draws])
    return counts[:, codes].astype(np.float64)


def _references(y_true: ArrayLike, metric: str) -> NDArray:
    if metric in CLASSIFICATION_BOOTSTRAP_METRICS:
        return np.asarray(y_true).astype(str).ravel()
    return np.asarray(y_true, dtype=np.float64).ravel()


def _recall(
    truth: NDArray[np.str_], labels: NDArray[np.str_], W: NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    """Weighted recall of one class, shape (n_resamples, n_columns); nan when its weight is 0."""
    members = truth == name
    support = W[:, members].sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (W[:, members] @ (labels[members] == name)) / support


def _weighted_auc(
    positive: NDArray[np.bool_], scores: NDArray[np.float64], W: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weighted Mann-Whitney form of the ROC-AUC, ties counting one half (Hanley & McNeil, 1982).

    Every negative below a positive's score adds its weight, every tied
    negative half of it, so one sort per column serves all the resamples.
    """
    n_resamples = W.shape[0]
    value = np.empty((n_resamples, scores.shape[1]))
    denominator = W[:, positive].sum(axis=1) * W[:, ~positive].sum(axis=1)
    for column in range(scores.shape[1]):
        order = np.argsort(scores[:, column], kind="mergesort")
        ranked = scores[order, column]
        below = np.searchsorted(ranked, ranked, side="left")
        through = np.searchsorted(ranked, ranked, side="right")
        ordered = W[:, order]
        negatives = np.zeros((n_resamples, order.size + 1))
        negatives[:, 1:] = np.cumsum(ordered * ~positive[order], axis=1)
        credit = 0.5 * (negatives[:, below] + negatives[:, through])
        wins = (ordered * positive[order] * credit).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            value[:, column] = wins / denominator
    return value


def _weighted_classification(
    truth: NDArray[np.str_],
    y_pred: ArrayLike,
    W: NDArray[np.float64],
    metric: str,
    positive_class: str | None,
) -> NDArray[np.float64]:
    classes = np.unique(truth).tolist()
    if metric in BINARY_BOOTSTRAP_METRICS and (len(classes) != 2 or positive_class not in classes):
        raise ValueError(
            f"{metric} needs two classes and a positive_class among them; "
            f"got classes {classes} and positive_class {positive_class!r}"
        )
    predicted = np.asarray(y_pred)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if predicted.shape[0] != truth.size:
        raise ValueError(f"{predicted.shape[0]} predictions for {truth.size} references")
    if metric == "roc_auc":
        return _weighted_auc(truth == positive_class, predicted.astype(np.float64), W)
    labels = predicted.astype(str)
    if metric == "accuracy":
        return (W @ (labels == truth[:, None])) / W.sum(axis=1, keepdims=True)
    if metric == "sensitivity":
        return _recall(truth, labels, W, str(positive_class))
    if metric == "specificity":
        negative = next(name for name in classes if name != positive_class)
        return _recall(truth, labels, W, negative)
    # A class missing from a resample makes the sum nan, so the UAR is undefined there.
    return sum(_recall(truth, labels, W, name) for name in classes) / len(classes)


def weighted_metric(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    metric: str = "rmse",
    positive_class: str | None = None,
) -> float | NDArray[np.float64]:
    """A metric with item weights, averaged over prediction columns.

    Parameters
    ----------
    y_true : array_like, shape (n_items,)
        Reference values or class names.
    y_pred : array_like, shape (n_items,) or (n_items, n_repeats)
        Predictions; with several columns the metric of each is averaged.
        Class names for classification, except the positive-class score for ``roc_auc``.
    weights : array_like, shape (n_items,) or (n_resamples, n_items), optional
        Item weights (all ones when omitted); one row per resample.
    metric : str
        ``rmse``, ``mae``, ``r2``, ``pearson_r``, ``ccc``, or ``uar``,
        ``accuracy``, ``sensitivity``, ``specificity``, ``roc_auc``. With unit
        weights each equals its value in
        :func:`~voice_severity_bench.evaluation.regression_metrics` or
        :func:`~voice_severity_bench.evaluation.classification_metrics`.
    positive_class : str, optional
        Class whose recall is the sensitivity; ``sensitivity``, ``specificity``
        and ``roc_auc`` need it and exactly two classes.

    Returns
    -------
    float or numpy.ndarray
        A float for one weight vector, one value per row for a weight matrix.
        ``nan`` where the weights leave out a class the metric needs.

    Examples
    --------
    >>> truth = np.array([10.0, 20.0, 30.0, 40.0])
    >>> predicted = np.array([12.0, 18.0, 33.0, 41.0])
    >>> round(weighted_metric(truth, predicted, metric="rmse"), 3)
    2.121
    >>> weighted_metric(truth, predicted, [[1, 1, 1, 1], [0, 0, 2, 2]], metric="mae").tolist()
    [2.0, 2.0]
    >>> labels = ["sick", "sick", "well", "well"]
    >>> guess = ["sick", "well", "well", "well"]
    >>> weighted_metric(labels, guess, metric="uar")
    0.75
    >>> weighted_metric(labels, guess, [[1, 1, 1, 1], [2, 0, 2, 0]], metric="uar").tolist()
    [0.75, 1.0]
    >>> weighted_metric(labels, guess, [0, 0, 1, 1], metric="sensitivity", positive_class="sick")
    nan
    >>> weighted_metric(labels, [0.9, 0.4, 0.2, 0.3], metric="roc_auc", positive_class="sick")
    1.0
    """
    if metric not in BOOTSTRAP_METRICS and metric not in CLASSIFICATION_BOOTSTRAP_METRICS:
        known = [*BOOTSTRAP_METRICS, *CLASSIFICATION_BOOTSTRAP_METRICS]
        raise ValueError(f"unknown metric {metric!r}; choose one of {known}")
    truth = _references(y_true, metric)
    single = weights is None or np.ndim(weights) == 1
    W = np.ones((1, truth.size)) if weights is None else np.atleast_2d(np.asarray(weights, float))
    if W.shape[1] != truth.size:
        raise ValueError(f"weights cover {W.shape[1]} items, the references {truth.size}")
    if metric in CLASSIFICATION_BOOTSTRAP_METRICS:
        averaged = _weighted_classification(truth, y_pred, W, metric, positive_class).mean(axis=1)
        return float(averaged[0]) if single else averaged
    predicted = np.asarray(y_pred, dtype=np.float64)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if predicted.shape[0] != truth.size:
        raise ValueError(f"{predicted.shape[0]} predictions for {truth.size} references")
    total = W.sum(axis=1, keepdims=True)
    error = predicted - truth[:, None]
    if metric == "rmse":
        value = np.sqrt((W @ error**2) / total)
    elif metric == "mae":
        value = (W @ np.abs(error)) / total
    else:
        mean_true = (W @ truth)[:, None] / total
        var_true = (W @ truth**2)[:, None] / total - mean_true**2
        if metric == "r2":
            with np.errstate(divide="ignore", invalid="ignore"):
                value = 1.0 - ((W @ error**2) / total) / var_true
        else:
            mean_pred = (W @ predicted) / total
            var_pred = (W @ predicted**2) / total - mean_pred**2
            covariance = (W @ (truth[:, None] * predicted)) / total - mean_true * mean_pred
            with np.errstate(divide="ignore", invalid="ignore"):
                if metric == "pearson_r":
                    value = covariance / np.sqrt(var_true * var_pred)
                else:
                    value = 2.0 * covariance / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    averaged = value.mean(axis=1)
    return float(averaged[0]) if single else averaged


def bootstrap_metric(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike | None = None,
    *,
    metric: str = "rmse",
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
    positive_class: str | None = None,
) -> BootstrapInterval:
    """A metric and its group-bootstrap percentile interval.

    Parameters
    ----------
    y_true, y_pred, metric, positive_class
        As in :func:`weighted_metric`.
    groups : array_like, shape (n_items,), optional
        Speakers; resampled whole.
    n_resamples : int, optional
        Number of resamples.
    ci_level : float, optional
        Coverage of the interval.
    seed : int, optional
        Seed of the resamples.

    Returns
    -------
    BootstrapInterval

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> truth = rng.uniform(0, 100, 60)
    >>> result = bootstrap_metric(truth, truth + rng.normal(0, 5, 60), n_resamples=500)
    >>> bool(result.low < result.estimate < result.high), bool(3.0 < result.estimate < 7.0)
    (True, True)
    """
    truth = _references(y_true, metric)
    weights = group_weights(groups, truth.size, n_resamples, seed=seed)
    samples = weighted_metric(truth, y_pred, weights, metric=metric, positive_class=positive_class)
    low, high = _percentiles(samples, ci_level)
    return BootstrapInterval(
        estimate=weighted_metric(truth, y_pred, metric=metric, positive_class=positive_class),
        low=low,
        high=high,
        ci_level=ci_level,
        samples=samples,
        n_undefined=int(np.count_nonzero(~np.isfinite(samples))),
    )


def paired_bootstrap(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    groups: ArrayLike | None = None,
    *,
    metric: str = "rmse",
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
    positive_class: str | None = None,
) -> BootstrapInterval:
    """Difference ``metric(a) - metric(b)`` on the same items, with a paired interval.

    Both systems are scored on every resample with the same item weights, so
    the interval reflects how much the *difference* varies between samples of
    speakers, not how much each score does.

    Parameters
    ----------
    y_true : array_like, shape (n_items,)
        Reference values.
    pred_a, pred_b : array_like, shape (n_items,) or (n_items, n_repeats)
        Predictions of the two systems; the numbers of columns may differ
        (e.g. a model's repetitions against a single human rating).
    groups, metric, n_resamples, ci_level, seed, positive_class
        As in :func:`bootstrap_metric`.

    Returns
    -------
    BootstrapInterval
        Of the difference; negative values favour ``a`` for RMSE and MAE,
        positive values for every other metric.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> truth = rng.uniform(0, 100, 80)
    >>> good, poor = truth + rng.normal(0, 4, 80), truth + rng.normal(0, 8, 80)
    >>> result = paired_bootstrap(truth, good, poor, n_resamples=500)
    >>> bool(result.high < 0.0)
    True
    """
    truth = _references(y_true, metric)
    weights = group_weights(groups, truth.size, n_resamples, seed=seed)
    options = {"metric": metric, "positive_class": positive_class}
    samples = weighted_metric(truth, pred_a, weights, **options) - weighted_metric(
        truth, pred_b, weights, **options
    )
    low, high = _percentiles(samples, ci_level)
    estimate = weighted_metric(truth, pred_a, **options) - weighted_metric(truth, pred_b, **options)
    return BootstrapInterval(
        estimate=float(estimate),
        low=low,
        high=high,
        ci_level=ci_level,
        samples=samples,
        n_undefined=int(np.count_nonzero(~np.isfinite(samples))),
    )
