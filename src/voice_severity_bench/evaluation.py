"""Cross-validated evaluation and the metrics of both tasks.

Every item is predicted exactly once by a model that never saw it (or, with
a ``group`` column, never saw its speaker). Folds come from
:func:`~voice_severity_bench.splitting.split_indices`; the model, including
its scaler, reducer and any hyperparameter search, is fitted on the training
fold only.

Regression metrics: mean absolute error, root mean squared error,
:math:`R^2`, Pearson's :math:`r`, Spearman's :math:`\\rho` and Lin's
concordance correlation coefficient (Lin, 1989). CCC is there because a
predicted severity has to *agree* with the clinical rating, not merely
correlate with it. Classification metrics: unweighted average recall (UAR,
Schuller et al., 2009) as the headline number for unbalanced classes,
accuracy and macro F1, plus sensitivity, specificity and ROC-AUC for two
classes. Each metric is reported per fold, as mean and sample standard
deviation (``ddof=1``) over folds, and pooled over all out-of-fold predictions.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import speechdsp
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from .association import bland_altman, concordance_ccc
from .errors import ConfigError
from .models import fit_model
from .splitting import has_groups, split_indices

if TYPE_CHECKING:
    from .config import Config
    from .dataset import FeatureSet

_LOGGER = logging.getLogger(__name__)

#: Scalar metrics of regression, in report order.
REGRESSION_METRICS: tuple[str, ...] = ("mae", "rmse", "r2", "pearson_r", "spearman_rho", "ccc")
#: Scalar metrics of every classification, in report order.
CLASSIFICATION_METRICS: tuple[str, ...] = ("uar", "accuracy", "f1_macro")
#: Extra scalar metrics of two-class problems.
BINARY_METRICS: tuple[str, ...] = ("sensitivity", "specificity", "roc_auc")
#: Metrics for which smaller is better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"mae", "rmse"})


def metric_names(task: str, *, binary: bool = False) -> tuple[str, ...]:
    """Scalar metric names of a task, in report order.

    Examples
    --------
    >>> metric_names("classification", binary=True)
    ('uar', 'accuracy', 'f1_macro', 'sensitivity', 'specificity', 'roc_auc')
    """
    if task == "regression":
        return REGRESSION_METRICS
    return CLASSIFICATION_METRICS + (BINARY_METRICS if binary else ())


def default_metric(task: str) -> str:
    """The headline metric: ``rmse`` for regression, ``uar`` for classification."""
    return "rmse" if task == "regression" else "uar"


def _correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if first.size < 2 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """MAE, RMSE, R², Pearson r, Spearman rho and Lin's CCC of one set of predictions.

    Correlations are ``nan`` when either side is constant (or has fewer than
    two values); R² and CCC need at least two and three pairs.

    Examples
    --------
    >>> scores = regression_metrics([10.0, 20.0, 30.0, 40.0], [12.0, 18.0, 33.0, 41.0])
    >>> [round(scores[key], 3) for key in ("mae", "rmse", "r2", "pearson_r", "spearman_rho", "ccc")]
    [2.0, 2.121, 0.964, 0.987, 1.0, 0.983]
    """
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    predicted = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.size != predicted.size or truth.size == 0:
        raise ValueError(
            f"need equally long, non-empty vectors, got {truth.size} and {predicted.size}"
        )
    many = truth.size >= 2 and np.ptp(truth) > 0.0
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)) if many else math.nan,
        "pearson_r": _correlation(truth, predicted),
        "spearman_rho": _correlation(stats.rankdata(truth), stats.rankdata(predicted)),
        "ccc": concordance_ccc(truth, predicted) if truth.size >= 3 else math.nan,
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    scores: ArrayLike | None = None,
    *,
    classes: Sequence[str],
    positive_class: str | None = None,
) -> dict[str, float]:
    """UAR, accuracy and macro F1, plus sensitivity, specificity and ROC-AUC for two classes.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_items,)
        Class names.
    scores : array_like, shape (n_items, n_classes), optional
        Class probabilities in the order of ``classes`` (needed for ROC-AUC).
    classes : sequence of str
        All class names, sorted.
    positive_class : str, optional
        The class whose recall is the sensitivity (two-class problems).

    Returns
    -------
    dict
        Keys of :func:`metric_names`. In a set with a single class only the
        recall of that class is defined: ``specificity`` is ``nan`` when the
        set has no negative item, ``sensitivity`` when it has no positive
        one. ``roc_auc`` is ``nan`` without scores or with a single class.

    Examples
    --------
    >>> truth = ["dysphonic", "dysphonic", "healthy", "healthy"]
    >>> guess = ["dysphonic", "healthy", "healthy", "healthy"]
    >>> probs = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]]
    >>> metrics = classification_metrics(
    ...     truth, guess, probs, classes=["dysphonic", "healthy"], positive_class="dysphonic"
    ... )
    >>> list(metrics)
    ['uar', 'accuracy', 'f1_macro', 'sensitivity', 'specificity', 'roc_auc']
    >>> [round(value, 3) for value in metrics.values()]
    [0.75, 0.75, 0.733, 0.5, 1.0, 1.0]
    """
    truth = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    labels = [str(name) for name in classes]
    present = np.unique(truth)
    if present.size < len(labels):
        _LOGGER.warning(
            "only %d of %d classes occur in this set; its UAR averages the classes that occur "
            "and its macro F1 the classes that are true or predicted here",
            present.size,
            len(labels),
        )
    with warnings.catch_warnings():
        # A set with one label makes scikit-learn's confusion_matrix warn; the log above covers it.
        warnings.simplefilter("ignore", UserWarning)
        # speechdsp.uar averages the recalls of the classes in the truth (balanced accuracy).
        uar = float(speechdsp.uar(truth, predicted))
    result = {
        "uar": uar,
        "accuracy": float(accuracy_score(truth, predicted)),
        # Unlike the UAR, macro F1 also counts a class that is only ever predicted.
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }
    if len(labels) == 2 and positive_class is not None:
        if positive_class not in labels:
            raise ConfigError(
                f"evaluation.positive_class={positive_class!r} is not one of the classes {labels}"
            )
        sensitivity = specificity = math.nan
        if present.size == 2:
            sensitivity, specificity = speechdsp.sensitivity_specificity(
                truth, predicted, pos_label=positive_class
            )
        elif present.size == 1:
            # Only this class's recall is defined; speechdsp would repeat the warning above.
            recall = float(np.mean(predicted == present[0]))
            if present[0] == positive_class:
                sensitivity = recall
            else:
                specificity = recall
        auc = math.nan
        if scores is not None and present.size == 2:
            probabilities = np.asarray(scores, dtype=np.float64)
            auc = float(
                roc_auc_score(
                    truth == positive_class, probabilities[:, labels.index(positive_class)]
                )
            )
        result.update(sensitivity=float(sensitivity), specificity=float(specificity), roc_auc=auc)
    return result


def class_scores(estimator: Any, X: ArrayLike, classes: Sequence[str]) -> NDArray[np.float64]:
    """Class probabilities aligned to ``classes`` (zero for a class the model never saw)."""
    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    fitted = [str(name) for name in estimator.classes_]
    aligned = np.zeros((probabilities.shape[0], len(classes)))
    for column, name in enumerate(classes):
        if name in fitted:
            aligned[:, column] = probabilities[:, fitted.index(name)]
    return aligned


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample standard deviation (``ddof=1``) of the finite values.

    Examples
    --------
    >>> summarize([0.7, 0.8, 0.9, float("nan")])
    {'mean': 0.8, 'std': 0.1}
    """
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan}
    mean = round(float(finite.mean()), 12)
    std = round(float(finite.std(ddof=1)), 12) if finite.size > 1 else math.nan
    return {"mean": mean, "std": std}


def per_class_table(
    y_true: ArrayLike, y_pred: ArrayLike, classes: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Recall, precision, F1 and support of every class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(y_true).astype(str),
        np.asarray(y_pred).astype(str),
        labels=list(classes),
        zero_division=0,
    )
    return {
        name: {
            "recall": float(recall[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(classes)
    }


@dataclass(frozen=True, eq=False)
class CrossValidationResult:
    """Out-of-fold predictions, per-fold metrics and their summary.

    Attributes
    ----------
    predictions : pandas.DataFrame
        One row per item in input order (``predictions.csv``).
    folds : pandas.DataFrame
        One row per fold (``folds.csv``).
    metrics : dict
        Contents of ``metrics.json``.
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """The ``metrics.json`` mapping."""
        return self.metrics


def cross_validate(features: FeatureSet, config: Config) -> CrossValidationResult:
    """K-fold cross-validation of the configured model.

    Parameters
    ----------
    features : FeatureSet
        Utterance-level features from ``prepare``.
    config : Config
        Run configuration (``data.task``, ``model``, ``evaluation``, ``seed``).

    Returns
    -------
    CrossValidationResult
        Predictions, fold metrics and the ``metrics.json`` mapping.

    Raises
    ------
    DatasetError
        If the data cannot be split into ``evaluation.n_splits`` folds.
    """
    task = config.data.task
    X = np.asarray(features.X, dtype=np.float64)
    y = np.asarray(features.y)
    groups = np.asarray(features.groups).astype(str)
    grouped = has_groups(groups)
    n_items = X.shape[0]
    folds = split_indices(
        y,
        groups if grouped else None,
        task=task,
        n_splits=config.evaluation.n_splits,
        random_state=config.seed,
    )
    target = config.data.target
    fold_of = np.empty(n_items, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    if task == "regression":
        truth = y.astype(np.float64)
        predicted = np.empty(n_items, dtype=np.float64)
        for k, (train, test) in enumerate(folds):
            model = fit_model(
                X[train], truth[train], config, groups=groups[train] if grouped else None
            )
            predicted[test] = model.predict(X[test])
            fold_of[test] = k
            rows.append(
                {"fold": k, "n_train": train.size, "n_test": test.size}
                | regression_metrics(truth[test], predicted[test])
            )
            _LOGGER.info("fold %d/%d done", k + 1, len(folds))
        names = REGRESSION_METRICS
        pooled: dict[str, Any] = regression_metrics(truth, predicted)
        pooled["bland_altman"] = bland_altman(predicted, truth).as_dict()
        predictions = pd.DataFrame(
            {
                "path": features.paths,
                "group": groups,
                "fold": fold_of,
                target: truth,
                "predicted": predicted,
            }
        )
    else:
        truth = y.astype(str)
        classes = sorted(np.unique(truth).tolist())
        binary = len(classes) == 2
        positive = config.evaluation.positive_class if binary else None
        if binary and positive not in classes:
            raise ConfigError(
                f"evaluation.positive_class={positive!r} is not one of the classes {classes}"
            )
        predicted_labels = np.empty(n_items, dtype=object)
        scores = np.zeros((n_items, len(classes)))
        for k, (train, test) in enumerate(folds):
            model = fit_model(
                X[train], truth[train], config, groups=groups[train] if grouped else None
            )
            predicted_labels[test] = np.asarray(model.predict(X[test])).astype(str)
            scores[test] = class_scores(model, X[test], classes)
            fold_of[test] = k
            rows.append(
                {"fold": k, "n_train": train.size, "n_test": test.size}
                | classification_metrics(
                    truth[test],
                    predicted_labels[test].astype(str),
                    scores[test],
                    classes=classes,
                    positive_class=positive,
                )
            )
            _LOGGER.info("fold %d/%d done", k + 1, len(folds))
        names = metric_names(task, binary=binary)
        labels_out = predicted_labels.astype(str)
        pooled = classification_metrics(
            truth, labels_out, scores, classes=classes, positive_class=positive
        )
        pooled["classes"] = classes
        pooled["confusion_matrix"] = confusion_matrix(truth, labels_out, labels=classes).tolist()
        pooled["per_class"] = per_class_table(truth, labels_out, classes)
        predictions = pd.DataFrame(
            {
                "path": features.paths,
                "group": groups,
                "fold": fold_of,
                target: truth,
                "predicted": labels_out,
            }
        )
        for column, name in enumerate(classes):
            predictions[f"score_{name}"] = scores[:, column]
    fold_table = pd.DataFrame(rows, columns=["fold", "n_train", "n_test", *names])
    metrics = {
        "task": task,
        "n_items": int(n_items),
        "n_splits": len(folds),
        "seed": int(config.seed),
        "folds": fold_table.to_dict(orient="records"),
        "summary": {name: summarize(fold_table[name].tolist()) for name in names},
        "pooled": pooled,
    }
    return CrossValidationResult(predictions=predictions, folds=fold_table, metrics=metrics)
