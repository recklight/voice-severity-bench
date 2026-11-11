"""Ablation study: feature settings and models compared on shared folds.

The ``ablation`` section lists values for the number of cepstra, filters,
dynamic-feature order (13 / 26 / 39 dimensions), energy term, framing,
lifter, pooling statistics, dimensionality reduction and estimator, and every
combination of them is evaluated by repeated k-fold cross-validation on
**the same folds**: splits depend only on the targets, groups and seed, so
the per-fold scores of two settings are paired.

Static cepstra are computed once per distinct ``audio`` + ``frontend``
setting (the delta settings aside), extended with the requested dynamics and
reused for every pooling and model choice; with a cache folder they are also
kept on disk between runs.

Every setting is compared with the baseline (the first value of every grid
list) on the paired fold differences :math:`d_j` of the ranking metric:

* the corrected resampled t test of Nadeau & Bengio (2003), whose variance
  :math:`(1/J + n_{test}/n_{train})\\, s_d^2` accounts for the overlap between
  training sets that makes the naive paired t test far too optimistic;
* the Wilcoxon signed-rank test (Demšar, 2006) as a distribution-free check;

and both families of p-values are adjusted with Holm's method (Holm, 1979)
by default.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .association import adjust_pvalues
from .cepstral_frontend import append_dynamics
from .config import Config
from .dataset import index_intervals, load_dataset_index
from .errors import ConfigError
from .evaluation import (
    LOWER_IS_BETTER,
    class_scores,
    classification_metrics,
    default_metric,
    metric_names,
    regression_metrics,
)
from .models import fit_model
from .pipeline import pool_recordings, recording_frames
from .pooling import pooled_feature_names
from .splitting import has_groups, repeated_split_indices

_LOGGER = logging.getLogger(__name__)

#: Grid dimensions: (column label, ablation field, dotted configuration keys).
_DIMENSIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("n_ceps", "n_ceps", ("frontend.n_ceps",)),
    ("n_mels", "n_mels", ("frontend.n_mels",)),
    ("delta_order", "delta_order", ("frontend.delta_order",)),
    ("energy_term", "energy_term", ("frontend.energy_term",)),
    ("frame_hop_ms", "frame_hop_ms", ("frontend.frame_ms", "frontend.hop_ms")),
    ("lifter", "lifter", ("frontend.lifter",)),
    ("statistics", "statistics", ("pooling.statistics",)),
    ("reducer", "reducer", ("model.reducer",)),
    ("model", "models", ("model.name",)),
)


def _label(value: Any) -> str:
    if isinstance(value, tuple) and value and all(isinstance(v, str) for v in value):
        return "+".join(value)
    if isinstance(value, tuple):
        return "/".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
    return str(value)


@dataclass(frozen=True)
class AblationPoint:
    """One combination of the grid.

    Attributes
    ----------
    config_id : str
        Readable identifier built from the dimensions that vary, e.g.
        ``n_ceps=12|delta_order=2|model=ridge`` (``baseline`` when nothing varies).
    settings : tuple of (str, str)
        Value of every grid dimension, as labels.
    config : Config
        The complete configuration of this combination.

    Examples
    --------
    >>> point = expand_grid(Config().override({"ablation.n_ceps": [6], "ablation.models": []}))[0]
    >>> point.config_id, dict(point.settings)["n_ceps"], point.n_features
    ('delta_order=0', '6', 14)
    """

    config_id: str
    settings: tuple[tuple[str, str], ...]
    config: Config

    @property
    def n_features(self) -> int:
        """Length of the pooled feature vector."""
        return len(pooled_feature_names(self.config.frontend, self.config.pooling))


def expand_grid(config: Config) -> list[AblationPoint]:
    """Every combination of the ``ablation`` grid, the baseline first.

    Parameters
    ----------
    config : Config
        Base configuration; empty grid lists keep its values.

    Returns
    -------
    list of AblationPoint
        In row-major order of the grid dimensions.

    Raises
    ------
    ConfigError
        If the grid does not fit the other sections (see
        :meth:`~voice_severity_bench.config.Config.ablation_problems`) or a
        combination is invalid.

    Examples
    --------
    >>> config = Config().override({"ablation.n_ceps": [12], "ablation.models": ["ridge"]})
    >>> [point.config_id for point in expand_grid(config)]
    ['delta_order=0', 'delta_order=1', 'delta_order=2']
    >>> [point.n_features for point in expand_grid(config)]
    [26, 52, 78]
    """
    problems = config.ablation_problems()
    if problems:
        raise ConfigError(" | ".join(problems))
    grid = config.ablation
    axes: list[tuple[str, tuple[str, ...], list[Any]]] = []
    for label, field_name, keys in _DIMENSIONS:
        values = list(getattr(grid, field_name))
        if values:
            axes.append((label, keys, values))
    points = []
    for combination in itertools.product(*(values for _, _, values in axes)):
        updates: dict[str, Any] = {}
        settings = []
        varying = []
        for (label, keys, values), value in zip(axes, combination, strict=True):
            if len(keys) == 2:
                updates[keys[0]], updates[keys[1]] = value
            else:
                updates[keys[0]] = list(value) if isinstance(value, tuple) else value
            settings.append((label, _label(value)))
            if len(values) > 1:
                varying.append(f"{label}={_label(value)}")
        point_config = config.override(updates) if updates else config
        config_id = "|".join(varying) if varying else "baseline"
        points.append(AblationPoint(config_id, tuple(settings), point_config))
    return points


def corrected_resampled_ttest(
    differences: ArrayLike, test_train_ratio: float
) -> tuple[float, float]:
    """Nadeau-Bengio corrected resampled t test of paired cross-validation scores.

    Parameters
    ----------
    differences : array_like, shape (J,)
        Paired score differences over all repeats and folds.
    test_train_ratio : float
        :math:`n_{test} / n_{train}` of one fold.

    Returns
    -------
    tuple of float
        ``(t, p)`` with :math:`J - 1` degrees of freedom, two-sided.

    References
    ----------
    .. [1] C. Nadeau and Y. Bengio, "Inference for the generalization error,"
       Machine Learning, vol. 52, no. 3, pp. 239-281, 2003.

    Examples
    --------
    The correction widens the naive interval, so the p-value grows:

    >>> d = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02])
    >>> t, p = corrected_resampled_ttest(d, 0.25)
    >>> naive = stats.ttest_1samp(d, 0.0).pvalue
    >>> round(t, 3), bool(p > naive)
    (4.098, True)
    """
    values = np.asarray(differences, dtype=np.float64).ravel()
    n = values.size
    if n < 2:
        return math.nan, math.nan
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    if variance == 0.0:
        return (0.0, 1.0) if mean == 0.0 else (math.copysign(math.inf, mean), 0.0)
    statistic = mean / math.sqrt((1.0 / n + test_train_ratio) * variance)
    return statistic, float(2.0 * stats.t.sf(abs(statistic), n - 1))


def wilcoxon_signed_rank(differences: ArrayLike) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when every difference is zero.

    References
    ----------
    .. [1] J. Demšar, "Statistical comparisons of classifiers over multiple
       data sets," Journal of Machine Learning Research, vol. 7, pp. 1-30, 2006.

    Examples
    --------
    >>> wilcoxon_signed_rank(np.zeros(6)), round(wilcoxon_signed_rank(np.arange(1.0, 9.0)), 4)
    (1.0, 0.0078)
    """
    values = np.asarray(differences, dtype=np.float64).ravel()
    if values.size == 0 or np.all(values == 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(values).pvalue)
    except ValueError:
        return math.nan


@dataclass(frozen=True, eq=False)
class AblationResult:
    """Scores of every grid point and their comparison with the baseline.

    Attributes
    ----------
    folds : pandas.DataFrame
        Tidy table, one row per setting, repeat and fold (``ablation_folds.csv``).
    summary : pandas.DataFrame
        One row per setting with mean and standard deviation of every metric
        and its rank (``ablation_summary.csv``).
    comparisons : pandas.DataFrame
        One row per non-baseline setting (``ablation_comparison.csv``).
    metric : str
        Ranking metric.
    baseline : str
        ``config_id`` of the baseline.

    Examples
    --------
    >>> summary = pd.DataFrame({"config_id": ["a", "b"], "rmse_mean": [10.25, 12.125]})
    >>> AblationResult(pd.DataFrame(), summary, pd.DataFrame(), "rmse", "a").as_dict()
    {'metric': 'rmse', 'baseline': 'a', 'n_settings': 2, 'best': 'a', 'best_mean': 10.25}
    """

    folds: pd.DataFrame
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    metric: str
    baseline: str

    def as_dict(self) -> dict[str, Any]:
        """Headline numbers for JSON output."""
        best = self.summary.iloc[0]
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "n_settings": len(self.summary),
            "best": str(best["config_id"]),
            "best_mean": float(best[f"{self.metric}_mean"]),
        }


def _static_key(
    config: Config, paths: Sequence[str], intervals: NDArray[np.float64] | None = None
) -> str:
    """Fingerprint of everything the static cepstra depend on (the delta settings excluded)."""
    files = [(path, Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths]
    tree = config.to_dict()
    frontend = {
        key: value
        for key, value in tree["frontend"].items()
        if key not in ("delta_order", "delta_widths")
    }
    payload: dict[str, Any] = {"audio": tree["audio"], "frontend": frontend, "files": files}
    if intervals is not None:
        payload["intervals"] = [
            [None if np.isnan(value) else float(value) for value in row] for row in intervals
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cached_frames(
    paths: Sequence[str],
    config: Config,
    cache_dir: Path | None,
    key: str,
    intervals: NDArray[np.float64] | None = None,
) -> list[NDArray[np.float64]]:
    target = None if cache_dir is None else cache_dir / f"frames_{key[:16]}.npz"
    if target is not None and target.is_file():
        with np.load(target, allow_pickle=False) as archive:
            if str(archive["key"]) == key:
                stacked, offsets = archive["frames"], archive["offsets"]
                _LOGGER.debug("frames reused from %s", target)
                return [stacked[a:b] for a, b in itertools.pairwise(offsets)]
    frames = recording_frames(paths, config, static_only=True, intervals=intervals)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.concatenate([[0], np.cumsum([matrix.shape[0] for matrix in frames])])
        np.savez_compressed(target, frames=np.vstack(frames), offsets=offsets, key=np.asarray(key))
    return frames


def summarize_ablation(folds: pd.DataFrame, metric: str, metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and standard deviation (``ddof=1``) of every metric per setting, best first.

    Parameters
    ----------
    folds : pandas.DataFrame
        Tidy fold table (``ablation_folds.csv``).
    metric : str
        Ranking metric.
    metrics : sequence of str
        Metrics to summarize.

    Returns
    -------
    pandas.DataFrame
        One row per ``config_id`` with ``<metric>_mean``, ``<metric>_std`` and ``rank``.

    Examples
    --------
    >>> folds = pd.DataFrame(
    ...     {
    ...         "config_id": ["a"] * 4 + ["b"] * 4,
    ...         "n_features": [26] * 4 + [78] * 4,
    ...         "rmse": [10.0, 11.0, 9.5, 10.5, 12.0, 12.5, 11.0, 13.0],
    ...     }
    ... )
    >>> table = summarize_ablation(folds, "rmse", ["rmse"])
    >>> table[["config_id", "rmse_mean", "rank"]].values.tolist()
    [['a', 10.25, 1], ['b', 12.125, 2]]
    """
    keys = ["config_id"]
    settings = [label for label, _, _ in _DIMENSIONS if label in folds.columns]
    grouped = folds.groupby("config_id", sort=False)
    table = grouped[[*settings, "n_features"]].first().reset_index()
    for name in metrics:
        table[f"{name}_mean"] = grouped[name].mean().to_numpy()
        table[f"{name}_std"] = grouped[name].std(ddof=1).to_numpy()
    ascending = metric in LOWER_IS_BETTER
    table["rank"] = table[f"{metric}_mean"].rank(ascending=ascending, method="min").astype(int)
    return table.sort_values(["rank", *keys], kind="mergesort").reset_index(drop=True)


def compare_to_baseline(
    folds: pd.DataFrame,
    *,
    metric: str,
    baseline: str,
    correction: str = "holm",
) -> pd.DataFrame:
    """Paired tests of every setting against the baseline on the same folds.

    Parameters
    ----------
    folds : pandas.DataFrame
        Tidy fold table with ``config_id``, ``repeat``, ``fold``, ``n_train``,
        ``n_test`` and the metric column.
    metric : str
        Compared metric.
    baseline : str
        ``config_id`` of the reference setting.
    correction : {"holm", "fdr_bh"}, optional
        Adjustment applied to each family of p-values.

    Returns
    -------
    pandas.DataFrame
        Columns ``config_id``, ``mean_difference`` (setting minus baseline),
        ``better`` (the difference points the right way for the metric),
        ``n_pairs``, ``t_statistic``, ``p_corrected_t``,
        ``p_corrected_t_adjusted``, ``p_wilcoxon`` and ``p_wilcoxon_adjusted``.

    Examples
    --------
    >>> folds = pd.DataFrame(
    ...     {
    ...         "config_id": ["a"] * 4 + ["b"] * 4,
    ...         "repeat": [0, 0, 1, 1] * 2,
    ...         "fold": [0, 1, 0, 1] * 2,
    ...         "n_train": [30] * 8,
    ...         "n_test": [10] * 8,
    ...         "rmse": [10.0, 11.0, 9.5, 10.5, 12.0, 12.5, 11.0, 13.0],
    ...     }
    ... )
    >>> table = compare_to_baseline(folds, metric="rmse", baseline="a")
    >>> table[["config_id", "mean_difference", "better", "n_pairs"]].values.tolist()
    [['b', 1.875, False, 4]]
    """
    pivot = folds.pivot_table(index=["repeat", "fold"], columns="config_id", values=metric)
    if baseline not in pivot.columns:
        raise ValueError(f"baseline {baseline!r} is not among the settings")
    ratio = float((folds["n_test"] / folds["n_train"]).mean())
    order = list(dict.fromkeys(folds["config_id"]))
    rows = []
    for config_id in order:
        if config_id == baseline:
            continue
        difference = (pivot[config_id] - pivot[baseline]).dropna().to_numpy()
        t_statistic, p_t = corrected_resampled_ttest(difference, ratio)
        mean = float(difference.mean()) if difference.size else math.nan
        better = mean < 0.0 if metric in LOWER_IS_BETTER else mean > 0.0
        rows.append(
            {
                "config_id": config_id,
                "mean_difference": mean,
                "better": bool(better),
                "n_pairs": int(difference.size),
                "t_statistic": t_statistic,
                "p_corrected_t": p_t,
                "p_wilcoxon": wilcoxon_signed_rank(difference),
            }
        )
    columns = [
        "config_id",
        "mean_difference",
        "better",
        "n_pairs",
        "t_statistic",
        "p_corrected_t",
        "p_corrected_t_adjusted",
        "p_wilcoxon",
        "p_wilcoxon_adjusted",
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=columns)
    table["p_corrected_t_adjusted"] = adjust_pvalues(table["p_corrected_t"].to_numpy(), correction)
    table["p_wilcoxon_adjusted"] = adjust_pvalues(table["p_wilcoxon"].to_numpy(), correction)
    return table[columns]


def run_ablation(
    dataset: str | Path, config: Config, *, cache_dir: str | Path | None = None
) -> AblationResult:
    """Evaluate every grid point by repeated cross-validation on shared folds.

    Parameters
    ----------
    dataset : str or pathlib.Path
        ``dataset.csv`` of the recordings.
    config : Config
        Base configuration with the ``ablation`` grid; ``evaluation.n_splits``
        folds are repeated ``ablation.n_repeats`` times.
    cache_dir : str or pathlib.Path, optional
        Folder for frame-level feature caches reused across runs.

    Returns
    -------
    AblationResult
        Fold scores, the ranked summary and the baseline comparisons.

    Examples
    --------
    >>> import tempfile
    >>> from voice_severity_bench.synthetic import write_synthetic_dataset
    >>> config = Config().override(
    ...     {
    ...         "synthetic.n_speakers": 4,
    ...         "synthetic.n_recordings_per_speaker": 2,
    ...         "synthetic.duration_s": 0.3,
    ...         "ablation.n_ceps": [12],
    ...         "ablation.delta_order": [0, 1],
    ...         "ablation.models": ["ridge"],
    ...         "ablation.n_repeats": 1,
    ...         "evaluation.n_splits": 2,
    ...     }
    ... )
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     index = write_synthetic_dataset(folder, config.synthetic, rng=0)
    ...     result = run_ablation(index, config)
    >>> sorted(result.summary["config_id"]), len(result.folds), result.baseline
    (['delta_order=0', 'delta_order=1'], 4, 'delta_order=0')
    """
    points = expand_grid(config)
    task = config.data.task
    index = load_dataset_index(dataset, target=config.data.target, task=task)
    paths = index["path"].tolist()
    intervals = index_intervals(index)
    y = index[config.data.target].to_numpy()
    y = y.astype(np.float64) if task == "regression" else y.astype(str)
    groups = index["group"].to_numpy().astype(str)
    grouped = has_groups(groups)
    classes = sorted(np.unique(y).tolist()) if task == "classification" else []
    binary = len(classes) == 2
    positive = config.evaluation.positive_class if binary else None
    names = metric_names(task, binary=binary)
    metric = config.ablation.metric or default_metric(task)
    if metric not in names:
        raise ValueError(
            f"ablation.metric={metric!r} is not available here; choose from {list(names)}"
        )
    splits = repeated_split_indices(
        y,
        groups,
        task=task,
        n_splits=config.evaluation.n_splits,
        n_repeats=config.ablation.n_repeats,
        seed=config.seed,
    )
    cache = None if cache_dir is None else Path(cache_dir)
    frame_store: dict[str, list[NDArray[np.float64]]] = {}
    rows: list[dict[str, Any]] = []
    _LOGGER.info(
        "%d settings x %d repeats x %d folds on %d recordings",
        len(points),
        len(splits),
        config.evaluation.n_splits,
        len(paths),
    )
    for number, point in enumerate(points, start=1):
        key = _static_key(point.config, paths, intervals)
        if key not in frame_store:
            frame_store[key] = _cached_frames(paths, point.config, cache, key, intervals)
        frontend = point.config.frontend
        frames = [
            append_dynamics(static, frontend.delta_order, frontend.delta_widths)
            for static in frame_store[key]
        ]
        X = pool_recordings(frames, point.config)
        for repeat, folds in enumerate(splits):
            for fold, (train, test) in enumerate(folds):
                model = fit_model(
                    X[train], y[train], point.config, groups=groups[train] if grouped else None
                )
                if task == "regression":
                    scores = regression_metrics(y[test], model.predict(X[test]))
                else:
                    scores = classification_metrics(
                        y[test],
                        np.asarray(model.predict(X[test])).astype(str),
                        class_scores(model, X[test], classes),
                        classes=classes,
                        positive_class=positive,
                    )
                rows.append(
                    {"config_id": point.config_id, **dict(point.settings)}
                    | {
                        "n_features": X.shape[1],
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": int(train.size),
                        "n_test": int(test.size),
                    }
                    | {name: scores[name] for name in names}
                )
        _LOGGER.info("setting %d/%d done: %s", number, len(points), point.config_id)
    folds_table = pd.DataFrame(rows)
    summary = summarize_ablation(folds_table, metric, names)
    comparisons = compare_to_baseline(
        folds_table,
        metric=metric,
        baseline=points[0].config_id,
        correction=config.analysis.correction,
    )
    return AblationResult(
        folds=folds_table,
        summary=summary,
        comparisons=comparisons,
        metric=metric,
        baseline=points[0].config_id,
    )
