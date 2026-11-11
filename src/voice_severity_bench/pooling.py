"""Utterance-level statistics of frame-level features.

The models need one vector per recording, so the frames of a recording are
summarized by statistics computed column by column: mean, standard deviation,
extremes, percentiles and the third and fourth standardized moments. Which
statistics are used is itself a setting that the ablation study can vary.

Optional per-recording normalization comes first: cepstral mean
normalization (CMN) removes the channel offset, mean and variance
normalization (CMVN) also equalizes the spread. Combinations that make a
statistic constant across recordings (the mean after CMN, the mean or the
standard deviation after CMVN) are rejected, because they would only feed the
model columns of zeros and ones.

The pooled vector is laid out statistic-major: all frame columns for the
first statistic, then all for the second, with names such as ``mean_c1`` and
``p90_d_c0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import speechdsp
from numpy.typing import ArrayLike, NDArray

from .cepstral_frontend import feature_layout

if TYPE_CHECKING:
    from .config import FrontendConfig, PoolingConfig

#: Statistics accepted by ``pooling.statistics``.
POOLING_STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "percentile",
    "skew",
    "kurtosis",
)
#: Per-recording normalizations accepted by ``pooling.normalization``.
FRAME_NORMALIZATIONS: tuple[str, ...] = ("none", "cmn", "cmvn")

#: Statistics that a normalization makes identical for every recording.
_CONSTANT_AFTER: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "cmn": frozenset({"mean"}),
    "cmvn": frozenset({"mean", "std"}),
}


def constant_statistics(normalization: str, statistics: Sequence[str]) -> list[str]:
    """Statistics that would be the same for every recording after ``normalization``.

    Parameters
    ----------
    normalization : {"none", "cmn", "cmvn"}
        Per-recording normalization.
    statistics : sequence of str
        Requested statistics.

    Returns
    -------
    list of str
        The offending statistics, empty when the combination is informative.

    Examples
    --------
    >>> constant_statistics("cmvn", ["mean", "std", "skew"])
    ['mean', 'std']
    """
    blocked = _CONSTANT_AFTER.get(normalization, frozenset())
    return [name for name in statistics if name in blocked]


def _percentile_label(value: float) -> str:
    return f"p{value:g}"


def statistic_labels(
    statistics: Sequence[str], percentiles: Sequence[float] = (10.0, 50.0, 90.0)
) -> tuple[str, ...]:
    """Labels of the pooled blocks, one per statistic (one per level for percentiles).

    Examples
    --------
    >>> statistic_labels(["mean", "percentile"], [25.0, 75.0])
    ('mean', 'p25', 'p75')
    """
    labels: list[str] = []
    for name in statistics:
        if name == "percentile":
            labels.extend(_percentile_label(q) for q in percentiles)
        else:
            labels.append(name)
    return tuple(labels)


def pool_frames(
    frames: ArrayLike,
    statistics: Sequence[str] = ("mean", "std"),
    *,
    percentiles: Sequence[float] = (10.0, 50.0, 90.0),
    normalization: str = "none",
) -> NDArray[np.float64]:
    """Summarize a frame-level matrix into one utterance vector.

    Parameters
    ----------
    frames : array_like, shape (n_frames, n_dims)
        Frame-level features of one recording.
    statistics : sequence of str, optional
        Any of :data:`POOLING_STATISTICS`, in output order.
    percentiles : sequence of float, optional
        Levels in percent for the ``"percentile"`` statistic.
    normalization : {"none", "cmn", "cmvn"}, optional
        Per-recording normalization applied before pooling.

    Returns
    -------
    numpy.ndarray, shape (n_blocks * n_dims,)
        Statistic-major vector; ``n_blocks`` is the length of
        :func:`statistic_labels`. Standard deviations use ``ddof=0``; skewness
        and excess kurtosis are the biased moment ratios of
        :func:`scipy.stats.skew` and :func:`scipy.stats.kurtosis`, and are 0
        for a constant column.

    Raises
    ------
    ValueError
        For an empty matrix, an unknown statistic or normalization, or a
        combination listed by :func:`constant_statistics`.

    Examples
    --------
    >>> frames = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0]])
    >>> pool_frames(frames, ["mean", "max"]).tolist()
    [2.0, 30.0, 3.0, 60.0]
    """
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"need a non-empty (frames x dims) matrix, got shape {values.shape}")
    unknown = sorted(set(statistics) - set(POOLING_STATISTICS))
    if unknown:
        raise ValueError(
            f"unknown pooling statistic(s) {unknown}; choose from {list(POOLING_STATISTICS)}"
        )
    if normalization not in FRAME_NORMALIZATIONS:
        raise ValueError(
            f"pooling.normalization={normalization!r} is not supported; "
            f"choose one of {list(FRAME_NORMALIZATIONS)}"
        )
    constant = constant_statistics(normalization, statistics)
    if constant:
        raise ValueError(f"{constant} would be constant after {normalization}")
    if normalization == "cmn":
        values = speechdsp.cmn(values)
    elif normalization == "cmvn":
        values = speechdsp.cmvn(values)

    centered = values - values.mean(axis=0)
    variance = np.mean(centered**2, axis=0)
    safe = np.where(variance > 0.0, variance, 1.0)
    blocks: list[NDArray[np.float64]] = []
    for name in statistics:
        if name == "mean":
            blocks.append(values.mean(axis=0))
        elif name == "std":
            blocks.append(np.sqrt(variance))
        elif name == "min":
            blocks.append(values.min(axis=0))
        elif name == "max":
            blocks.append(values.max(axis=0))
        elif name == "percentile":
            blocks.append(np.percentile(values, list(percentiles), axis=0).ravel())
        elif name == "skew":
            skew = np.mean(centered**3, axis=0) / safe**1.5
            blocks.append(np.where(variance > 0.0, skew, 0.0))
        else:  # kurtosis
            kurtosis = np.mean(centered**4, axis=0) / safe**2 - 3.0
            blocks.append(np.where(variance > 0.0, kurtosis, 0.0))
    return np.concatenate(blocks)


def pooled_feature_names(frontend: FrontendConfig, pooling: PoolingConfig) -> tuple[str, ...]:
    """Column names of the pooled vector for one front end and pooling setting.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig, PoolingConfig
    >>> names = pooled_feature_names(FrontendConfig(n_ceps=1, delta_order=0), PoolingConfig())
    >>> names
    ('mean_c1', 'mean_c0', 'std_c1', 'std_c0')
    """
    frame_names = feature_layout(frontend)
    return tuple(
        f"{label}_{name}"
        for label in statistic_labels(pooling.statistics, pooling.percentiles)
        for name in frame_names
    )
