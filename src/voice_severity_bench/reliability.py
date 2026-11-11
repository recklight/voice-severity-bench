"""Agreement between raters, and how close a model can get to the mean rating.

Ratings come as an items x ratings matrix with NaN where an item has fewer
ratings than the widest row; clinical corpora often have three raters for
most recordings and four for some.

* :func:`icc_oneway`: the one-way random-effects ICC(1,1) and ICC(1,k)
  (Shrout & Fleiss, 1979) with F-based intervals. One-way because the
  columns are usually rating positions: the second rater of one recording
  need not be the second rater of the next, so there is no rater effect to
  estimate. With unequal numbers of ratings k is replaced by k0 (McGraw &
  Wong, 1996).
* :func:`rating_noise_floor`: the RMSE a perfect predictor of the true score
  would still have against the mean of k noisy ratings,
  :math:`\\sqrt{\\overline{s_i^2 / k_i}}`. No model can be expected to go below it.
* :func:`leave_one_rater_out`: every rating against the mean of the *other*
  ratings of the same item, which is how far one clinician is from the panel.
  A model compared on the same references (the ``reference`` column) is
  compared with a single rater on equal terms.

The correlation a model can reach with the mean rating is bounded by
:math:`\\sqrt{ICC(1,k)}`, the square root of the reliability of that mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def _matrix(ratings: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(ratings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"ratings must be an items x ratings matrix, got shape {matrix.shape}")
    return matrix


@dataclass(frozen=True)
class ICCResult:
    """One-way random-effects intraclass correlations.

    Attributes
    ----------
    icc1, icc1_low, icc1_high : float
        ICC(1,1), the reliability of a single rating, and its interval.
    icck, icck_low, icck_high : float
        ICC(1,k), the reliability of the mean of an item's ratings, and its interval.
    k0 : float
        Effective number of ratings per item (k itself when balanced).
    n_items : int
        Items with at least two ratings, the ones the estimate uses.
    f, df_between, df_within, p : float
        F test of the item effect.
    ci_level : float
        Coverage of the intervals.
    """

    icc1: float
    icc1_low: float
    icc1_high: float
    icck: float
    icck_low: float
    icck_high: float
    k0: float
    n_items: int
    f: float
    df_between: float
    df_within: float
    p: float
    ci_level: float


def icc_oneway(ratings: ArrayLike, ci_level: float = 0.95) -> ICCResult:
    """ICC(1,1) and ICC(1,k) of an items x ratings matrix, NaN for absent ratings.

    Parameters
    ----------
    ratings : array_like, shape (n_items, max_ratings)
        Ratings; items with fewer than two are left out.
    ci_level : float, optional
        Coverage of the F-based intervals.

    Returns
    -------
    ICCResult

    References
    ----------
    .. [1] P. E. Shrout and J. L. Fleiss, "Intraclass correlations: uses in
       assessing rater reliability," Psychological Bulletin, vol. 86, no. 2,
       pp. 420-428, 1979.
    .. [2] K. O. McGraw and S. P. Wong, "Forming inferences about some
       intraclass correlation coefficients," Psychological Methods, vol. 1,
       no. 1, pp. 30-46, 1996.

    Examples
    --------
    The worked example of Shrout & Fleiss (1979), six targets and four judges:

    >>> from voice_severity_bench.reference_data import shrout_fleiss_ratings
    >>> result = icc_oneway(shrout_fleiss_ratings())
    >>> round(result.icc1, 3), round(result.icck, 3)
    (0.166, 0.443)
    """
    rows = [row[~np.isnan(row)] for row in _matrix(ratings)]
    rows = [row for row in rows if row.size >= 2]
    n = len(rows)
    if n < 2:
        raise ValueError("need at least two items with two or more ratings")
    sizes = np.array([row.size for row in rows], dtype=np.float64)
    total = sizes.sum()
    grand = float(np.concatenate(rows).mean())
    means = np.array([row.mean() for row in rows])
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(sum(np.sum((row - row.mean()) ** 2) for row in rows))
    df_between, df_within = n - 1.0, total - n
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    k0 = float((total - np.sum(sizes**2) / total) / (n - 1))
    f_observed = ms_between / ms_within
    tail = (1.0 - ci_level) / 2.0
    f_low = f_observed / stats.f.ppf(1.0 - tail, df_between, df_within)
    f_high = f_observed * stats.f.ppf(1.0 - tail, df_within, df_between)
    return ICCResult(
        icc1=float((ms_between - ms_within) / (ms_between + (k0 - 1.0) * ms_within)),
        icc1_low=float((f_low - 1.0) / (f_low + k0 - 1.0)),
        icc1_high=float((f_high - 1.0) / (f_high + k0 - 1.0)),
        icck=float((ms_between - ms_within) / ms_between),
        icck_low=float(1.0 - 1.0 / f_low),
        icck_high=float(1.0 - 1.0 / f_high),
        k0=k0,
        n_items=n,
        f=float(f_observed),
        df_between=df_between,
        df_within=df_within,
        p=float(stats.f.sf(f_observed, df_between, df_within)),
        ci_level=ci_level,
    )


def rating_noise_floor(ratings: ArrayLike) -> float:
    """Expected RMSE of the true score against the mean of each item's ratings.

    Each item contributes the variance of its own mean, :math:`s_i^2 / k_i`
    (sample variance, ``ddof=1``); items with a single rating have none to give
    and are left out.

    Examples
    --------
    >>> round(rating_noise_floor([[10.0, 14.0], [50.0, 50.0], [80.0, np.nan]]), 4)
    1.4142
    """
    matrix = _matrix(ratings)
    counts = np.sum(~np.isnan(matrix), axis=1)
    usable = counts >= 2
    if not np.any(usable):
        raise ValueError("no item has two or more ratings")
    variances = np.nanvar(matrix[usable], axis=1, ddof=1)
    return float(np.sqrt(np.mean(variances / counts[usable])))


def leave_one_rater_out(ratings: ArrayLike) -> pd.DataFrame:
    """Every rating paired with the mean of the other ratings of its item.

    Parameters
    ----------
    ratings : array_like, shape (n_items, max_ratings)
        Ratings, NaN for absent ones; items with a single rating are skipped.

    Returns
    -------
    pandas.DataFrame
        Columns ``item`` (row index), ``rater`` (column index), ``rating`` and
        ``reference``, one row per rating. Score it with
        :func:`~voice_severity_bench.evaluation.regression_metrics` (``reference``
        as the truth) for the single-rater benchmark.

    Examples
    --------
    >>> pairs = leave_one_rater_out([[10.0, 20.0, 30.0], [40.0, 60.0, np.nan]])
    >>> pairs[["rating", "reference"]].values.tolist()
    [[10.0, 25.0], [20.0, 20.0], [30.0, 15.0], [40.0, 60.0], [60.0, 40.0]]
    """
    matrix = _matrix(ratings)
    rows = []
    for item, row in enumerate(matrix):
        present = np.flatnonzero(~np.isnan(row))
        if present.size < 2:
            continue
        for rater in present:
            others = row[present[present != rater]]
            rows.append((item, int(rater), float(row[rater]), float(others.mean())))
    return pd.DataFrame(rows, columns=["item", "rater", "rating", "reference"])
