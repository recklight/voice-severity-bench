"""Association and agreement between voice features and continuous severity ratings.

Two questions are kept apart on purpose:

* Association: does a feature move with the rating? Pearson's :math:`r`
  with a Fisher :math:`z` confidence interval (Fisher, 1915) and Spearman's
  rank correlation :math:`\\rho` (Spearman, 1904), each feature tested
  separately and the family of p-values adjusted by Holm's step-down method
  (Holm, 1979) or the Benjamini-Hochberg false discovery rate (Benjamini &
  Hochberg, 1995).
* Agreement: do predicted ratings *equal* the clinical ones? A high
  correlation does not prove it: predictions that are all 10 points too high
  correlate perfectly. Lin's concordance correlation coefficient

  .. math:: \\rho_c = \\frac{2 s_{xy}}{s_x^2 + s_y^2 + (\\bar x - \\bar y)^2}

  penalizes both scatter and bias (Lin, 1989), and the Bland-Altman analysis
  reports the mean difference and its 95 % limits of agreement
  :math:`\\bar d \\pm 1.96\\, s_d` (Bland & Altman, 1986).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

_LOGGER = logging.getLogger(__name__)

#: p-value adjustments accepted by ``analysis.correction``.
P_VALUE_CORRECTIONS: tuple[str, ...] = ("holm", "fdr_bh")


def _paired(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first = np.asarray(x, dtype=np.float64).ravel()
    second = np.asarray(y, dtype=np.float64).ravel()
    if first.shape != second.shape:
        raise ValueError(f"paired samples differ in length: {first.size} and {second.size}")
    if first.size < 3:
        raise ValueError(f"need at least 3 pairs, got {first.size}")
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        raise ValueError("paired samples contain NaN or infinite values")
    return first, second


@dataclass(frozen=True)
class CorrelationResult:
    """A correlation coefficient with its interval and test.

    Attributes
    ----------
    coefficient : float
        Pearson's r (or ``nan`` for a constant sample).
    ci_low, ci_high : float
        Fisher z confidence interval.
    p_value : float
        Two-sided p-value of :math:`H_0: \\rho = 0`.
    n : int
        Number of pairs.
    ci_level : float
        Confidence level of the interval.

    Examples
    --------
    >>> result = pearson_ci([1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 1.9, 3.2, 3.9, 5.1])
    >>> result.n, round(result.coefficient, 4), sorted(result.as_dict())[:2]
    (5, 0.9964, ['ci_high', 'ci_level'])
    """

    coefficient: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    ci_level: float

    def as_dict(self) -> dict[str, float]:
        """Plain mapping for JSON output."""
        return {
            "coefficient": self.coefficient,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n": self.n,
            "ci_level": self.ci_level,
        }


def pearson_ci(x: ArrayLike, y: ArrayLike, ci_level: float = 0.95) -> CorrelationResult:
    """Pearson correlation with a Fisher z confidence interval.

    The interval is :math:`\\tanh(\\operatorname{atanh} r \\pm z_{1-\\alpha/2} / \\sqrt{n - 3})`.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Paired samples, :math:`n \\ge 4` for an interval.
    ci_level : float, optional
        Confidence level.

    Returns
    -------
    CorrelationResult
        Coefficient, interval and p-value; all ``nan`` when a sample is constant.

    References
    ----------
    .. [1] R. A. Fisher, "Frequency distribution of the values of the
       correlation coefficient in samples from an indefinitely large
       population," Biometrika, vol. 10, no. 4, pp. 507-521, 1915.

    Examples
    --------
    >>> x = np.arange(20.0)
    >>> result = pearson_ci(x, x + np.sin(x))
    >>> round(result.coefficient, 3), result.ci_low < result.coefficient < result.ci_high
    (0.993, True)
    """
    first, second = _paired(x, y)
    n = first.size
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return CorrelationResult(math.nan, math.nan, math.nan, math.nan, n, ci_level)
    r, p_value = stats.pearsonr(first, second)
    r = float(np.clip(r, -1.0, 1.0))
    if n > 3 and abs(r) < 1.0:
        half = stats.norm.ppf(0.5 + ci_level / 2.0) / math.sqrt(n - 3)
        z = math.atanh(r)
        low, high = math.tanh(z - half), math.tanh(z + half)
    elif abs(r) == 1.0:
        low = high = r
    else:
        low = high = math.nan
    return CorrelationResult(r, low, high, float(p_value), n, ci_level)


def spearman(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Spearman's rank correlation and its two-sided p-value.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Paired samples.

    Returns
    -------
    tuple of float
        ``(rho, p_value)``; ``nan`` for a constant sample.

    References
    ----------
    .. [1] C. Spearman, "The proof and measurement of association between two
       things," The American Journal of Psychology, vol. 15, no. 1, pp. 72-101, 1904.

    Examples
    --------
    >>> rho, p = spearman([1, 2, 3, 4, 5], [2, 4, 8, 16, 32])
    >>> round(rho, 6)
    1.0
    """
    first, second = _paired(x, y)
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return math.nan, math.nan
    rho, p_value = stats.spearmanr(first, second)
    return float(rho), float(p_value)


def concordance_ccc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Lin's concordance correlation coefficient.

    Uses the population (``1/n``) moments, as in Lin (1989).

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        Reference ratings and predictions.

    Returns
    -------
    float
        :math:`\\rho_c \\in [-1, 1]`; 1 only when every prediction equals its
        reference.

    References
    ----------
    .. [1] L. I.-K. Lin, "A concordance correlation coefficient to evaluate
       reproducibility," Biometrics, vol. 45, no. 1, pp. 255-268, 1989.

    Examples
    --------
    A constant offset keeps Pearson's r at 1 but lowers the concordance:

    >>> truth = np.linspace(0.0, 100.0, 11)
    >>> concordance_ccc(truth, truth), round(concordance_ccc(truth, truth + 10.0), 4)
    (1.0, 0.9524)
    """
    first, second = _paired(y_true, y_pred)
    covariance = np.mean((first - first.mean()) * (second - second.mean()))
    denominator = first.var() + second.var() + (first.mean() - second.mean()) ** 2
    if denominator == 0.0:
        return 1.0 if np.array_equal(first, second) else math.nan
    return float(2.0 * covariance / denominator)


@dataclass(frozen=True, eq=False)
class BlandAltmanResult:
    """Bland-Altman agreement between two measurements of the same quantity.

    Attributes
    ----------
    means : numpy.ndarray
        Pairwise means :math:`(a_i + b_i) / 2`.
    differences : numpy.ndarray
        Pairwise differences :math:`a_i - b_i`.
    bias : float
        Mean difference.
    sd : float
        Sample standard deviation of the differences (``ddof=1``).
    lower_loa, upper_loa : float
        Limits of agreement :math:`\\bar d \\pm z\\, s_d`.
    ci_level : float
        Coverage of the limits (0.95 gives the classical 1.96).

    Examples
    --------
    >>> result = bland_altman([10.0, 20.0, 30.0], [9.0, 21.0, 27.0])
    >>> result.differences.tolist(), result.means.tolist(), sorted(result.as_dict())[0]
    ([1.0, -1.0, 3.0], [9.5, 20.5, 28.5], 'bias')
    """

    means: NDArray[np.float64]
    differences: NDArray[np.float64]
    bias: float
    sd: float
    lower_loa: float
    upper_loa: float
    ci_level: float

    def as_dict(self) -> dict[str, float]:
        """Summary statistics for JSON output."""
        return {
            "bias": self.bias,
            "sd": self.sd,
            "lower_loa": self.lower_loa,
            "upper_loa": self.upper_loa,
            "ci_level": self.ci_level,
        }


def bland_altman(a: ArrayLike, b: ArrayLike, ci_level: float = 0.95) -> BlandAltmanResult:
    """Mean difference and limits of agreement of two paired measurements.

    Parameters
    ----------
    a, b : array_like, shape (n,)
        Paired measurements, for instance predicted and clinical ratings.
    ci_level : float, optional
        Expected share of differences inside the limits.

    Returns
    -------
    BlandAltmanResult
        Means, differences, bias and limits.

    References
    ----------
    .. [1] J. M. Bland and D. G. Altman, "Statistical methods for assessing
       agreement between two methods of clinical measurement," The Lancet,
       vol. 327, no. 8476, pp. 307-310, 1986.

    Examples
    --------
    >>> result = bland_altman([10.0, 20.0, 30.0, 40.0], [8.0, 21.0, 27.0, 38.0])
    >>> result.bias, round(result.upper_loa, 3)
    (1.5, 4.895)
    """
    first, second = _paired(a, b)
    differences = first - second
    bias = float(differences.mean())
    sd = float(differences.std(ddof=1))
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    return BlandAltmanResult(
        means=(first + second) / 2.0,
        differences=differences,
        bias=bias,
        sd=sd,
        lower_loa=bias - z * sd,
        upper_loa=bias + z * sd,
        ci_level=ci_level,
    )


def adjust_pvalues(p_values: ArrayLike, method: str = "holm") -> NDArray[np.float64]:
    """Adjust a family of p-values for multiple comparisons.

    ``nan`` entries (untestable hypotheses) stay ``nan`` and do not count
    towards the family size.

    Parameters
    ----------
    p_values : array_like, shape (m,)
        Unadjusted p-values.
    method : {"holm", "fdr_bh"}, optional
        Holm's step-down family-wise error control, or the Benjamini-Hochberg
        step-up false discovery rate control.

    Returns
    -------
    numpy.ndarray, shape (m,)
        Adjusted p-values in the input order, capped at 1 and non-decreasing
        in the ranking of the unadjusted p-values.

    References
    ----------
    .. [1] S. Holm, "A simple sequentially rejective multiple test procedure,"
       Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65-70, 1979.
    .. [2] Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate:
       a practical and powerful approach to multiple testing," Journal of the
       Royal Statistical Society, Series B, vol. 57, no. 1, pp. 289-300, 1995.

    Examples
    --------
    >>> adjust_pvalues([0.01, 0.04, 0.03], "holm").round(3).tolist()
    [0.03, 0.06, 0.06]
    >>> adjust_pvalues([0.01, 0.04, 0.03], "fdr_bh").round(3).tolist()
    [0.03, 0.04, 0.04]
    """
    if method not in P_VALUE_CORRECTIONS:
        raise ValueError(
            f"unknown p-value correction {method!r}; choose one of {list(P_VALUE_CORRECTIONS)}"
        )
    values = np.asarray(p_values, dtype=np.float64).ravel()
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    m = valid.size
    if m == 0:
        return adjusted
    order = valid[np.argsort(values[valid], kind="mergesort")]
    ranked = values[order]
    if method == "holm":
        scaled = (m - np.arange(m)) * ranked
        stepped = np.maximum.accumulate(scaled)
    else:
        scaled = ranked * m / np.arange(1, m + 1)
        stepped = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def feature_target_correlations(
    features: ArrayLike,
    target: ArrayLike,
    names: Sequence[str],
    *,
    ci_level: float = 0.95,
    correction: str = "holm",
) -> pd.DataFrame:
    """Correlate every feature column with a continuous target.

    Parameters
    ----------
    features : array_like, shape (n_items, n_features)
        Utterance-level features.
    target : array_like, shape (n_items,)
        Severity ratings.
    names : sequence of str
        Feature names, one per column.
    ci_level : float, optional
        Level of the Pearson intervals.
    correction : {"holm", "fdr_bh"}, optional
        Adjustment applied separately to the Pearson and the Spearman p-values.

    Returns
    -------
    pandas.DataFrame
        One row per feature, sorted by decreasing :math:`|r|`, with columns
        ``feature``, ``pearson_r``, ``ci_low``, ``ci_high``, ``pearson_p``,
        ``pearson_p_adjusted``, ``spearman_rho``, ``spearman_p``,
        ``spearman_p_adjusted``. Constant features get ``nan`` and sort last.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> severity = rng.uniform(0, 100, 40)
    >>> X = np.column_stack([severity + rng.normal(0, 10, 40), rng.normal(size=40)])
    >>> table = feature_target_correlations(X, severity, ["related", "noise"])
    >>> table["feature"].tolist(), bool(table["pearson_p_adjusted"].iloc[0] < 1e-6)
    (['related', 'noise'], True)
    """
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64).ravel()
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        raise ValueError(
            f"features must be (n_items x n_features) with n_items={values.size}, "
            f"got shape {matrix.shape}"
        )
    if matrix.shape[1] != len(names):
        raise ValueError(f"{len(names)} names for {matrix.shape[1]} feature columns")
    rows = []
    for column, name in enumerate(names):
        pearson = pearson_ci(matrix[:, column], values, ci_level)
        rho, rho_p = spearman(matrix[:, column], values)
        rows.append(
            {
                "feature": name,
                "pearson_r": pearson.coefficient,
                "ci_low": pearson.ci_low,
                "ci_high": pearson.ci_high,
                "pearson_p": pearson.p_value,
                "spearman_rho": rho,
                "spearman_p": rho_p,
            }
        )
    table = pd.DataFrame(rows)
    table["pearson_p_adjusted"] = adjust_pvalues(table["pearson_p"].to_numpy(), correction)
    table["spearman_p_adjusted"] = adjust_pvalues(table["spearman_p"].to_numpy(), correction)
    n_constant = int(table["pearson_r"].isna().sum())
    if n_constant:
        _LOGGER.warning("%d constant feature(s) have no defined correlation", n_constant)
    table["_order"] = -table["pearson_r"].abs().fillna(-1.0)
    table = table.sort_values("_order", kind="mergesort").drop(columns="_order")
    columns = [
        "feature",
        "pearson_r",
        "ci_low",
        "ci_high",
        "pearson_p",
        "pearson_p_adjusted",
        "spearman_rho",
        "spearman_p",
        "spearman_p_adjusted",
    ]
    return table[columns].reset_index(drop=True)
