"""Residual, influence and collinearity diagnostics of a least-squares fit.

The standard errors, tests and intervals of :mod:`voice_severity_bench.ols`
assume normal, homoscedastic and independent errors. A fit can be checked
against each of them, and for influential points and collinearity:

* normal residuals: Shapiro-Wilk (Shapiro & Wilk, 1965) and Jarque-Bera
  :math:`JB = \\tfrac n6 (S^2 + (K - 3)^2/4)` (Jarque & Bera, 1980);
* constant variance: the Breusch-Pagan Lagrange multiplier test in its
  studentized form, which regresses :math:`e_i^2` on the design and compares
  :math:`nR^2_{aux}` with :math:`\\chi^2_{p-1}` (Breusch & Pagan, 1979;
  Koenker, 1981);
* independent residuals: the Durbin-Watson statistic
  :math:`d = \\sum (e_t - e_{t-1})^2 / \\sum e_t^2`, near 2 without lag-one
  autocorrelation (Durbin & Watson, 1950);
* influential observations: leverages :math:`h_{ii}`, the diagonal of
  the hat matrix :math:`H = X(X^\\top X)^{-1}X^\\top`, and Cook's distance
  :math:`D_i = \\frac{e_i^2}{p\\,\\hat\\sigma^2}\\frac{h_{ii}}{(1-h_{ii})^2}`
  (Cook, 1977);
* collinearity: variance inflation factors
  :math:`\\mathrm{VIF}_j = 1/(1 - R_j^2)` from regressing each predictor on
  the others (Marquardt, 1970), and the condition number of the design.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, stats

from .ols import OLSResult


@dataclass(frozen=True)
class HypothesisTest:
    """A test statistic with its p-value.

    Attributes
    ----------
    statistic : float
        Value of the test statistic.
    p_value : float
        p-value under the null hypothesis (``nan`` when it is undefined).

    Examples
    --------
    >>> HypothesisTest(statistic=2.5, p_value=0.29)
    HypothesisTest(statistic=2.5, p_value=0.29)
    """

    statistic: float
    p_value: float


def _residual_vector(residuals: ArrayLike, minimum: int) -> NDArray[np.float64]:
    values = np.asarray(residuals, dtype=np.float64).ravel()
    if values.size < minimum:
        raise ValueError(f"need at least {minimum} residuals, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("residuals contain NaN or infinite values")
    return values


def shapiro_wilk(residuals: ArrayLike) -> HypothesisTest:
    """Shapiro-Wilk test of normality.

    References
    ----------
    .. [1] S. S. Shapiro and M. B. Wilk, "An analysis of variance test for
       normality (complete samples)," Biometrika, vol. 52, no. 3-4,
       pp. 591-611, 1965.

    Examples
    --------
    >>> sample = np.random.default_rng(0).standard_normal(200)
    >>> bool(shapiro_wilk(sample).p_value > 0.05), bool(shapiro_wilk(sample**3).p_value < 1e-6)
    (True, True)
    """
    values = _residual_vector(residuals, 3)
    if np.ptp(values) == 0.0:
        return HypothesisTest(math.nan, math.nan)
    statistic, p_value = stats.shapiro(values)
    return HypothesisTest(float(statistic), float(p_value))


def jarque_bera(residuals: ArrayLike) -> HypothesisTest:
    """Jarque-Bera test of normality from sample skewness and kurtosis.

    :math:`JB = \\tfrac n6 \\left(S^2 + \\tfrac14 (K - 3)^2\\right)` with the
    biased moment estimates, compared with :math:`\\chi^2_2`.

    References
    ----------
    .. [1] C. M. Jarque and A. K. Bera, "Efficient tests for normality,
       homoscedasticity and serial independence of regression residuals,"
       Economics Letters, vol. 6, no. 3, pp. 255-259, 1980.

    Examples
    --------
    >>> result = jarque_bera([-2.0, -1.0, 0.0, 1.0, 2.0, 7.0])
    >>> round(result.statistic, 4), round(result.p_value, 4)
    (1.1054, 0.5754)
    """
    values = _residual_vector(residuals, 3)
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if variance == 0.0:
        return HypothesisTest(math.nan, math.nan)
    skew = float(np.mean(centered**3)) / variance**1.5
    kurtosis = float(np.mean(centered**4)) / variance**2
    statistic = values.size / 6.0 * (skew**2 + 0.25 * (kurtosis - 3.0) ** 2)
    return HypothesisTest(statistic, float(stats.chi2.sf(statistic, 2)))


def breusch_pagan(residuals: ArrayLike, design: ArrayLike) -> HypothesisTest:
    """Studentized Breusch-Pagan test of heteroscedasticity.

    Parameters
    ----------
    residuals : array_like, shape (n,)
        OLS residuals.
    design : array_like, shape (n, p)
        The fitted design, intercept column included.

    Returns
    -------
    HypothesisTest
        :math:`LM = nR^2` of the auxiliary regression of :math:`e^2` on the
        design, with :math:`p - 1` degrees of freedom.

    References
    ----------
    .. [1] T. S. Breusch and A. R. Pagan, "A simple test for heteroscedasticity
       and random coefficient variation," Econometrica, vol. 47, no. 5,
       pp. 1287-1294, 1979.
    .. [2] R. Koenker, "A note on studentizing a test for heteroscedasticity,"
       Journal of Econometrics, vol. 17, no. 1, pp. 107-112, 1981.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> x = np.linspace(1.0, 10.0, 300)
    >>> design = np.column_stack([np.ones_like(x), x])
    >>> bool(breusch_pagan(rng.normal(0, x), design).p_value < 1e-6)
    True
    """
    values = _residual_vector(residuals, 3)
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        raise ValueError(f"design must have {values.size} rows, got shape {matrix.shape}")
    squared = values**2
    coef, *_ = np.linalg.lstsq(matrix, squared, rcond=None)
    fitted = matrix @ coef
    total = float(np.sum((squared - squared.mean()) ** 2))
    if total == 0.0:
        return HypothesisTest(math.nan, math.nan)
    r2 = 1.0 - float(np.sum((squared - fitted) ** 2)) / total
    df = int(np.linalg.matrix_rank(matrix)) - 1
    if df < 1:
        return HypothesisTest(math.nan, math.nan)
    statistic = values.size * r2
    return HypothesisTest(statistic, float(stats.chi2.sf(statistic, df)))


def durbin_watson(residuals: ArrayLike) -> float:
    """Durbin-Watson statistic of lag-one residual autocorrelation.

    References
    ----------
    .. [1] J. Durbin and G. S. Watson, "Testing for serial correlation in least
       squares regression. I," Biometrika, vol. 37, no. 3-4, pp. 409-428, 1950.

    Examples
    --------
    >>> durbin_watson([1.0, -1.0, 1.0, -1.0]), durbin_watson([1.0, 1.0, -1.0, -1.0])
    (3.0, 1.0)
    """
    values = _residual_vector(residuals, 2)
    denominator = float(values @ values)
    if denominator == 0.0:
        return math.nan
    return float(np.sum(np.diff(values) ** 2) / denominator)


def leverage(design: ArrayLike) -> NDArray[np.float64]:
    """Diagonal of the hat matrix, from a thin QR factorization.

    Examples
    --------
    >>> design = np.column_stack([np.ones(5), np.arange(5.0)])
    >>> leverage(design).round(2).tolist()
    [0.6, 0.3, 0.2, 0.3, 0.6]
    """
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {matrix.shape}")
    q, _ = linalg.qr(matrix, mode="economic")
    return np.sum(q**2, axis=1)


def cooks_distance(result: OLSResult) -> NDArray[np.float64]:
    """Cook's distance of every observation of a fitted model.

    References
    ----------
    .. [1] R. D. Cook, "Detection of influential observation in linear
       regression," Technometrics, vol. 19, no. 1, pp. 15-18, 1977.

    Examples
    --------
    >>> from voice_severity_bench.ols import fit_ols
    >>> x = np.arange(8.0)
    >>> y = 2.0 * x + np.array([0.1, -0.2, 0.1, 0.0, -0.1, 0.2, -0.1, 3.0])
    >>> int(np.argmax(cooks_distance(fit_ols(x, y))))
    7
    """
    if not result.estimable:
        return np.full(result.n_obs, np.nan)
    hat = leverage(result.design)
    with np.errstate(divide="ignore", invalid="ignore"):
        return result.residuals**2 / (result.n_params * result.sigma2) * hat / (1.0 - hat) ** 2


def studentized_residuals(result: OLSResult) -> NDArray[np.float64]:
    """Internally studentized residuals :math:`e_i / (\\hat\\sigma \\sqrt{1 - h_{ii}})`.

    Examples
    --------
    >>> from voice_severity_bench.ols import fit_ols
    >>> x = np.arange(8.0)
    >>> y = 2.0 * x + np.array([0.1, -0.2, 0.1, 0.0, -0.1, 0.2, -0.1, 3.0])
    >>> int(np.argmax(np.abs(studentized_residuals(fit_ols(x, y)))))
    7
    """
    if not result.estimable:
        return np.full(result.n_obs, np.nan)
    hat = leverage(result.design)
    with np.errstate(divide="ignore", invalid="ignore"):
        return result.residuals / np.sqrt(result.sigma2 * (1.0 - hat))


def vif(X: ArrayLike, names: Sequence[str] | None = None) -> pd.DataFrame:
    """Variance inflation factor of every predictor.

    Each predictor is regressed (with an intercept) on all the others;
    :math:`\\mathrm{VIF}_j = 1/(1 - R_j^2)`, infinite for an exact linear
    dependence. Values above about 10 are the usual warning sign.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_predictors)
        Predictors *without* the intercept column.
    names : sequence of str, optional
        Predictor names.

    Returns
    -------
    pandas.DataFrame
        Columns ``term``, ``r2`` and ``vif``.

    References
    ----------
    .. [1] D. W. Marquardt, "Generalized inverses, ridge regression, biased
       linear estimation, and nonlinear estimation," Technometrics, vol. 12,
       no. 3, pp. 591-612, 1970.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.normal(size=(2, 50))
    >>> table = vif(np.column_stack([a, b, a + b + rng.normal(0, 0.05, 50)]), ["a", "b", "sum"])
    >>> bool((table["vif"] > 100).all())
    True
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"VIF needs at least two predictor columns, got shape {matrix.shape}")
    labels = list(names) if names is not None else [f"x{i + 1}" for i in range(matrix.shape[1])]
    if len(labels) != matrix.shape[1]:
        raise ValueError(f"{len(labels)} names for {matrix.shape[1]} predictor columns")
    rows = []
    for column in range(matrix.shape[1]):
        target = matrix[:, column]
        others = np.column_stack([np.ones(matrix.shape[0]), np.delete(matrix, column, axis=1)])
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        residual = target - others @ coef
        total = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - float(residual @ residual) / total if total > 0.0 else math.nan
        r2 = min(r2, 1.0)
        value = math.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"term": labels[column], "r2": r2, "vif": value})
    return pd.DataFrame(rows)


def condition_number(X: ArrayLike) -> float:
    """Ratio of the largest to the smallest singular value of a matrix.

    Examples
    --------
    >>> condition_number(np.diag([10.0, 1.0]))
    10.0
    """
    singular = linalg.svd(np.asarray(X, dtype=np.float64), compute_uv=False)
    if singular[-1] == 0.0:
        return math.inf
    return float(singular[0] / singular[-1])


def residual_diagnostics(result: OLSResult) -> dict[str, float]:
    """Every residual test of a fit in one mapping (for ``regression_summary.json``).

    Returns
    -------
    dict
        ``shapiro_w``, ``shapiro_p``, ``jarque_bera``, ``jarque_bera_p``,
        ``breusch_pagan``, ``breusch_pagan_p``, ``durbin_watson``,
        ``max_leverage``, ``max_cooks_distance`` and ``condition_number``;
        ``nan`` where a test is undefined.

    Examples
    --------
    >>> from voice_severity_bench.ols import fit_ols
    >>> rng = np.random.default_rng(3)
    >>> x = rng.uniform(0, 10, 80)
    >>> report = residual_diagnostics(fit_ols(x, 1.0 + 0.5 * x + rng.normal(0, 1, 80)))
    >>> sorted(report)[:3], bool(1.0 < report["durbin_watson"] < 3.0)
    (['breusch_pagan', 'breusch_pagan_p', 'condition_number'], True)
    """
    nan = math.nan
    report = {
        "shapiro_w": nan,
        "shapiro_p": nan,
        "jarque_bera": nan,
        "jarque_bera_p": nan,
        "breusch_pagan": nan,
        "breusch_pagan_p": nan,
        "durbin_watson": nan,
        "max_leverage": float(np.max(leverage(result.design))),
        "max_cooks_distance": nan,
        "condition_number": result.condition_number,
    }
    if not result.estimable or result.n_obs < 3:
        return report
    shapiro = shapiro_wilk(result.residuals)
    jb = jarque_bera(result.residuals)
    bp = breusch_pagan(result.residuals, result.design)
    report.update(
        shapiro_w=shapiro.statistic,
        shapiro_p=shapiro.p_value,
        jarque_bera=jb.statistic,
        jarque_bera_p=jb.p_value,
        breusch_pagan=bp.statistic,
        breusch_pagan_p=bp.p_value,
        durbin_watson=durbin_watson(result.residuals),
        max_cooks_distance=float(np.nanmax(cooks_distance(result))),
    )
    return report


def influence_table(result: OLSResult) -> pd.DataFrame:
    """Per-observation fitted values, residuals and influence measures.

    Returns
    -------
    pandas.DataFrame
        Columns ``observed``, ``fitted``, ``residual``, ``studentized``,
        ``leverage``, ``cooks_distance`` and ``normal_quantile`` (the
        theoretical normal quantile of each studentized residual's rank, for
        a Q-Q plot).

    Examples
    --------
    >>> from voice_severity_bench.ols import fit_ols
    >>> x = np.arange(10.0)
    >>> table = influence_table(fit_ols(x, 1.0 + x + np.sin(x)))
    >>> table.shape, round(float(table["leverage"].sum()), 6)
    ((10, 7), 2.0)
    """
    student = studentized_residuals(result)
    n = result.n_obs
    quantiles = np.full(n, np.nan)
    if np.all(np.isfinite(student)):
        ranks = stats.rankdata(student, method="ordinal")
        quantiles = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
    return pd.DataFrame(
        {
            "observed": result.response,
            "fitted": result.fitted,
            "residual": result.residuals,
            "studentized": student,
            "leverage": leverage(result.design),
            "cooks_distance": cooks_distance(result),
            "normal_quantile": quantiles,
        }
    )
