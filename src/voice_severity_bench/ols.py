"""Ordinary least squares with its classical inference.

For a design :math:`X` (:math:`n \\times p`, intercept included) the estimate
:math:`\\hat\\beta = \\arg\\min \\lVert y - X\\beta \\rVert^2` is computed from a
QR factorization, never from :math:`(X^\\top X)^{-1}` directly. With
:math:`\\mathrm{RSS} = \\sum e_i^2`, :math:`\\mathrm{TSS} = \\sum (y_i - \\bar y)^2`
and :math:`\\mathrm{ESS} = \\mathrm{TSS} - \\mathrm{RSS}` (Draper & Smith, 1998):

.. math::

   \\hat\\sigma^2 = \\frac{\\mathrm{RSS}}{n - p}, \\qquad
   \\operatorname{SE}(\\hat\\beta_j) = \\hat\\sigma \\sqrt{[(X^\\top X)^{-1}]_{jj}}, \\qquad
   t_j = \\frac{\\hat\\beta_j}{\\operatorname{SE}(\\hat\\beta_j)},

.. math::

   R^2 = 1 - \\frac{\\mathrm{RSS}}{\\mathrm{TSS}}, \\qquad
   \\bar R^2 = 1 - (1 - R^2)\\frac{n - 1}{n - p}, \\qquad
   F = \\frac{\\mathrm{ESS}/(p - 1)}{\\mathrm{RSS}/(n - p)}.

Confidence intervals use :math:`t_{1-\\alpha/2,\\,n-p}`. The Gaussian
log-likelihood :math:`\\ell = -\\tfrac n2 (\\ln 2\\pi + \\ln(\\mathrm{RSS}/n) + 1)`
gives AIC :math:`= -2\\ell + 2p` (Akaike, 1974) and BIC
:math:`= -2\\ell + p \\ln n` (Schwarz, 1978).

A model with as many parameters as observations interpolates the data: its
residual degrees of freedom are zero, so :math:`\\hat\\sigma^2`, the standard
errors, the F test and every p-value are undefined. They are reported as
``nan`` with ``estimable = False``; the formulas above would give the
meaningless :math:`R^2 = 1` and divide by a zero residual variance. More parameters than
observations, or linearly dependent columns, raise
:class:`~voice_severity_bench.errors.DesignMatrixError`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, stats

from .errors import DesignMatrixError

_LOGGER = logging.getLogger(__name__)

#: Name given to the intercept column.
INTERCEPT_NAME: str = "const"


@dataclass(frozen=True, eq=False)
class OLSResult:
    """A fitted least-squares model with its inference.

    Attributes
    ----------
    names : tuple of str
        Parameter names, the intercept first when present.
    coef, se, t, p_values, ci_low, ci_high : numpy.ndarray
        Estimates, standard errors, t statistics, two-sided p-values and
        confidence limits, one entry per parameter.
    ci_level : float
        Confidence level of the intervals.
    n_obs, n_params, df_resid : int
        Observations, parameters and residual degrees of freedom.
    rss, tss, ess : float
        Residual, total and explained sums of squares.
    r2, adj_r2 : float
        Coefficient of determination and its adjusted version.
    f_stat, f_pvalue : float
        Overall F test of all non-intercept coefficients.
    sigma2 : float
        Residual variance :math:`\\mathrm{RSS}/(n-p)`.
    loglik, aic, bic : float
        Gaussian log-likelihood and information criteria.
    condition_number : float
        Ratio of the largest to the smallest singular value of the design.
    has_intercept : bool
        Whether the first column is the intercept.
    estimable : bool
        ``False`` when there are no residual degrees of freedom.
    design : numpy.ndarray
        The design matrix that was fitted (intercept column included).
    response, fitted, residuals : numpy.ndarray
        Observed, fitted and residual values.

    Examples
    --------
    >>> result = fit_ols(np.arange(6.0), [1.0, 3.1, 4.9, 7.2, 9.0, 10.8], ["x"])
    >>> result.coefficient_table()["term"].tolist(), result.df_resid, result.estimable
    (['const', 'x'], 4, True)
    >>> [round(float(v), 4) for v in result.predict([10.0])]
    [20.7857]
    """

    names: tuple[str, ...]
    coef: NDArray[np.float64]
    se: NDArray[np.float64]
    t: NDArray[np.float64]
    p_values: NDArray[np.float64]
    ci_low: NDArray[np.float64]
    ci_high: NDArray[np.float64]
    ci_level: float
    n_obs: int
    n_params: int
    df_resid: int
    rss: float
    tss: float
    ess: float
    r2: float
    adj_r2: float
    f_stat: float
    f_pvalue: float
    sigma2: float
    loglik: float
    aic: float
    bic: float
    condition_number: float
    has_intercept: bool
    estimable: bool
    design: NDArray[np.float64]
    response: NDArray[np.float64]
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]

    def coefficient_table(self) -> pd.DataFrame:
        """One row per parameter: estimate, SE, t, p and the confidence limits."""
        return pd.DataFrame(
            {
                "term": list(self.names),
                "estimate": self.coef,
                "std_error": self.se,
                "t": self.t,
                "p_value": self.p_values,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        """Model-level statistics for JSON output."""
        return {
            "n_obs": self.n_obs,
            "n_params": self.n_params,
            "df_resid": self.df_resid,
            "estimable": self.estimable,
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "f_stat": self.f_stat,
            "f_pvalue": self.f_pvalue,
            "sigma2": self.sigma2,
            "rss": self.rss,
            "tss": self.tss,
            "ess": self.ess,
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "condition_number": self.condition_number,
            "ci_level": self.ci_level,
        }

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predictions for new rows given *without* the intercept column."""
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if self.has_intercept:
            matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        if matrix.shape[1] != self.n_params:
            raise ValueError(
                f"expected {self.n_params - self.has_intercept} predictor columns, "
                f"got {matrix.shape[1] - self.has_intercept}"
            )
        return matrix @ self.coef

    def summary(self) -> str:
        """Plain-text report: coefficient table followed by the model statistics."""
        level = round(100 * self.ci_level)
        header = f"{'term':<14}{'estimate':>12}{'std err':>12}{'t':>9}{'p':>10}"
        header += f"{f'[{level}% CI':>13}{'':>12}"
        lines = [header, "-" * len(header)]
        for row in self.coefficient_table().itertuples(index=False):
            lines.append(
                f"{row.term:<14}{row.estimate:>12.5g}{row.std_error:>12.4g}{row.t:>9.3f}"
                f"{row.p_value:>10.3g}{row.ci_low:>13.5g}{row.ci_high:>12.5g}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"n = {self.n_obs}, parameters = {self.n_params}, residual df = {self.df_resid}"
            + ("" if self.estimable else "  (not estimable: no residual degrees of freedom)")
        )
        lines.append(f"R^2 = {self.r2:.4f}, adjusted R^2 = {self.adj_r2:.4f}")
        lines.append(f"F = {self.f_stat:.4g}, p = {self.f_pvalue:.4g}")
        lines.append(
            f"AIC = {self.aic:.4f}, BIC = {self.bic:.4f}, log-likelihood = {self.loglik:.4f}"
        )
        lines.append(f"condition number of the design = {self.condition_number:.4g}")
        return "\n".join(lines)


def fit_ols(
    X: ArrayLike,
    y: ArrayLike,
    names: Sequence[str] | None = None,
    *,
    add_intercept: bool = True,
    ci_level: float = 0.95,
) -> OLSResult:
    """Fit an ordinary least-squares regression and its classical inference.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_predictors)
        Predictors, *without* an intercept column; a 1-D array is one predictor.
    y : array_like, shape (n_samples,)
        Response.
    names : sequence of str, optional
        Predictor names (default ``x1, x2, ...``).
    add_intercept : bool, optional
        Prepend a column of ones named ``const``.
    ci_level : float, optional
        Confidence level of the coefficient intervals.

    Returns
    -------
    OLSResult
        Coefficients, inference, fit statistics and residuals.

    Raises
    ------
    DesignMatrixError
        If there are more parameters than observations or the columns are
        linearly dependent.
    ValueError
        For mismatched shapes or non-finite values.

    References
    ----------
    .. [1] N. R. Draper and H. Smith, Applied Regression Analysis, 3rd ed.
       New York, NY, USA: Wiley, 1998.
    .. [2] H. Akaike, "A new look at the statistical model identification,"
       IEEE Transactions on Automatic Control, vol. 19, no. 6, pp. 716-723, 1974.
    .. [3] G. Schwarz, "Estimating the dimension of a model," The Annals of
       Statistics, vol. 6, no. 2, pp. 461-464, 1978.

    Examples
    --------
    >>> x = np.arange(10.0)
    >>> result = fit_ols(x, 3.0 + 2.0 * x + np.array([1, -1] * 5) * 0.1, ["x"])
    >>> result.names, [round(float(b), 4) for b in result.coef], round(result.r2, 5)
    (('const', 'x'), [3.0273, 1.9939], 0.9997)
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    response = np.asarray(y, dtype=np.float64).ravel()
    if matrix.ndim != 2 or matrix.shape[0] != response.size:
        raise ValueError(
            f"X must be (n_samples x n_predictors) with n_samples={response.size}, "
            f"got shape {matrix.shape}"
        )
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(response))):
        raise ValueError("X and y must not contain NaN or infinite values")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie inside (0, 1), got {ci_level!r}")
    labels = (
        tuple(names) if names is not None else tuple(f"x{i + 1}" for i in range(matrix.shape[1]))
    )
    if len(labels) != matrix.shape[1]:
        raise ValueError(f"{len(labels)} names for {matrix.shape[1]} predictor columns")
    if add_intercept:
        matrix = np.column_stack([np.ones(response.size), matrix])
        labels = (INTERCEPT_NAME, *labels)
    n, p = matrix.shape
    if p == 0:
        raise DesignMatrixError("the design has no columns")
    if p > n:
        raise DesignMatrixError(
            f"{p} parameters cannot be estimated from {n} observations; "
            "drop predictors (for example with stepwise selection) or add data"
        )

    singular = linalg.svd(matrix, compute_uv=False)
    tolerance = singular[0] * max(n, p) * np.finfo(np.float64).eps
    rank = int(np.sum(singular > tolerance))
    if rank < p:
        raise DesignMatrixError(
            f"the design is rank deficient (rank {rank} < {p} columns): some predictors are "
            "exact linear combinations of others"
        )
    condition = float(singular[0] / singular[-1])

    q, r = linalg.qr(matrix, mode="economic")
    coef = linalg.solve_triangular(r, q.T @ response)
    fitted = matrix @ coef
    residuals = response - fitted
    rss = float(residuals @ residuals)
    centered = response - response.mean() if add_intercept else response
    tss = float(centered @ centered)
    ess = tss - rss
    df_resid = n - p
    df_model = p - int(add_intercept)
    r2 = 1.0 - rss / tss if tss > 0.0 else math.nan

    nan_vector = np.full(p, np.nan)
    estimable = df_resid > 0
    if estimable:
        sigma2 = rss / df_resid
        r_inverse = linalg.solve_triangular(r, np.eye(p))
        se = np.sqrt(sigma2 * np.sum(r_inverse**2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = coef / se
        p_values = 2.0 * stats.t.sf(np.abs(t_values), df_resid)
        half = stats.t.ppf(0.5 + ci_level / 2.0, df_resid) * se
        ci_low, ci_high = coef - half, coef + half
        adj_r2 = 1.0 - (1.0 - r2) * (n - int(add_intercept)) / df_resid
        if df_model > 0 and rss > 0.0:
            f_stat = (ess / df_model) / sigma2
            f_pvalue = float(stats.f.sf(f_stat, df_model, df_resid))
        else:
            f_stat = f_pvalue = math.nan
    else:
        _LOGGER.warning(
            "%d parameters for %d observations: the fit interpolates the data and no "
            "standard error, test or interval is defined",
            p,
            n,
        )
        sigma2 = adj_r2 = f_stat = f_pvalue = math.nan
        se = t_values = p_values = ci_low = ci_high = nan_vector
        r2 = math.nan

    if estimable and rss > 0.0:
        loglik = -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss / n) + 1.0)
        aic = -2.0 * loglik + 2.0 * p
        bic = -2.0 * loglik + p * math.log(n)
    else:
        loglik = aic = bic = math.nan

    return OLSResult(
        names=labels,
        coef=coef,
        se=np.asarray(se, dtype=np.float64),
        t=np.asarray(t_values, dtype=np.float64),
        p_values=np.asarray(p_values, dtype=np.float64),
        ci_low=np.asarray(ci_low, dtype=np.float64),
        ci_high=np.asarray(ci_high, dtype=np.float64),
        ci_level=ci_level,
        n_obs=n,
        n_params=p,
        df_resid=df_resid,
        rss=rss,
        tss=tss,
        ess=ess,
        r2=r2,
        adj_r2=adj_r2,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        sigma2=sigma2,
        loglik=loglik,
        aic=aic,
        bic=bic,
        condition_number=condition,
        has_intercept=add_intercept,
        estimable=estimable,
        design=matrix,
        response=response,
        fitted=fitted,
        residuals=residuals,
    )
