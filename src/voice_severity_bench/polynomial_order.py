"""Choosing the degree of a polynomial regression in and out of sample.

In-sample statistics always favor the bigger model: :math:`R^2` can only
grow with the degree and reaches 1 once the polynomial interpolates the data.
The sweep therefore reports, for every degree,

* the in-sample view: :math:`R^2`, adjusted :math:`R^2`, the overall F test,
  the residual sum of squares, AIC (Akaike, 1974) and BIC (Schwarz, 1978);
* the out-of-sample view: leave-one-out RMSE from the closed-form PRESS
  residuals :math:`e_i / (1 - h_{ii})`, and k-fold RMSE in which the
  polynomial basis is refitted on every training fold (Stone, 1974).

Degrees with no residual degrees of freedom are marked as not estimable and
get ``nan`` instead of the meaningless :math:`R^2 = 1`. The recommended
degree minimizes the configured criterion; ties go to the lower degree.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold

from .design_matrix import PolynomialBasis
from .errors import DesignMatrixError
from .ols import fit_ols
from .regression_diagnostics import leverage

_LOGGER = logging.getLogger(__name__)

#: Criteria accepted by ``analysis.order_criterion``.
ORDER_CRITERIA: tuple[str, ...] = ("kfold_rmse", "loocv_rmse", "bic", "aic")

#: Columns of the sweep table, in order.
_COLUMNS: tuple[str, ...] = (
    "degree",
    "n_params",
    "df_resid",
    "estimable",
    "r2",
    "adj_r2",
    "f_stat",
    "f_pvalue",
    "sse",
    "aic",
    "bic",
    "loocv_rmse",
    "kfold_rmse",
)


@dataclass(frozen=True, eq=False)
class OrderSweepResult:
    """Fit statistics of every polynomial degree.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per degree; columns ``degree``, ``n_params``, ``df_resid``,
        ``estimable``, ``r2``, ``adj_r2``, ``f_stat``, ``f_pvalue``, ``sse``,
        ``aic``, ``bic``, ``loocv_rmse`` and ``kfold_rmse``.
    criterion : str
        Column used for the recommendation.
    recommended_degree : int or None
        Degree with the smallest criterion, ``None`` if no degree is estimable.

    Examples
    --------
    >>> x = np.linspace(0.0, 1.0, 12)
    >>> sweep_polynomial_order(x, 2.0 * x + 1.0 + 0.01 * np.cos(9 * x), [1, 2], rng=0).as_dict()
    {'criterion': 'kfold_rmse', 'recommended_degree': 1}
    """

    table: pd.DataFrame
    criterion: str
    recommended_degree: int | None

    def as_dict(self) -> dict[str, object]:
        """Recommendation summary for JSON output."""
        return {"criterion": self.criterion, "recommended_degree": self.recommended_degree}


def _kfold_rmse(
    x: np.ndarray, y: np.ndarray, degree: int, basis: str, n_splits: int, random_state: int
) -> float:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions = np.empty_like(y)
    for train, test in splitter.split(x):
        if train.size <= degree + 1:
            return math.nan  # a training fold this small leaves no residual freedom
        try:
            transform = PolynomialBasis(degree, basis).fit(x[train])
            fold = fit_ols(transform.transform(x[train]), y[train])
        except (DesignMatrixError, ValueError):
            return math.nan
        if fold.df_resid <= 0:
            return math.nan
        predictions[test] = fold.predict(transform.transform(x[test]))
    return float(np.sqrt(np.mean((y - predictions) ** 2)))


def _degree_row(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    basis: str,
    n_splits: int,
    random_state: int,
) -> dict[str, object]:
    n_params = degree + 1
    row: dict[str, object] = dict.fromkeys(_COLUMNS, math.nan)
    row.update(degree=degree, n_params=n_params, df_resid=x.size - n_params, estimable=False)
    if x.size - n_params <= 0:
        return row
    try:
        result = fit_ols(PolynomialBasis(degree, basis).fit_transform(x), y)
    except (DesignMatrixError, ValueError) as exc:
        _LOGGER.warning("degree %d cannot be fitted: %s", degree, exc)
        return row
    hat = leverage(result.design)
    with np.errstate(divide="ignore", invalid="ignore"):
        press = result.residuals / (1.0 - hat)
    loocv = float(np.sqrt(np.mean(press**2))) if np.all(hat < 1.0 - 1e-10) else math.nan
    row.update(
        estimable=True,
        r2=result.r2,
        adj_r2=result.adj_r2,
        f_stat=result.f_stat,
        f_pvalue=result.f_pvalue,
        sse=result.rss,
        aic=result.aic,
        bic=result.bic,
        loocv_rmse=loocv,
        kfold_rmse=_kfold_rmse(x, y, degree, basis, n_splits, random_state)
        if x.size >= n_splits
        else math.nan,
    )
    return row


def sweep_polynomial_order(
    x: ArrayLike,
    y: ArrayLike,
    degrees: Sequence[int] | None = None,
    *,
    basis: str = "orthogonal",
    n_splits: int = 5,
    criterion: str = "kfold_rmse",
    rng: np.random.Generator | int | None = None,
) -> OrderSweepResult:
    """Fit polynomials of several degrees and compare them in and out of sample.

    Parameters
    ----------
    x, y : array_like, shape (n_samples,)
        Predictor and response.
    degrees : sequence of int, optional
        Degrees to fit; ``None`` means 1 to 8.
    basis : {"orthogonal", "centered", "raw"}, optional
        Polynomial basis, see :mod:`voice_severity_bench.design_matrix`.
    n_splits : int, optional
        Folds of the k-fold estimate (reduced to the sample size if needed).
    criterion : {"kfold_rmse", "loocv_rmse", "bic", "aic"}, optional
        Column minimized by the recommendation.
    rng : numpy.random.Generator or int, optional
        Source of the fold shuffle.

    Returns
    -------
    OrderSweepResult
        The per-degree table and the recommended degree.

    References
    ----------
    .. [1] M. Stone, "Cross-validatory choice and assessment of statistical
       predictions," Journal of the Royal Statistical Society, Series B,
       vol. 36, no. 2, pp. 111-147, 1974.
    .. [2] G. Schwarz, "Estimating the dimension of a model," The Annals of
       Statistics, vol. 6, no. 2, pp. 461-464, 1978.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = np.linspace(-2.0, 2.0, 60)
    >>> y = 1.0 - x + 0.5 * x**3 + rng.normal(0, 0.5, x.size)
    >>> sweep = sweep_polynomial_order(x, y, range(1, 7), rng=0)
    >>> sweep.recommended_degree, bool(sweep.table["r2"].is_monotonic_increasing)
    (3, True)

    Seven points cannot support seven parameters:

    >>> tiny = sweep_polynomial_order(np.arange(7.0), np.arange(7.0) ** 2, [2, 6], rng=0)
    >>> tiny.table[["degree", "df_resid", "estimable"]].values.tolist()
    [[2, 4, True], [6, 0, False]]
    >>> bool(np.isnan(tiny.table.loc[1, "r2"]))
    True
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    response = np.asarray(y, dtype=np.float64).ravel()
    if values.size != response.size:
        raise ValueError(f"x and y differ in length: {values.size} and {response.size}")
    if values.size < 3:
        raise ValueError(f"need at least 3 observations, got {values.size}")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(response))):
        raise ValueError("x and y must not contain NaN or infinite values")
    if criterion not in ORDER_CRITERIA:
        raise ValueError(
            f"unknown order criterion {criterion!r}; choose one of {list(ORDER_CRITERIA)}"
        )
    requested = list(range(1, 9)) if degrees is None else [int(d) for d in degrees]
    chosen = sorted(set(requested))
    if not chosen or chosen[0] < 1:
        raise ValueError(f"degrees must be positive integers, got {requested!r}")
    generator = np.random.default_rng(rng)
    random_state = int(generator.integers(2**31 - 1))
    folds = max(2, min(int(n_splits), values.size))
    rows = [_degree_row(values, response, degree, basis, folds, random_state) for degree in chosen]
    table = pd.DataFrame(rows, columns=list(_COLUMNS))
    table["estimable"] = table["estimable"].astype(bool)
    for column in ("degree", "n_params", "df_resid"):
        table[column] = table[column].astype(int)
    scores = pd.to_numeric(table[criterion], errors="coerce")
    usable = scores[np.isfinite(scores)]
    recommended = None if usable.empty else int(table.loc[usable.idxmin(), "degree"])
    if recommended is None:
        _LOGGER.warning("no degree could be scored by %s", criterion)
    return OrderSweepResult(table=table, criterion=criterion, recommended_degree=recommended)
