"""Stepwise variable selection with partial F tests.

Starting from an empty (forward) or full (backward) model, every step first
tries to *add* the candidate whose partial F test

.. math::

   F = \\frac{\\mathrm{RSS}_{reduced} - \\mathrm{RSS}_{full}}
             {\\mathrm{RSS}_{full} / (n - p_{full})}

has the smallest p-value below ``p_enter``; if none qualifies it tries to
*remove* the term with the largest p-value above ``p_remove``; it stops when
neither move is possible (Efroymson, 1960). Requiring ``p_enter <= p_remove``
prevents a term from being added and removed forever, and a visited-model
check stops any remaining cycle.

The procedure is deterministic and records every step. Its p-values are
*not* valid inference for the final model, because they were used to choose
it; check the selected model for collinearity
(:func:`~voice_severity_bench.regression_diagnostics.vif`) and validate it
out of sample.
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

from .ols import OLSResult, fit_ols

_LOGGER = logging.getLogger(__name__)

#: Starting models accepted by ``analysis.stepwise_start``.
STEPWISE_STARTS: tuple[str, ...] = ("empty", "full")


@dataclass(frozen=True, eq=False)
class StepwiseResult:
    """Every step of a stepwise selection and the final model.

    Attributes
    ----------
    steps : pandas.DataFrame
        Columns ``step``, ``action`` (``start``, ``enter`` or ``remove``),
        ``term``, ``f_stat``, ``p_value``, ``n_terms``, ``r2`` and ``adj_r2``
        (the last three describe the model *after* the step).
    selected : tuple of str
        Terms of the final model, in predictor order.
    final : OLSResult
        Least-squares fit of the final model.

    Examples
    --------
    >>> from voice_severity_bench.reference_data import cement_hardening_heat
    >>> data = cement_hardening_heat()
    >>> names = ["x1", "x2", "x3", "x4"]
    >>> result = stepwise_select(data[names], data["heat"], names)
    >>> result.as_dict(), result.steps["action"].tolist()
    ({'selected': ['x1', 'x4'], 'n_steps': 2}, ['start', 'enter', 'enter'])
    """

    steps: pd.DataFrame
    selected: tuple[str, ...]
    final: OLSResult

    def as_dict(self) -> dict[str, object]:
        """Selection summary for JSON output."""
        return {"selected": list(self.selected), "n_steps": int(len(self.steps) - 1)}


def _rss(design_columns: NDArray[np.float64], response: NDArray[np.float64]) -> float:
    matrix = np.column_stack([np.ones(response.size), design_columns])
    coef, *_ = np.linalg.lstsq(matrix, response, rcond=None)
    residual = response - matrix @ coef
    return float(residual @ residual)


def partial_f_test(
    rss_reduced: float, rss_full: float, df_full: int, n_restrictions: int = 1
) -> tuple[float, float]:
    """Partial F statistic and p-value for dropping ``n_restrictions`` terms.

    Parameters
    ----------
    rss_reduced, rss_full : float
        Residual sums of squares without and with the tested terms.
    df_full : int
        Residual degrees of freedom of the larger model.
    n_restrictions : int, optional
        Number of tested terms.

    Returns
    -------
    tuple of float
        ``(F, p)``; ``nan`` when ``df_full < 1``.

    Examples
    --------
    >>> f, p = partial_f_test(rss_reduced=120.0, rss_full=100.0, df_full=20)
    >>> round(f, 3), round(p, 4)
    (4.0, 0.0593)
    """
    if df_full < 1:
        return math.nan, math.nan
    if rss_full <= 0.0:
        return math.inf, 0.0
    statistic = ((rss_reduced - rss_full) / n_restrictions) / (rss_full / df_full)
    statistic = max(statistic, 0.0)
    return float(statistic), float(stats.f.sf(statistic, n_restrictions, df_full))


def stepwise_select(
    X: ArrayLike,
    y: ArrayLike,
    names: Sequence[str] | None = None,
    *,
    start: str = "empty",
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    max_steps: int | None = None,
    ci_level: float = 0.95,
) -> StepwiseResult:
    """Select predictors by stepwise partial F tests.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_predictors)
        Candidate predictors (no intercept column; one is always included).
    y : array_like, shape (n_samples,)
        Response.
    names : sequence of str, optional
        Predictor names (default ``x1, x2, ...``).
    start : {"empty", "full"}, optional
        Initial model.
    p_enter, p_remove : float, optional
        Entry and removal thresholds, ``p_enter <= p_remove``.
    max_steps : int, optional
        Safety limit on the number of moves (default ``2 * n_predictors + 10``).
    ci_level : float, optional
        Confidence level of the final model's intervals.

    Returns
    -------
    StepwiseResult
        The recorded steps, the selected terms and the final fit.

    References
    ----------
    .. [1] M. A. Efroymson, "Multiple regression analysis," in Mathematical
       Methods for Digital Computers, A. Ralston and H. S. Wilf, Eds. New York,
       NY, USA: Wiley, 1960, pp. 191-203.

    Examples
    --------
    On the cement hardening data the starting model changes the answer:

    >>> from voice_severity_bench.reference_data import cement_hardening_heat
    >>> data = cement_hardening_heat()
    >>> X, y = data[["x1", "x2", "x3", "x4"]], data["heat"]
    >>> stepwise_select(X, y, X.columns, start="empty").selected
    ('x1', 'x4')
    >>> stepwise_select(X, y, X.columns, start="full").selected
    ('x1', 'x2')
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
    labels = (
        tuple(str(name) for name in names)
        if names is not None
        else tuple(f"x{i + 1}" for i in range(matrix.shape[1]))
    )
    if len(labels) != matrix.shape[1]:
        raise ValueError(f"{len(labels)} names for {matrix.shape[1]} predictor columns")
    if len(set(labels)) != len(labels):
        raise ValueError(f"predictor names must be unique, got {list(labels)!r}")
    if start not in STEPWISE_STARTS:
        raise ValueError(f"unknown start {start!r}; choose one of {list(STEPWISE_STARTS)}")
    if not 0.0 < p_enter <= p_remove < 1.0:
        raise ValueError(
            f"need 0 < p_enter <= p_remove < 1, got p_enter={p_enter!r}, p_remove={p_remove!r}"
        )
    n, n_predictors = matrix.shape
    if start == "full" and n_predictors + 1 >= n:
        raise ValueError(
            f"the full model has {n_predictors + 1} parameters for {n} observations; "
            "start from the empty model instead"
        )
    limit = 2 * n_predictors + 10 if max_steps is None else int(max_steps)
    total = float(np.sum((response - response.mean()) ** 2))

    def describe(model: list[int]) -> tuple[float, float, float]:
        rss = _rss(matrix[:, model], response) if model else total
        df = n - len(model) - 1
        r2 = 1.0 - rss / total if total > 0.0 else math.nan
        adj = 1.0 - (1.0 - r2) * (n - 1) / df if df > 0 else math.nan
        return rss, r2, adj

    model = list(range(n_predictors)) if start == "full" else []
    rss, r2, adj = describe(model)
    rows = [
        {
            "step": 0,
            "action": "start",
            "term": "",
            "f_stat": math.nan,
            "p_value": math.nan,
            "n_terms": len(model),
            "r2": r2,
            "adj_r2": adj,
        }
    ]
    visited = {frozenset(model)}
    for step in range(1, limit + 1):
        move = None
        candidates = [j for j in range(n_predictors) if j not in model]
        df_add = n - len(model) - 2
        entries = []
        for j in candidates:
            rss_new = _rss(matrix[:, [*model, j]], response)
            f_stat, p_value = partial_f_test(rss, rss_new, df_add)
            if np.isfinite(p_value):
                entries.append((p_value, j, f_stat))
        if entries:
            p_value, j, f_stat = min(entries)
            if p_value < p_enter:
                move = ("enter", j, f_stat, p_value, sorted([*model, j]))
        if move is None and model:
            df_full = n - len(model) - 1
            removals = []
            for j in model:
                reduced = [k for k in model if k != j]
                f_stat, p_value = partial_f_test(_rss(matrix[:, reduced], response), rss, df_full)
                removals.append((p_value, j, f_stat))
            p_value, j, f_stat = max(removals)
            if p_value > p_remove:
                move = ("remove", j, f_stat, p_value, [k for k in model if k != j])
        if move is None:
            break
        action, j, f_stat, p_value, new_model = move
        if frozenset(new_model) in visited:
            _LOGGER.warning("stepwise selection revisits a model; stopping to avoid a cycle")
            break
        model = new_model
        visited.add(frozenset(model))
        rss, r2, adj = describe(model)
        _LOGGER.debug("step %d: %s %s (p=%.4g)", step, action, labels[j], p_value)
        rows.append(
            {
                "step": step,
                "action": action,
                "term": labels[j],
                "f_stat": f_stat,
                "p_value": p_value,
                "n_terms": len(model),
                "r2": r2,
                "adj_r2": adj,
            }
        )
    else:
        _LOGGER.warning("stepwise selection stopped after max_steps=%d moves", limit)
    selected = tuple(labels[j] for j in sorted(model))
    final = fit_ols(matrix[:, sorted(model)], response, selected, ci_level=ci_level)
    return StepwiseResult(steps=pd.DataFrame(rows), selected=selected, final=final)
