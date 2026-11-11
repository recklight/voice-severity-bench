"""Linear least squares by batch gradient descent, checked against the closed form.

Gradient descent minimizes :math:`J(\\theta) = \\tfrac{1}{2m}\\lVert Z\\theta - y\\rVert^2`
by the simultaneous update (Cauchy, 1847)

.. math:: \\theta \\leftarrow \\theta - \\frac{\\alpha}{m} Z^\\top (Z\\theta - y),

where :math:`Z` is the design with a leading column of ones.

Scaling decides whether this works at all. With raw predictors such as
calendar years (:math:`x \\approx 2000`) the cost surface is a narrow valley:
any stable learning rate is tiny and progress stalls. By default the descent
therefore runs on standardized predictors (:math:`z = (x - \\bar x)/s_x`)
and maps the coefficients back to the original units at the end.

The iterations stop when the relative decrease of the cost falls below
``tol``, when ``max_iter`` is reached, or when the cost grows or overflows
(learning rate too large); the result records which. A converged run agrees
with the closed-form solution of :func:`~voice_severity_bench.ols.fit_ols`
to numerical precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class GradientDescentResult:
    """Outcome of :func:`fit_linear_gd`.

    Attributes
    ----------
    intercept : float
        Intercept in the original units (0 when not fitted).
    coef : numpy.ndarray, shape (n_predictors,)
        Slopes in the original units.
    cost_history : numpy.ndarray, shape (n_iter + 1,)
        Cost :math:`J` before the first update and after every update, on the
        scale the descent ran on.
    n_iter : int
        Number of updates performed.
    converged : bool
        ``True`` when the relative cost decrease fell below ``tol``.
    diverged : bool
        ``True`` when the cost increased or became non-finite.

    Examples
    --------
    >>> result = fit_linear_gd(np.arange(10.0), 3.0 + 2.0 * np.arange(10.0))
    >>> result.converged, [round(float(v), 6) for v in result.predict([0.0, 1.0])]
    (True, [3.0, 5.0])
    """

    intercept: float
    coef: NDArray[np.float64]
    cost_history: NDArray[np.float64]
    n_iter: int
    converged: bool
    diverged: bool

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predictions for new rows."""
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        return self.intercept + matrix @ self.coef


def fit_linear_gd(
    X: ArrayLike,
    y: ArrayLike,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 10_000,
    tol: float = 1e-14,
    standardize: bool = True,
    fit_intercept: bool = True,
) -> GradientDescentResult:
    """Fit a linear regression by batch gradient descent.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_predictors)
        Predictors; a 1-D array is one predictor.
    y : array_like, shape (n_samples,)
        Response.
    learning_rate : float, optional
        Step size :math:`\\alpha`.
    max_iter : int, optional
        Upper bound on the number of updates.
    tol : float, optional
        Stop when :math:`(J_{k-1} - J_k) / J_{k-1} < tol`.
    standardize : bool, optional
        Descend on standardized predictors and map the result back.
    fit_intercept : bool, optional
        Include an intercept.

    Returns
    -------
    GradientDescentResult
        Coefficients in the original units, the cost history and the stopping reason.

    References
    ----------
    .. [1] A. Cauchy, "Méthode générale pour la résolution des systèmes
       d'équations simultanées," Comptes Rendus de l'Académie des Sciences,
       vol. 25, pp. 536-538, 1847.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> y = 1.0 + X @ np.array([2.0, -3.0]) + rng.normal(0, 0.1, 100)
    >>> result = fit_linear_gd(X, y)
    >>> design = np.column_stack([np.ones(100), X])
    >>> closed_form = np.linalg.lstsq(design, y, rcond=None)[0]
    >>> estimate = [result.intercept, *result.coef]
    >>> result.converged, bool(np.allclose(estimate, closed_form, atol=1e-6))
    (True, True)
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    response = np.asarray(y, dtype=np.float64).ravel()
    if matrix.ndim != 2 or matrix.shape[0] != response.size or response.size == 0:
        raise ValueError(
            f"X must be (n_samples x n_predictors) with n_samples={response.size}, "
            f"got shape {matrix.shape}"
        )
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(response))):
        raise ValueError("X and y must not contain NaN or infinite values")
    if not learning_rate > 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")

    m = response.size
    center = np.zeros(matrix.shape[1])
    scale = np.ones(matrix.shape[1])
    if standardize:
        spread = matrix.std(axis=0) if fit_intercept else np.sqrt(np.mean(matrix**2, axis=0))
        scale = np.where(spread > 0.0, spread, 1.0)
        if fit_intercept:
            # Centering is only a reparametrization when an intercept absorbs it.
            center = matrix.mean(axis=0)
    scaled = (matrix - center) / scale
    design = np.column_stack([np.ones(m), scaled]) if fit_intercept else scaled
    theta = np.zeros(design.shape[1])

    def cost(parameters: NDArray[np.float64]) -> float:
        residual = design @ parameters - response
        return float(residual @ residual) / (2.0 * m)

    history = [cost(theta)]
    converged = diverged = False
    iterations = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for iterations in range(1, max_iter + 1):
            gradient = design.T @ (design @ theta - response) / m
            theta = theta - learning_rate * gradient
            current = cost(theta)
            history.append(current)
            previous = history[-2]
            if not np.isfinite(current) or current > previous * (1.0 + 1e-9):
                diverged = True
                _LOGGER.warning(
                    "gradient descent diverged after %d iterations; lower the learning rate",
                    iterations,
                )
                break
            if previous == 0.0 or previous - current <= tol * previous:
                converged = True
                break
    if not converged and not diverged:
        _LOGGER.warning("gradient descent stopped at max_iter=%d before converging", max_iter)

    slopes_scaled = theta[1:] if fit_intercept else theta
    coef = slopes_scaled / scale
    intercept = float(theta[0] - np.sum(slopes_scaled * center / scale)) if fit_intercept else 0.0
    return GradientDescentResult(
        intercept=intercept,
        coef=coef,
        cost_history=np.asarray(history),
        n_iter=iterations,
        converged=converged,
        diverged=diverged,
    )
