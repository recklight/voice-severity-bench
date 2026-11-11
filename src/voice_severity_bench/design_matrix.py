"""Design matrices for polynomial and second-order response-surface regression.

Raw powers :math:`x, x^2, \\dots, x^6` of an uncentered predictor are nearly
collinear: for :math:`x \\approx 2000` the columns differ by factors of
:math:`10^{3}` and the condition number of the design explodes, so
coefficients and their standard errors become numerically meaningless.
Besides ``raw``, two bases avoid this:

* ``centered``: powers of the standardized predictor :math:`z = (x - \\bar x)/s_x`;
* ``orthogonal``: discrete orthogonal polynomials on the observed :math:`x`
  built by the three-term (Stieltjes) recurrence

  .. math:: p_{k+1}(x) = (x - a_k)\\, p_k(x) - b_k\\, p_{k-1}(x), \\quad
            a_k = \\frac{\\sum x p_k^2}{\\sum p_k^2}, \\quad
            b_k = \\frac{\\sum p_k^2}{\\sum p_{k-1}^2},

  whose columns are mutually orthogonal (and scaled to unit norm), so adding
  a degree never changes the coefficients of the lower ones.

The fitted basis is stored, so held-out data are transformed with the
training recurrence, which keeps cross-validation honest. The second-order
response surface (Box & Wilson, 1951)

.. math:: y = \\beta_0 + \\sum_i \\beta_i x_i + \\sum_i \\beta_{ii} x_i^2
          + \\sum_{i<j} \\beta_{ij} x_i x_j

is built from centered predictors for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Polynomial bases accepted by ``analysis.polynomial_basis``.
POLYNOMIAL_BASES: tuple[str, ...] = ("orthogonal", "centered", "raw")


class PolynomialBasis:
    """Polynomial features of one predictor, fitted on training data.

    Parameters
    ----------
    degree : int
        Highest power (number of output columns).
    basis : {"orthogonal", "centered", "raw"}, optional
        Construction of the columns, see the module documentation.

    Attributes
    ----------
    names : tuple of str
        Column names, e.g. ``("x^1", "x^2")``.

    Examples
    --------
    >>> x = np.linspace(1990.0, 2020.0, 7)
    >>> basis = PolynomialBasis(3, "orthogonal").fit(x)
    >>> Z = basis.transform(x)
    >>> bool(np.allclose(Z.T @ Z, np.eye(3))), bool(np.allclose(Z.sum(axis=0), 0.0))
    (True, True)
    """

    def __init__(self, degree: int, basis: str = "orthogonal") -> None:
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError(f"polynomial degree must be a positive integer, got {degree!r}")
        if basis not in POLYNOMIAL_BASES:
            raise ValueError(
                f"unknown polynomial basis {basis!r}; choose one of {list(POLYNOMIAL_BASES)}"
            )
        self.degree = degree
        self.basis = basis
        self.names = tuple(f"x^{power}" for power in range(1, degree + 1))
        self._center = 0.0
        self._scale = 1.0
        self._alphas: NDArray[np.float64] | None = None
        self._norms: NDArray[np.float64] | None = None
        self._fitted = False

    def fit(self, x: ArrayLike) -> PolynomialBasis:
        """Learn the centering (and the recurrence) from training values."""
        values = _vector(x)
        unique = np.unique(values).size
        if self.basis == "orthogonal" and unique <= self.degree:
            raise ValueError(
                f"an orthogonal basis of degree {self.degree} needs at least {self.degree + 1} "
                f"distinct x values, got {unique}"
            )
        self._center = float(values.mean())
        spread = float(values.std())
        self._scale = spread if spread > 0.0 else 1.0
        if self.basis == "orthogonal":
            self._fit_recurrence(values)
        self._fitted = True
        return self

    def _fit_recurrence(self, values: NDArray[np.float64]) -> None:
        x = (values - self._center) / self._scale
        alphas = np.zeros(self.degree)
        norms = np.zeros(self.degree + 1)
        previous = np.zeros_like(x)
        current = np.ones_like(x)
        norms[0] = float(current @ current)
        for k in range(self.degree):
            alphas[k] = float(x @ current**2) / norms[k]
            beta = norms[k] / norms[k - 1] if k > 0 else 0.0
            previous, current = current, (x - alphas[k]) * current - beta * previous
            norms[k + 1] = float(current @ current)
        self._alphas, self._norms = alphas, norms

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        """Columns for the powers ``1 .. degree`` of new values.

        Returns
        -------
        numpy.ndarray, shape (n_samples, degree)
            The design columns (no intercept).
        """
        if not self._fitted:
            raise ValueError("call fit() before transform()")
        values = _vector(x)
        if self.basis == "raw":
            return np.column_stack([values**power for power in range(1, self.degree + 1)])
        z = (values - self._center) / self._scale
        if self.basis == "centered":
            return np.column_stack([z**power for power in range(1, self.degree + 1)])
        assert self._alphas is not None and self._norms is not None
        columns = []
        previous, current = np.zeros_like(z), np.ones_like(z)
        for k in range(self.degree):
            beta = self._norms[k] / self._norms[k - 1] if k > 0 else 0.0
            previous, current = current, (z - self._alphas[k]) * current - beta * previous
            columns.append(current / np.sqrt(self._norms[k + 1]))
        return np.column_stack(columns)

    def fit_transform(self, x: ArrayLike) -> NDArray[np.float64]:
        """:meth:`fit` then :meth:`transform` on the same values."""
        return self.fit(x).transform(x)


def _vector(x: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected a non-empty 1-D predictor, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictor contains NaN or infinite values")
    return values


def polynomial_features(
    x: ArrayLike, degree: int, *, basis: str = "orthogonal"
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Polynomial design columns of one predictor (fit and transform in one step).

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Predictor values.
    degree : int
        Highest power.
    basis : {"orthogonal", "centered", "raw"}, optional
        Column construction.

    Returns
    -------
    matrix : numpy.ndarray, shape (n_samples, degree)
        Design columns without the intercept.
    names : tuple of str
        ``("x^1", ..., "x^degree")``.

    Examples
    --------
    The raw sixth-degree design of years is numerically singular; the
    orthogonal one is perfectly conditioned:

    >>> years = np.array([1950.0, 1960, 1970, 1980, 1990, 2000, 2010])
    >>> raw, _ = polynomial_features(years, 6, basis="raw")
    >>> ortho, names = polynomial_features(years, 6)
    >>> bool(np.linalg.cond(raw) > 1e15), round(float(np.linalg.cond(ortho)), 6), names[-1]
    (True, 1.0, 'x^6')
    """
    basis_object = PolynomialBasis(degree, basis)
    return basis_object.fit_transform(x), basis_object.names


def quadratic_surface(
    X: ArrayLike,
    names: Sequence[str] | None = None,
    *,
    interactions: bool = True,
    center: bool = True,
    means: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Full second-order response-surface design (no intercept column).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_predictors)
        Predictors.
    names : sequence of str, optional
        Predictor names (default ``x1, x2, ...``).
    interactions : bool, optional
        Include the products :math:`x_i x_j` for :math:`i < j`.
    center : bool, optional
        Subtract each predictor's mean before forming squares and products,
        which removes most of the collinearity between :math:`x` and :math:`x^2`.
    means : array_like, shape (n_predictors,), optional
        Centers to subtract instead of the column means of ``X``; pass the
        training means to evaluate a fitted surface on new points.

    Returns
    -------
    matrix : numpy.ndarray, shape (n_samples, n_terms)
        Linear terms, then squares, then interactions.
    names : tuple of str
        Term names such as ``x1``, ``x1^2``, ``x1*x2``.

    References
    ----------
    .. [1] G. E. P. Box and K. B. Wilson, "On the experimental attainment of
       optimum conditions," Journal of the Royal Statistical Society, Series B,
       vol. 13, no. 1, pp. 1-45, 1951.

    Examples
    --------
    >>> design, terms = quadratic_surface(np.arange(12.0).reshape(6, 2) ** 1.5, ["a", "b"])
    >>> design.shape, terms
    ((6, 5), ('a', 'b', 'a^2', 'b^2', 'a*b'))
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"predictors must be a non-empty 2-D array, got shape {matrix.shape}")
    labels = (
        tuple(names) if names is not None else tuple(f"x{i + 1}" for i in range(matrix.shape[1]))
    )
    if len(labels) != matrix.shape[1]:
        raise ValueError(f"{len(labels)} names for {matrix.shape[1]} predictor columns")
    if center:
        offsets = matrix.mean(axis=0) if means is None else np.asarray(means, dtype=np.float64)
        if offsets.shape != (matrix.shape[1],):
            raise ValueError(f"means must hold {matrix.shape[1]} values, got shape {offsets.shape}")
        base = matrix - offsets
    else:
        base = matrix
    columns = [base, base**2]
    terms = list(labels) + [f"{name}^2" for name in labels]
    if interactions:
        pairs = list(combinations(range(matrix.shape[1]), 2))
        if pairs:
            columns.append(np.column_stack([base[:, i] * base[:, j] for i, j in pairs]))
            terms.extend(f"{labels[i]}*{labels[j]}" for i, j in pairs)
    return np.hstack(columns), tuple(terms)
