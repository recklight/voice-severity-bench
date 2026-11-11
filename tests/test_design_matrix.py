"""Design matrices: orthogonal polynomial recurrence, held-out transforms and response surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from voice_severity_bench import PolynomialBasis, polynomial_features, quadratic_surface


def test_orthogonal_basis_is_orthonormal_and_spans_the_raw_powers(rng):
    x = rng.uniform(1990.0, 2020.0, 30)
    ortho, names = polynomial_features(x, 4)
    assert names == ("x^1", "x^2", "x^3", "x^4")
    assert np.allclose(ortho.T @ ortho, np.eye(4), atol=1e-10)
    centered, _ = polynomial_features(x, 4, basis="centered")
    full_ortho = np.column_stack([np.ones(30), ortho])
    full_centered = np.column_stack([np.ones(30), centered])
    projection = full_ortho @ np.linalg.lstsq(full_ortho, full_centered, rcond=None)[0]
    assert np.allclose(projection, full_centered, atol=1e-8)


def test_held_out_values_use_the_training_recurrence(rng):
    train = rng.uniform(-1, 1, 25)
    test = rng.uniform(-1, 1, 5)
    basis = PolynomialBasis(3).fit(train)
    together = PolynomialBasis(3).fit(train).transform(np.concatenate([train, test]))
    assert np.allclose(basis.transform(test), together[-5:])
    assert np.allclose(basis.transform(train), together[:25])


def test_raw_and_centered_bases():
    x = np.array([1.0, 2.0, 3.0])
    raw, _ = polynomial_features(x, 2, basis="raw")
    assert raw.tolist() == [[1.0, 1.0], [2.0, 4.0], [3.0, 9.0]]
    centered, _ = polynomial_features(x, 2, basis="centered")
    z = (x - 2.0) / np.std(x)
    assert np.allclose(centered, np.column_stack([z, z**2]))


def test_basis_validation():
    with pytest.raises(ValueError, match="distinct"):
        PolynomialBasis(3).fit([1.0, 1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="basis"):
        PolynomialBasis(2, "legendre")
    with pytest.raises(ValueError, match="fit"):
        PolynomialBasis(2).transform([1.0])


def test_quadratic_surface_terms_and_centering(rng):
    X = rng.uniform(0, 5, size=(12, 3))
    design, names = quadratic_surface(X, ["a", "b", "c"])
    assert names == ("a", "b", "c", "a^2", "b^2", "c^2", "a*b", "a*c", "b*c")
    centered = X - X.mean(axis=0)
    assert np.allclose(design[:, 6], centered[:, 0] * centered[:, 1])
    plain, _ = quadratic_surface(X, interactions=False, center=False)
    assert plain.shape == (12, 6) and np.allclose(plain[:, 3], X[:, 0] ** 2)
    shifted, _ = quadratic_surface(X[:2], ["a", "b", "c"], means=X.mean(axis=0))
    assert np.allclose(shifted, design[:2])
