"""Association and agreement: Fisher intervals, CCC, Bland-Altman and p-value adjustment."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from voice_severity_bench import (
    adjust_pvalues,
    bland_altman,
    concordance_ccc,
    feature_target_correlations,
    pearson_ci,
    spearman,
)


def test_pearson_interval_follows_the_fisher_transform(rng):
    x = rng.normal(size=50)
    y = 0.6 * x + rng.normal(size=50)
    result = pearson_ci(x, y, 0.9)
    r = stats.pearsonr(x, y)[0]
    half = stats.norm.ppf(0.95) / math.sqrt(47)
    assert result.coefficient == pytest.approx(r)
    assert result.ci_low == pytest.approx(math.tanh(math.atanh(r) - half))
    assert result.ci_high == pytest.approx(math.tanh(math.atanh(r) + half))


def test_interval_coverage_is_close_to_nominal():
    rng = np.random.default_rng(7)
    rho = 0.5
    cov = [[1.0, rho], [rho, 1.0]]
    covered = 0
    for _ in range(1000):
        sample = rng.multivariate_normal([0, 0], cov, 40)
        result = pearson_ci(sample[:, 0], sample[:, 1])
        covered += int(result.ci_low <= rho <= result.ci_high)
    assert 0.92 < covered / 1000 < 0.97


def test_spearman_and_constant_samples(rng):
    x = rng.normal(size=30)
    rho, p = spearman(x, np.exp(x))
    assert rho == pytest.approx(1.0) and p < 1e-6
    assert all(math.isnan(v) for v in spearman(x, np.ones(30)))
    assert math.isnan(pearson_ci(x, np.ones(30)).coefficient)


def test_ccc_is_one_only_for_identity_and_penalizes_bias(rng):
    truth = rng.uniform(0, 100, 40)
    assert concordance_ccc(truth, truth) == pytest.approx(1.0)
    shifted = concordance_ccc(truth, truth + 15.0)
    scaled = concordance_ccc(truth, 0.5 * truth + 25.0)
    assert stats.pearsonr(truth, truth + 15.0)[0] == pytest.approx(1.0)
    assert shifted < 1.0 and scaled < 1.0
    expected = 2 * np.cov(truth, truth + 15.0, bias=True)[0, 1] / (2 * truth.var() + 15.0**2)
    assert shifted == pytest.approx(expected)
    assert concordance_ccc(truth, -truth) < 0


def test_bland_altman_limits(rng):
    a = rng.normal(50, 10, 200)
    b = a + rng.normal(2.0, 3.0, 200)
    result = bland_altman(b, a)
    differences = b - a
    assert result.bias == pytest.approx(differences.mean())
    assert result.upper_loa - result.bias == pytest.approx(
        1.959964 * differences.std(ddof=1), rel=1e-5
    )
    inside = np.mean((differences >= result.lower_loa) & (differences <= result.upper_loa))
    assert 0.9 < inside <= 1.0
    assert set(result.as_dict()) == {"bias", "sd", "lower_loa", "upper_loa", "ci_level"}


def test_holm_and_benjamini_hochberg_by_hand():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.20])
    assert np.allclose(adjust_pvalues(p, "holm"), [0.05, 0.08, 0.09, 0.09, 0.20])
    assert np.allclose(adjust_pvalues(p, "fdr_bh"), [0.05, 0.05, 0.05, 0.05, 0.20])
    with_nan = adjust_pvalues([0.01, np.nan, 0.04], "holm")
    assert np.isnan(with_nan[1]) and np.allclose(with_nan[[0, 2]], [0.02, 0.04])
    with pytest.raises(ValueError, match="correction"):
        adjust_pvalues(p, "bonferroni")


def test_feature_table_is_sorted_and_adjusted(rng):
    severity = rng.uniform(0, 100, 60)
    X = np.column_stack(
        [rng.normal(size=60), severity + rng.normal(0, 30, 60), np.ones(60), -severity]
    )
    table = feature_target_correlations(X, severity, ["noise", "weak", "constant", "mirror"])
    assert table["feature"].tolist() == ["mirror", "weak", "noise", "constant"]
    assert table["pearson_r"].iloc[0] == pytest.approx(-1.0)
    assert (table["pearson_p_adjusted"].dropna() >= table["pearson_p"].dropna()).all()
    assert math.isnan(table["pearson_r"].iloc[-1])
