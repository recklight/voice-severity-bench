"""Rater agreement: ICC against the textbook example, the noise floor, leave one rater out."""

from __future__ import annotations

import numpy as np
import pytest

from voice_severity_bench import regression_metrics
from voice_severity_bench.reference_data import shrout_fleiss_ratings
from voice_severity_bench.reliability import icc_oneway, leave_one_rater_out, rating_noise_floor


def test_shrout_fleiss_example():
    result = icc_oneway(shrout_fleiss_ratings())
    assert (round(result.icc1, 2), round(result.icck, 2)) == (0.17, 0.44)
    assert result.k0 == 4.0 and result.n_items == 6
    assert result.icc1_low < result.icc1 < result.icc1_high
    assert result.icck_low < result.icck < result.icck_high


def test_icc_matches_the_variance_components_of_simulated_raters(rng):
    # True-score sd 20, rater noise sd 10: ICC(1,1) = 400 / 500, ICC(1,3) = 400 / (400 + 100 / 3).
    true = rng.normal(50, 20, 4000)
    ratings = true[:, None] + rng.normal(0, 10, (4000, 3))
    result = icc_oneway(ratings)
    assert result.icc1 == pytest.approx(0.8, abs=0.015)
    assert result.icck == pytest.approx(400 / (400 + 100 / 3), abs=0.01)
    assert rating_noise_floor(ratings) == pytest.approx(10 / np.sqrt(3), rel=0.03)


def test_unbalanced_items_use_k0_and_single_ratings_are_left_out(rng):
    true = rng.normal(50, 20, 300)
    ratings = true[:, None] + rng.normal(0, 10, (300, 4))
    ratings[::5, 3] = np.nan  # a fourth rater for 240 of the items
    ratings[0, 1:] = np.nan  # one item with a single rating
    result = icc_oneway(ratings)
    sizes = np.r_[np.full(59, 3.0), np.full(240, 4.0)]
    expected_k0 = (sizes.sum() - np.sum(sizes**2) / sizes.sum()) / (sizes.size - 1)
    assert result.n_items == 299 and result.k0 == pytest.approx(expected_k0)


def test_one_rater_against_the_others(rng):
    true = rng.normal(50, 20, 500)
    ratings = true[:, None] + rng.normal(0, 10, (500, 4))
    pairs = leave_one_rater_out(ratings)
    assert len(pairs) == 2000
    # The error of one rating against the mean of three others: sd 10 * sqrt(1 + 1/3).
    scores = regression_metrics(pairs["reference"], pairs["rating"])
    assert scores["rmse"] == pytest.approx(10 * np.sqrt(4 / 3), rel=0.05)


def test_too_few_ratings_are_rejected():
    with pytest.raises(ValueError, match="two or more"):
        rating_noise_floor([[1.0, np.nan], [2.0, np.nan]])
    with pytest.raises(ValueError, match="at least two items"):
        icc_oneway([[1.0, 2.0]])
    with pytest.raises(ValueError, match="matrix"):
        icc_oneway([1.0, 2.0])
