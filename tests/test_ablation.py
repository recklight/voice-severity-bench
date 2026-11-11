"""Ablation study: grid expansion, shared folds, paired tests and the frame cache."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from voice_severity_bench import (
    Config,
    ConfigError,
    compare_to_baseline,
    corrected_resampled_ttest,
    expand_grid,
    run_ablation,
    wilcoxon_signed_rank,
)


def test_grid_is_the_cartesian_product_with_the_baseline_first():
    config = Config().override(
        {
            "ablation.n_ceps": [12, 6],
            "ablation.delta_order": [0, 2],
            "ablation.statistics": [["mean", "std"], ["mean"]],
            "ablation.models": ["ridge"],
        }
    )
    points = expand_grid(config)
    assert len(points) == 8
    assert points[0].config_id == "n_ceps=12|delta_order=0|statistics=mean+std"
    assert points[0].config.frontend.n_ceps == 12 and points[0].config.frontend.delta_order == 0
    assert dict(points[-1].settings) == {
        "n_ceps": "6",
        "delta_order": "2",
        "statistics": "mean",
        "model": "ridge",
    }
    assert points[-1].n_features == 7 * 3
    assert len({point.config_id for point in points}) == 8


def test_frame_and_hop_pairs_and_invalid_combinations():
    config = Config().override(
        {
            "ablation.frame_hop_ms": [[30, 15], [25, 10]],
            "ablation.n_ceps": [],
            "ablation.delta_order": [],
            "ablation.models": ["ridge"],
        }
    )
    points = expand_grid(config)
    assert [p.config_id for p in points] == ["frame_hop_ms=30/15", "frame_hop_ms=25/10"]
    assert points[1].config.frontend.frame_ms == 25.0 and points[1].config.frontend.hop_ms == 10.0
    with pytest.raises(ConfigError, match=r"ablation\.n_mels"):
        expand_grid(Config().override({"ablation.n_mels": [10], "ablation.n_ceps": [12]}))


@pytest.mark.parametrize(
    ("updates", "fragment"),
    [
        ({"ablation.n_ceps": [12, 30]}, "ablation.n_ceps"),
        ({"ablation.models": ["svc"]}, "ablation.models"),
        ({"ablation.reducer": ["lda"]}, "ablation.reducer"),
        ({"ablation.metric": "uar"}, "ablation.metric"),
        (
            {
                "pooling.normalization": "cmn",
                "pooling.statistics": ["std"],
                "ablation.statistics": [["mean"]],
            },
            "ablation.statistics",
        ),
    ],
)
def test_a_grid_that_does_not_fit_the_other_sections_is_rejected_before_any_work(updates, fragment):
    config = Config().override(updates)  # the configuration itself stays usable
    with pytest.raises(ConfigError, match=fragment):
        expand_grid(config)


def test_corrected_t_test_is_more_conservative_than_the_naive_one(rng):
    differences = rng.normal(0.5, 1.0, 15)
    t, p = corrected_resampled_ttest(differences, 0.25)
    naive = stats.ttest_1samp(differences, 0.0)
    assert abs(t) < abs(naive.statistic) and p > naive.pvalue
    expected = differences.mean() / math.sqrt((1 / 15 + 0.25) * differences.var(ddof=1))
    assert t == pytest.approx(expected)
    assert corrected_resampled_ttest(np.zeros(5), 0.25) == (0.0, 1.0)
    assert wilcoxon_signed_rank(np.zeros(4)) == 1.0


def test_comparison_pairs_folds_and_adjusts_p_values():
    folds = pd.DataFrame(
        {
            "config_id": ["base"] * 6 + ["worse"] * 6 + ["same"] * 6,
            "repeat": [0, 0, 0, 1, 1, 1] * 3,
            "fold": [0, 1, 2] * 6,
            "n_train": 20,
            "n_test": 10,
            "rmse": [
                *[5.0, 6.0, 7.0, 5.5, 6.5, 7.5],
                *[6.0, 7.2, 8.1, 6.4, 7.6, 8.3],
                *[5.0, 6.0, 7.0, 5.5, 6.5, 7.5],
            ],
        }
    )
    table = compare_to_baseline(folds, metric="rmse", baseline="base")
    assert table["config_id"].tolist() == ["worse", "same"]
    worse = table.iloc[0]
    assert worse["mean_difference"] == pytest.approx(6.1 / 6) and not worse["better"]
    assert worse["n_pairs"] == 6
    assert table.iloc[1]["p_corrected_t"] == 1.0
    assert (table["p_corrected_t_adjusted"] >= table["p_corrected_t"]).all()
    with pytest.raises(ValueError, match="baseline"):
        compare_to_baseline(folds, metric="rmse", baseline="missing")


def test_run_ablation_shares_folds_and_uses_the_cache(tmp_path, synthetic_dataset, fast_config):
    cache = tmp_path / "cache"
    result = run_ablation(synthetic_dataset, fast_config, cache_dir=cache)
    folds = result.folds
    assert set(folds["config_id"]) == {"delta_order=0", "delta_order=2"}
    per_setting = folds.groupby("config_id")[["repeat", "fold", "n_test"]]
    layouts = [part.reset_index(drop=True) for _, part in per_setting]
    assert layouts[0].equals(layouts[1])
    assert len(folds) == 2 * 2 * 4
    assert result.summary["rank"].tolist() == [1, 2]
    assert result.baseline == "delta_order=0" and result.metric == "rmse"
    assert len(result.comparisons) == 1
    assert len(list(cache.glob("frames_*.npz"))) == 1  # both settings share one front end
    again = run_ablation(synthetic_dataset, fast_config, cache_dir=cache)
    assert again.folds.equals(folds)
    assert result.as_dict()["n_settings"] == 2


def test_classification_ablation_ranks_by_uar(synthetic_dataset, classification_config):
    config = classification_config.override({"ablation.models": ["logistic", "svc"]})
    result = run_ablation(synthetic_dataset, config)
    assert result.metric == "uar"
    assert {"uar", "roc_auc", "sensitivity"} <= set(result.folds.columns)
    assert result.summary["uar_mean"].is_monotonic_decreasing


def test_ablation_reads_the_intervals_of_the_index(synthetic_dataset, fast_config):
    index = pd.read_csv(synthetic_dataset)
    whole = run_ablation(synthetic_dataset, fast_config)
    index.assign(start_s=0.0, end_s=0.25).to_csv(synthetic_dataset, index=False)
    halves = run_ablation(synthetic_dataset, fast_config)
    assert len(halves.folds) == len(whole.folds)
    # Same folds, different audio: the scores must change.
    assert not np.allclose(halves.folds["rmse"], whole.folds["rmse"])
