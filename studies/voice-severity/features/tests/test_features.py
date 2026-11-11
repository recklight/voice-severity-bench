"""Checks of the S4 extractors that need no ratings (study .venv: python -m pytest features/tests)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

from voice_severity_bench import SyntheticConfig, make_vowel  # noqa: E402


def test_indexes_use_valid_intervals_and_merged_groups():
    a = common.index("a")
    whole = common.index("whole")
    assert len(a) == common.plan()["data"]["expected_counts"]["primary"]
    assert len(whole) == common.plan()["data"]["expected_counts"]["all"]
    assert a["group"].nunique() == common.plan()["data"]["expected_counts"]["groups"]
    assert whole["start_s"].isna().all() and whole["end_s"].isna().all()
    assert (a["end_s"] > a["start_s"]).all()
    assert a.loc[a["file_id"] == "NYU1015", "path"].iloc[0].endswith("data/derived/NYU1015.wav")


def test_the_derived_channel_is_the_voiced_one():
    from scipy.io import wavfile

    rate, mono = wavfile.read(common.ensure_derived()["NYU1015"])
    assert mono.ndim == 1 and rate == 44100
    level = 20 * np.log10(np.sqrt(np.mean((mono / 32768.0) ** 2)))
    assert level > -40.0  # S0: the right channel sits at -27 dBFS, the left at -84


def test_preprocessing_matches_the_plan():
    row = common.index("a").iloc[0]
    signal, rate = common.load_interval(row)
    assert rate == 16000
    assert 20 * np.log10(np.sqrt(np.mean(signal**2))) == pytest.approx(-26.0, abs=1e-6)
    assert signal.size == pytest.approx((row["end_s"] - row["start_s"]) * 16000, abs=2)


def test_clinical_measures_follow_synthetic_severity():
    pytest.importorskip("parselmouth")
    import extract_clinical

    rate = 16000
    config = SyntheticConfig(duration_s=1.5, sample_rate=rate)
    mild = make_vowel(10.0, config=config, f0_hz=120.0, rng=1)
    severe = make_vowel(90.0, config=config, f0_hz=120.0, rng=1)
    a, _ = extract_clinical.measures(mild, rate)
    b, _ = extract_clinical.measures(severe, rate)
    assert 12 * np.log2(115 / 100) < a["f0_mean_st"] < 12 * np.log2(125 / 100)
    assert a["hnr_mean_db"] > b["hnr_mean_db"]
    assert a["jitter_local"] < b["jitter_local"] and a["shimmer_local"] < b["shimmer_local"]
    assert a["cpps_db"] > b["cpps_db"]


def test_silence_gives_nan_not_an_error():
    pytest.importorskip("parselmouth")
    import extract_clinical

    rng = np.random.default_rng(0)
    values, notes = extract_clinical.measures(1e-4 * rng.standard_normal(16000), 16000)
    assert all(np.isnan(v) for v in values.values()) and notes
