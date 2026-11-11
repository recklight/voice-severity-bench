"""Synthetic data: reproducibility and the structure it is designed to have."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from speechdsp import read_wav

from voice_severity_bench import (
    SyntheticConfig,
    make_collinear_data,
    make_polynomial_data,
    make_quadratic_surface_data,
    make_vowel,
    write_synthetic_dataset,
)
from voice_severity_bench.synthetic import (
    glottal_source,
    rosenberg_pulse,
    severity_parameters,
)

SMALL = SyntheticConfig(n_speakers=4, n_recordings_per_speaker=2, duration_s=0.3)


def test_same_seed_writes_identical_bytes(tmp_path):
    first = write_synthetic_dataset(tmp_path / "a", SMALL, rng=7)
    second = write_synthetic_dataset(tmp_path / "b", SMALL, rng=7)
    assert first.read_bytes() == second.read_bytes()
    for name in ("item_000.wav", "item_007.wav"):
        assert (tmp_path / "a" / "items" / name).read_bytes() == (
            tmp_path / "b" / "items" / name
        ).read_bytes()


def test_different_seeds_differ(tmp_path):
    first = write_synthetic_dataset(tmp_path / "a", SMALL, rng=1)
    second = write_synthetic_dataset(tmp_path / "b", SMALL, rng=2)
    assert first.read_bytes() != second.read_bytes()


def test_index_layout_labels_and_groups(tmp_path):
    index = pd.read_csv(write_synthetic_dataset(tmp_path, SMALL, rng=0))
    assert index.columns.tolist() == ["path", "severity", "label", "group"]
    assert len(index) == SMALL.n_recordings
    assert index["group"].value_counts().tolist() == [2, 2, 2, 2]
    assert index["severity"].between(0.0, 100.0).all()
    expected = np.where(index["severity"] >= SMALL.label_threshold, "dysphonic", "healthy")
    assert (index["label"] == expected).all()
    signal, rate = read_wav(tmp_path / index["path"].iloc[0])
    assert rate == SMALL.sample_rate and signal.size == round(SMALL.duration_s * rate)


def test_perturbations_grow_with_severity():
    low, high = severity_parameters(0.0, SMALL), severity_parameters(100.0, SMALL)
    assert low["jitter"] < high["jitter"] and low["shimmer"] < high["shimmer"]
    assert low["hnr_db"] > high["hnr_db"]


def test_jitter_spreads_the_cycle_lengths():
    rate = 16_000

    def cycle_lengths(jitter):
        flow = glottal_source(rate, rate, 125.0, jitter=jitter, shimmer=0.0, rng=3)
        onsets = np.flatnonzero((flow[1:] > 0) & (flow[:-1] == 0))
        return np.diff(onsets)

    assert np.std(cycle_lengths(0.0)) < 1.0
    assert np.std(cycle_lengths(0.05)) > 3.0
    assert np.mean(cycle_lengths(0.0)) == pytest.approx(rate / 125.0, abs=1.0)


def test_rosenberg_pulse_shape():
    phases = np.linspace(0.0, 0.999, 1000)
    pulse = rosenberg_pulse(phases)
    assert pulse.min() >= 0.0 and pulse.max() == pytest.approx(1.0, abs=1e-3)
    assert np.all(pulse[phases >= 0.56] == 0.0)


def test_noise_lowers_the_harmonic_peak_prominence():
    config = SyntheticConfig(duration_s=0.5, jitter_range=(0.0, 0.0), shimmer_range=(0.0, 0.0))
    spectra = []
    for severity in (0.0, 100.0):
        signal = make_vowel(severity, config=config, f0_hz=125.0, rng=4)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
        spectra.append(spectrum)
    harmonic = round(125.0 * 0.5)  # bin of the fundamental at 2 Hz resolution
    between = harmonic + harmonic // 2
    contrast = [s[harmonic] / s[between] for s in spectra]
    assert contrast[0] > 10.0 * contrast[1]


def test_regression_toys_follow_their_generating_models():
    surface = make_quadratic_surface_data(200, noise_sd=0.0, rng=0)
    x1, x2 = surface["x1"], surface["x2"]
    expected = 5.0 + 2.0 * x1 - 1.5 * x2 - 0.8 * x1**2 + 0.3 * x2**2 + 0.6 * x1 * x2
    assert np.allclose(surface["y"], expected)
    curve = make_polynomial_data(9, noise_sd=0.0)
    assert np.allclose(curve["y"], 1.0 - 2.0 * curve["x"] + 0.5 * curve["x"] ** 3)
    mixture = make_collinear_data(50, rng=0)
    totals = mixture[["p1", "p2", "p3", "p4"]].sum(axis=1)
    assert totals.between(98.0, 100.0).all()
