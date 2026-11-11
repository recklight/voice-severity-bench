"""Cepstral front end: layouts, scaling conventions, lifter, floors and sampling-rate handling."""

from __future__ import annotations

import math

import numpy as np
import pytest
import speechdsp
from scipy.fft import dct
from scipy.signal import get_window

from voice_severity_bench import (
    FrontendConfig,
    append_dynamics,
    cepstral_features,
    compact_preset,
    feature_layout,
    frame_features,
    htk_c0_preset,
    htk_dct,
    sinusoidal_lifter,
)
from voice_severity_bench.cepstral_frontend import analysis_window, frame_sizes

RATE = 16_000


def tone(seconds=0.5, frequency=220.0, rate=RATE):
    return np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate)


@pytest.mark.parametrize(("order", "width"), [(0, 13), (1, 26), (2, 39)])
def test_the_three_classic_layouts(order, width):
    config = FrontendConfig(delta_order=order)
    features = frame_features(tone(), RATE, config)
    assert features.shape[1] == width == config.n_dims == len(feature_layout(config))


def test_c0_uses_the_sqrt_2_over_n_scaling():
    rng = np.random.default_rng(0)
    energies = rng.normal(size=(4, 23))
    htk = htk_dct(energies, 13)
    manual = np.sqrt(2.0 / 23) * np.array(
        [
            [np.sum(row * np.cos(np.pi * m * (np.arange(1, 24) - 0.5) / 23)) for m in range(13)]
            for row in energies
        ]
    )
    assert np.allclose(htk, manual, rtol=1e-12)
    ortho = dct(energies, type=2, norm="ortho")[:, :13]
    assert np.allclose(htk[:, 0], math.sqrt(2.0) * ortho[:, 0])


def test_lifter_weights_and_their_effect():
    weights = sinusoidal_lifter(12, 22)
    assert weights.shape == (12,) and weights[10] == pytest.approx(12.0)
    assert np.allclose(sinusoidal_lifter(12, 0), 1.0)
    plain = cepstral_features(tone(), RATE, FrontendConfig(lifter=0))
    liftered = cepstral_features(tone(), RATE, FrontendConfig(lifter=22))
    assert np.allclose(liftered[:, :12], plain[:, :12] * weights)
    assert np.allclose(liftered[:, 12], plain[:, 12])  # c0 is never liftered


def test_energy_term_goes_last_and_is_the_raw_frame_energy():
    config = FrontendConfig(energy_term="log_energy", preemphasis=0.97, delta_order=0)
    signal = tone()
    features = cepstral_features(signal, RATE, config)
    frame_length, hop, _ = frame_sizes(RATE, config)
    # Measured before pre-emphasis and windowing, so the pre-emphasis leaves it alone.
    for index in (0, 3):
        raw = signal[index * hop : index * hop + frame_length]
        assert features[index, -1] == pytest.approx(np.log(np.sum(raw**2)), rel=1e-12)
    assert feature_layout(config)[-1] == "log_e"
    assert cepstral_features(signal, RATE, FrontendConfig(energy_term="none")).shape[1] == 12


def test_configured_like_speechdsp_mfcc_the_cepstra_agree():
    # speechdsp.mfcc: 25 / 10 ms, periodic Hamming, 26 filters from 0 Hz, 512-point FFT, L = 22.
    noise = np.random.default_rng(3).standard_normal(RATE // 2) + tone()
    matching = FrontendConfig(
        frame_ms=25.0,
        hop_ms=10.0,
        n_fft=512,
        window="hamming",
        n_mels=26,
        fmin_hz=0.0,
        energy_term="none",
        delta_order=0,
    )
    ours = cepstral_features(noise, RATE, matching)
    reference = speechdsp.mfcc(noise, RATE, n_mfcc=13, n_mels=26, n_fft=512, append_energy=False)
    assert ours.shape == (reference.shape[0], 12)
    assert np.allclose(ours, reference[:, 1:], rtol=1e-10, atol=1e-10)


def test_digital_silence_stays_finite():
    features = frame_features(np.zeros(RATE // 2), RATE, FrontendConfig())
    assert np.all(np.isfinite(features))
    assert features[:, 12].max() == pytest.approx(
        math.sqrt(2.0 / 23) * 23 * math.log(1e-10), rel=1e-9
    )


def test_frame_sizes_follow_the_actual_sampling_rate():
    config = FrontendConfig()
    assert frame_sizes(44_100, config) == (1323, 661, 2048)
    assert frame_sizes(16_000, config) == (480, 240, 512)
    assert frame_sizes(8_000, config) == (240, 120, 256)
    for rate in (8_000, 16_000, 44_100):
        n_frames = frame_features(tone(1.0, rate=rate), rate, config).shape[0]
        assert n_frames == (rate - frame_sizes(rate, config)[0]) // frame_sizes(rate, config)[1] + 1


def test_short_signals_give_zero_frames():
    assert cepstral_features(np.ones(100), RATE, FrontendConfig()).shape == (0, 13)
    assert frame_features(np.ones(100), RATE, FrontendConfig()).shape == (0, 39)


def test_window_definitions():
    assert np.allclose(analysis_window("hamming_symmetric", 7), get_window("hamming", 7, False))
    assert np.allclose(analysis_window("hann", 8), get_window("hann", 8, True))
    assert np.all(analysis_window("rectangular", 4) == 1.0)
    with pytest.raises(ValueError, match=r"frontend\.window"):
        analysis_window("kaiser", 8)


def test_invalid_combinations_are_reported():
    with pytest.raises(ValueError, match="Nyquist"):
        cepstral_features(tone(), 8_000, FrontendConfig(fmax_hz=6_000.0))
    with pytest.raises(ValueError, match="n_fft"):
        frame_sizes(16_000, FrontendConfig(n_fft=256))


def test_delta_of_a_ramp_is_its_slope():
    ramp = np.outer(np.arange(30.0), [1.0, -2.0])
    dynamics = append_dynamics(ramp, 2, (7, 5))
    assert np.allclose(dynamics[5:-5, 2:4], [1.0, -2.0])
    assert np.allclose(dynamics[8:-8, 4:6], 0.0)
    with pytest.raises(ValueError, match="delta order"):
        append_dynamics(ramp, 3)


def test_presets():
    assert htk_c0_preset() == FrontendConfig()
    compact = compact_preset()
    assert compact.n_mels == 26 and compact.energy_term == "log_energy"
    assert frame_features(tone(), RATE, compact).shape[1] == 39


def test_a_higher_tone_changes_the_cepstrum():
    low = cepstral_features(tone(frequency=300.0), RATE, FrontendConfig())
    high = cepstral_features(tone(frequency=3000.0), RATE, FrontendConfig())
    assert not np.allclose(low.mean(axis=0)[:12], high.mean(axis=0)[:12], atol=1.0)
