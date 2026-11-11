"""Synthetic sustained vowels with a controllable severity, and toy regression data.

Voice recordings
----------------
Each recording is a source-filter vowel:

1. a glottal flow train of Rosenberg pulses (Rosenberg, 1971) whose cycle
   lengths and peak amplitudes are perturbed from cycle to cycle by Gaussian
   jitter and shimmer;
2. a cascade of three formant resonators for /a/, /i/ or /u/ (male averages
   of Peterson & Barney, 1952) and a first difference for lip radiation;
3. aspiration noise shaped by the same vocal tract, scaled to a
   harmonics-to-noise ratio.

A severity rating :math:`s \\in [0, 100]` (a CAPE-V-like visual analogue
scale) moves jitter, shimmer and the noise level linearly between the
``synthetic.*_range`` endpoints. Speakers differ in mean pitch and in their
own severity; the recordings of one speaker vary around both and in vowel
and loudness. A model therefore has to find the severity cues among nuisance
variation, and the ``group`` column lets cross-validation keep speakers apart.
Recordings at or above ``label_threshold`` are labeled ``dysphonic``.

Regression toys
---------------
Small tables with a known generating model: a second-order response
surface, a cubic polynomial and four nearly collinear mixture proportions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import speechdsp
from numpy.typing import NDArray
from scipy.signal import sosfilt

from .dataset import DATASET_INDEX_NAME

if TYPE_CHECKING:
    from .config import SyntheticConfig

_LOGGER = logging.getLogger(__name__)

#: Formant (frequency, bandwidth) pairs in Hz of the synthetic vowels.
VOWEL_FORMANTS: dict[str, tuple[tuple[float, float], ...]] = {
    "a": ((730.0, 90.0), (1090.0, 110.0), (2440.0, 170.0)),
    "i": ((270.0, 60.0), (2290.0, 100.0), (3010.0, 170.0)),
    "u": ((300.0, 60.0), (870.0, 90.0), (2240.0, 160.0)),
}
#: Labels written for recordings below and at or above the severity threshold.
CLASS_NAMES: tuple[str, str] = ("healthy", "dysphonic")

#: Open and closing phases of the Rosenberg pulse, as fractions of a cycle.
_OPEN_QUOTIENT = 0.4
_CLOSING_QUOTIENT = 0.16
#: Range of the peak level of a written recording (full scale = 1).
_PEAK_RANGE = (0.3, 0.9)
#: Relative spread of a recording's pitch around its speaker's mean.
_PITCH_SPREAD = 0.04


def rosenberg_pulse(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rosenberg glottal flow at cycle phases in ``[0, 1)``.

    A raised-cosine opening phase, a quarter-cosine closing phase and a
    closed phase of zero flow.

    References
    ----------
    .. [1] A. E. Rosenberg, "Effect of glottal pulse shape on the quality of
       natural vowels," The Journal of the Acoustical Society of America,
       vol. 49, no. 2B, pp. 583-590, 1971.

    Examples
    --------
    >>> rosenberg_pulse(np.array([0.0, 0.2, 0.4, 0.48, 0.9])).round(3).tolist()
    [0.0, 0.5, 1.0, 0.707, 0.0]
    """
    phase = np.asarray(phase, dtype=np.float64)
    opening = 0.5 * (1.0 - np.cos(np.pi * phase / _OPEN_QUOTIENT))
    closing = np.cos(0.5 * np.pi * (phase - _OPEN_QUOTIENT) / _CLOSING_QUOTIENT)
    return np.where(
        phase < _OPEN_QUOTIENT,
        opening,
        np.where(phase < _OPEN_QUOTIENT + _CLOSING_QUOTIENT, closing, 0.0),
    )


def glottal_source(
    n_samples: int,
    sample_rate: int,
    f0_hz: float,
    *,
    jitter: float,
    shimmer: float,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Rosenberg glottal flow train whose cycle lengths and amplitudes carry jitter and shimmer.

    Parameters
    ----------
    n_samples : int
        Output length.
    sample_rate : int
        Sampling rate in Hz.
    f0_hz : float
        Mean fundamental frequency.
    jitter, shimmer : float
        Relative standard deviations of the cycle length and the cycle amplitude.
    rng : numpy.random.Generator or int, optional
        Random source.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)
        Glottal flow, non-negative.

    Examples
    --------
    >>> flow = glottal_source(1600, 16_000, 100.0, jitter=0.0, shimmer=0.0, rng=0)
    >>> bool(np.allclose(flow[:160], flow[160:320])), float(flow.max())
    (True, 1.0)
    """
    generator = np.random.default_rng(rng)
    period = sample_rate / f0_hz
    n_cycles = math.ceil(n_samples / (period * 0.5)) + 2
    periods = period * np.clip(1.0 + jitter * generator.standard_normal(n_cycles), 0.5, 1.5)
    amplitudes = np.clip(1.0 + shimmer * generator.standard_normal(n_cycles), 0.05, None)
    starts = np.concatenate([[0.0], np.cumsum(periods)])
    t = np.arange(n_samples, dtype=np.float64)
    cycle = np.searchsorted(starts, t, side="right") - 1
    phase = (t - starts[cycle]) / periods[cycle]
    return rosenberg_pulse(phase) * amplitudes[cycle]


def vocal_tract(signal: NDArray[np.float64], sample_rate: int, vowel: str) -> NDArray[np.float64]:
    """Cascade of second-order formant resonators with unit gain at 0 Hz."""
    if vowel not in VOWEL_FORMANTS:
        raise ValueError(f"unknown vowel {vowel!r}; choose one of {sorted(VOWEL_FORMANTS)}")
    sections = []
    for frequency, bandwidth in VOWEL_FORMANTS[vowel]:
        radius = math.exp(-math.pi * bandwidth / sample_rate)
        angle = 2.0 * math.pi * frequency / sample_rate
        a1, a2 = -2.0 * radius * math.cos(angle), radius**2
        sections.append([1.0 + a1 + a2, 0.0, 0.0, 1.0, a1, a2])
    return sosfilt(np.asarray(sections), signal)


def severity_parameters(severity: float, config: SyntheticConfig) -> dict[str, float]:
    """Jitter, shimmer and harmonics-to-noise ratio for one severity rating.

    Examples
    --------
    >>> from voice_severity_bench.config import SyntheticConfig
    >>> severity_parameters(50.0, SyntheticConfig())
    {'jitter': 0.016, 'shimmer': 0.16, 'hnr_db': 18.0}
    """
    fraction = float(np.clip(severity, 0.0, 100.0)) / 100.0

    def between(pair: tuple[float, float]) -> float:
        return round(pair[0] + fraction * (pair[1] - pair[0]), 12)

    return {
        "jitter": between(config.jitter_range),
        "shimmer": between(config.shimmer_range),
        "hnr_db": between(config.hnr_range_db),
    }


def make_vowel(
    severity: float,
    *,
    config: SyntheticConfig | None = None,
    f0_hz: float = 120.0,
    vowel: str = "a",
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """One synthetic sustained vowel whose perturbations follow ``severity``.

    Parameters
    ----------
    severity : float
        Rating on the 0-100 scale.
    config : SyntheticConfig, optional
        Duration, sampling rate and the severity-to-perturbation ranges.
    f0_hz : float, optional
        Mean fundamental frequency.
    vowel : {"a", "i", "u"}, optional
        Vowel quality.
    rng : numpy.random.Generator or int, optional
        Random source.

    Returns
    -------
    numpy.ndarray, shape (round(duration_s * sample_rate),)
        Waveform with its peak at 0.9.

    Examples
    --------
    >>> from voice_severity_bench.config import SyntheticConfig
    >>> short = SyntheticConfig(duration_s=0.5)
    >>> vowel = make_vowel(40.0, config=short, f0_hz=110.0, vowel="i", rng=1)
    >>> vowel.shape, round(float(np.abs(vowel).max()), 6)
    ((8000,), 0.9)
    """
    from .config import SyntheticConfig

    synthetic = config if config is not None else SyntheticConfig()
    generator = np.random.default_rng(rng)
    rate = synthetic.sample_rate
    n_samples = round(synthetic.duration_s * rate)
    parameters = severity_parameters(severity, synthetic)
    flow = glottal_source(
        n_samples,
        rate,
        f0_hz,
        jitter=parameters["jitter"],
        shimmer=parameters["shimmer"],
        rng=generator,
    )
    flow -= flow.mean()
    voiced = np.diff(vocal_tract(flow, rate, vowel), prepend=0.0)
    noise = np.diff(vocal_tract(generator.standard_normal(n_samples), rate, vowel), prepend=0.0)
    voiced_power = float(np.mean(voiced**2))
    noise_power = float(np.mean(noise**2))
    gain = math.sqrt(voiced_power / (noise_power * 10.0 ** (parameters["hnr_db"] / 10.0)))
    signal = voiced + gain * noise
    return 0.9 * signal / float(np.max(np.abs(signal)))


def write_synthetic_dataset(
    folder: str | Path,
    config: SyntheticConfig | None = None,
    *,
    rng: np.random.Generator | int | None = None,
) -> Path:
    """Write synthetic recordings and their ``dataset.csv`` index.

    Recordings go to ``items/item_<NNN>.wav`` (16-bit PCM); the index has
    the columns ``path``, ``severity`` (0-100, one decimal), ``label``
    (``healthy`` or ``dysphonic``) and ``group`` (``speaker_<NN>``).

    Parameters
    ----------
    folder : str or pathlib.Path
        Output folder (created if needed).
    config : SyntheticConfig, optional
        Sizes and generating parameters; ``None`` means ``SyntheticConfig()``.
    rng : numpy.random.Generator or int, optional
        Random source; the same seed writes byte-identical files.

    Returns
    -------
    pathlib.Path
        Path of ``dataset.csv``.

    Examples
    --------
    >>> import tempfile
    >>> from voice_severity_bench.config import SyntheticConfig
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     tiny = SyntheticConfig(n_speakers=2, n_recordings_per_speaker=2, duration_s=0.2)
    ...     index = pd.read_csv(write_synthetic_dataset(folder, tiny, rng=0))
    ...     index.columns.tolist(), index["path"].tolist()[:2], index["group"].nunique()
    (['path', 'severity', 'label', 'group'], ['items/item_000.wav', 'items/item_001.wav'], 2)
    """
    from .config import SyntheticConfig

    synthetic = config if config is not None else SyntheticConfig()
    generator = np.random.default_rng(rng)
    root = Path(folder)
    n_speakers, per_speaker = synthetic.n_speakers, synthetic.n_recordings_per_speaker
    speaker_severity = generator.uniform(0.0, 100.0, n_speakers)
    speaker_f0 = generator.uniform(*synthetic.f0_range_hz, n_speakers)
    vowels = sorted(VOWEL_FORMANTS)
    rows = []
    for item in range(synthetic.n_recordings):
        speaker = item // per_speaker
        severity = float(
            np.clip(
                speaker_severity[speaker]
                + synthetic.within_speaker_sd * generator.standard_normal(),
                0.0,
                100.0,
            )
        )
        severity = round(severity, 1)
        f0 = float(speaker_f0[speaker] * (1.0 + _PITCH_SPREAD * generator.standard_normal()))
        vowel = vowels[int(generator.integers(len(vowels)))]
        peak = float(generator.uniform(*_PEAK_RANGE))
        item_seed = int(generator.integers(2**32))
        waveform = make_vowel(severity, config=synthetic, f0_hz=f0, vowel=vowel, rng=item_seed)
        relative = Path("items") / f"item_{item:03d}.wav"
        speechdsp.write_wav(root / relative, waveform * (peak / 0.9), synthetic.sample_rate)
        rows.append(
            {
                "path": relative.as_posix(),
                "severity": severity,
                "label": CLASS_NAMES[int(severity >= synthetic.label_threshold)],
                "group": f"speaker_{speaker:02d}",
            }
        )
    index = root / DATASET_INDEX_NAME
    pd.DataFrame(rows).to_csv(index, index=False, lineterminator="\n")
    _LOGGER.info("%d synthetic recordings written to %s", len(rows), root.as_posix())
    return index


def make_quadratic_surface_data(
    n_samples: int = 30, *, noise_sd: float = 1.0, rng: np.random.Generator | int | None = None
) -> pd.DataFrame:
    """Two predictors and a response from a known second-order surface.

    :math:`y = 5 + 2x_1 - 1.5x_2 - 0.8x_1^2 + 0.3x_2^2 + 0.6x_1x_2 + \\varepsilon`
    with :math:`x_1, x_2 \\sim U(-3, 3)`.

    Examples
    --------
    >>> make_quadratic_surface_data(5, rng=0).columns.tolist()
    ['x1', 'x2', 'y']
    """
    generator = np.random.default_rng(rng)
    x1, x2 = generator.uniform(-3.0, 3.0, size=(2, n_samples))
    y = 5.0 + 2.0 * x1 - 1.5 * x2 - 0.8 * x1**2 + 0.3 * x2**2 + 0.6 * x1 * x2
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "y": y + noise_sd * generator.standard_normal(n_samples)}
    )


def make_polynomial_data(
    n_samples: int = 10,
    *,
    coefficients: tuple[float, ...] = (1.0, -2.0, 0.0, 0.5),
    noise_sd: float = 0.5,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Points of a known polynomial with Gaussian noise.

    ``coefficients`` are in increasing power; :math:`x` is evenly spaced in
    :math:`[-2, 2]`.

    Examples
    --------
    >>> data = make_polynomial_data(5, noise_sd=0.0)
    >>> data["y"].round(2).tolist()
    [1.0, 2.5, 1.0, -0.5, 1.0]
    """
    generator = np.random.default_rng(rng)
    x = np.linspace(-2.0, 2.0, n_samples)
    y = np.polynomial.polynomial.polyval(x, coefficients)
    return pd.DataFrame({"x": x, "y": y + noise_sd * generator.standard_normal(n_samples)})


def make_collinear_data(
    n_samples: int = 40, *, noise_sd: float = 2.0, rng: np.random.Generator | int | None = None
) -> pd.DataFrame:
    """Four mixture percentages that nearly sum to 100, and a response.

    The proportions are Dirichlet draws scaled to 98-100 %, so any one of them
    is almost a linear function of the other three. Variance inflation factors
    explode on such data and stepwise selection becomes unstable.
    :math:`y = 50 + 1.5 p_1 + 0.6 p_2 + 0.1 p_3 - 0.2 p_4 + \\varepsilon`.

    Examples
    --------
    >>> data = make_collinear_data(10, rng=0)
    >>> bool(data[["p1", "p2", "p3", "p4"]].sum(axis=1).between(98, 100).all())
    True
    """
    generator = np.random.default_rng(rng)
    shares = generator.dirichlet([2.0, 5.0, 2.0, 3.0], size=n_samples)
    proportions = shares * generator.uniform(98.0, 100.0, size=(n_samples, 1))
    coef = np.array([1.5, 0.6, 0.1, -0.2])
    y = 50.0 + proportions @ coef + noise_sd * generator.standard_normal(n_samples)
    frame = pd.DataFrame(proportions, columns=["p1", "p2", "p3", "p4"])
    frame["y"] = y
    return frame
