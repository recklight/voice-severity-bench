"""Configurable cepstral front end: MFCCs, an explicit energy term and dynamics.

Every choice that distinguishes one MFCC recipe from another is a field of
:class:`~voice_severity_bench.config.FrontendConfig`, so a recipe is picked in
the configuration file and :mod:`~voice_severity_bench.ablation` can compare
several in one run. The chain is

1. pre-emphasis :math:`y[n] = x[n] - a\\,x[n-1]` over the whole signal (the
   first sample passes through unchanged);
2. frames of ``frame_ms`` advanced by ``hop_ms``, both converted to samples
   with the *actual* sampling rate of the file;
3. an analysis window (a symmetric Hamming window by default) and the power
   spectrum :math:`|X_k|^2` of an ``n_fft``-point FFT, not divided by ``n_fft``;
4. ``n_mels`` triangular filters with unit peaks, equally spaced on the mel
   scale :math:`m(f) = 1127 \\ln(1 + f/700)` between ``fmin_hz`` and ``fmax_hz``
   (Davis & Mermelstein, 1980; O'Shaughnessy, 1987);
5. the natural logarithm with a floor, so digital silence stays finite;
6. the DCT-II in the HTK convention (Young et al., 2006)

   .. math::

      c_m = \\sqrt{2/N} \\sum_{n=1}^{N} \\log E_n
            \\cos\\left(\\frac{\\pi m (n - 0.5)}{N}\\right),

   which equals SciPy's orthonormal DCT for :math:`m \\ge 1` and is
   :math:`\\sqrt 2` times larger for :math:`c_0`;
7. the sinusoidal lifter :math:`1 + (L/2) \\sin(\\pi m / L)` on
   :math:`c_1 \\dots c_M` (Juang et al., 1987);
8. an energy term appended *after* the cepstra, HTK style: ``c0``, the log
   energy of the raw frame (before pre-emphasis and windowing), or nothing;
9. optional regression deltas and delta-deltas (Furui, 1986), giving the
   familiar 13, 26 and 39 dimensional layouts ``[static | delta | delta-delta]``.

The defaults borrow the filterbank of the distributed speech recognition front
end (ETSI ES 201 108): 23 filters from 64 Hz to Nyquist, pre-emphasis 0.97, a
symmetric Hamming window and 12 cepstra plus ``c0``. They do not reproduce that
standard: the frames are 30 ms every 15 ms (the standard uses 25 ms every
10 ms), the filters weight the power rather than the magnitude spectrum, and
the cepstra use the HTK scaling and are liftered.

:func:`speechdsp.mfcc` implements one fixed recipe (periodic Hamming window,
filters from 0 Hz, orthonormal DCT, ``c0`` replaced by the log energy). The
choices that recipe fixes are exactly what this module compares, so it builds
the chain from the :mod:`speechdsp` building blocks (pre-emphasis, framing, mel
filterbank, deltas); configured like :func:`speechdsp.mfcc`, it returns the
same ``c1 .. c12``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import speechdsp
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dct
from scipy.signal import get_window

if TYPE_CHECKING:
    from .config import FrontendConfig


#: Analysis windows accepted by ``frontend.window``.
WINDOWS: tuple[str, ...] = ("hamming_symmetric", "hamming", "hann", "rectangular")
#: Energy terms accepted by ``frontend.energy_term``.
ENERGY_TERMS: tuple[str, ...] = ("c0", "log_energy", "none")
#: Orders of dynamic features accepted by ``frontend.delta_order``.
DELTA_ORDERS: tuple[int, ...] = (0, 1, 2)

#: Column-name prefixes of the static, delta and delta-delta blocks.
_BLOCK_PREFIXES: tuple[str, ...] = ("", "d_", "dd_")
#: Tolerance that keeps ``floor(ms * rate / 1000)`` from losing a sample to rounding.
_SAMPLE_EPSILON = 1e-9


def frame_sizes(sample_rate: int, frontend: FrontendConfig) -> tuple[int, int, int]:
    """Frame length, hop and FFT size in samples for one sampling rate.

    Durations are converted with ``floor``, so 15 ms at 44.1 kHz is 661
    samples; ``frontend.n_fft = None`` picks the next power of two that holds a
    frame.

    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz.
    frontend : FrontendConfig
        Front-end settings.

    Returns
    -------
    tuple of int
        ``(frame_length, hop, n_fft)``.

    Raises
    ------
    ValueError
        If a frame is shorter than two samples or does not fit ``n_fft``.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig
    >>> frame_sizes(44_100, FrontendConfig()), frame_sizes(16_000, FrontendConfig())
    ((1323, 661, 2048), (480, 240, 512))
    """
    if sample_rate <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate!r}")
    frame_length = math.floor(frontend.frame_ms * sample_rate / 1000.0 + _SAMPLE_EPSILON)
    hop = math.floor(frontend.hop_ms * sample_rate / 1000.0 + _SAMPLE_EPSILON)
    if frame_length < 2 or hop < 1:
        raise ValueError(
            f"frontend.frame_ms={frontend.frame_ms} and frontend.hop_ms={frontend.hop_ms} "
            f"are too short for {sample_rate} Hz"
        )
    if frontend.n_fft is None:
        n_fft = 1 << (frame_length - 1).bit_length()
    else:
        n_fft = int(frontend.n_fft)
        if n_fft < frame_length:
            raise ValueError(
                f"frontend.n_fft={n_fft} is shorter than a frame of {frame_length} samples "
                f"at {sample_rate} Hz; raise it or set it to null (automatic)"
            )
    return frame_length, hop, n_fft


def analysis_window(name: str, length: int) -> NDArray[np.float64]:
    """Taps of one of the :data:`WINDOWS`.

    ``"hamming_symmetric"`` is :math:`0.54 - 0.46\\cos(2\\pi n/(N-1))`, the
    window of HTK and the ETSI front end; ``"hamming"`` and ``"hann"`` are the
    periodic (DFT-even) variants.

    Parameters
    ----------
    name : {"hamming_symmetric", "hamming", "hann", "rectangular"}
        Window name.
    length : int
        Number of taps.

    Returns
    -------
    numpy.ndarray, shape (length,)
        Window taps.

    Examples
    --------
    >>> taps = analysis_window("hamming_symmetric", 5)
    >>> [round(float(v), 2) for v in taps]
    [0.08, 0.54, 1.0, 0.54, 0.08]
    """
    if name not in WINDOWS:
        raise ValueError(
            f"frontend.window={name!r} is not supported; choose one of {list(WINDOWS)}"
        )
    if name == "rectangular":
        return np.ones(length, dtype=np.float64)
    if name == "hamming_symmetric":
        return np.asarray(get_window("hamming", length, fftbins=False), dtype=np.float64)
    return np.asarray(get_window(name, length, fftbins=True), dtype=np.float64)


def htk_dct(log_energies: ArrayLike, n_coefficients: int) -> NDArray[np.float64]:
    """DCT-II of log filterbank energies in the HTK scaling, ``c0`` included.

    Parameters
    ----------
    log_energies : array_like, shape (n_frames, n_filters)
        Log mel energies.
    n_coefficients : int
        Number of outputs ``c0 .. c_{n-1}``; at most ``n_filters``.

    Returns
    -------
    numpy.ndarray, shape (n_frames, n_coefficients)
        :math:`c_m = \\sqrt{2/N}\\sum_n \\log E_n \\cos(\\pi m (n - 0.5) / N)`.

    References
    ----------
    .. [1] S. Young et al., The HTK Book (for HTK Version 3.4). Cambridge, U.K.:
       Cambridge University Engineering Department, 2006.

    Examples
    --------
    The HTK scaling equals the orthonormal DCT except for :math:`c_0`, which is
    :math:`\\sqrt 2` times larger:

    >>> from scipy.fft import dct
    >>> energies = np.log(np.arange(1.0, 24.0))[None, :]
    >>> htk = htk_dct(energies, 13)
    >>> ortho = dct(energies, type=2, norm="ortho")[:, :13]
    >>> bool(np.allclose(htk[:, 1:], ortho[:, 1:])), round(float(htk[0, 0] / ortho[0, 0]), 6)
    (True, 1.414214)
    """
    values = np.asarray(log_energies, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"log energies must be 2-D (frames x filters), got shape {values.shape}")
    n_filters = values.shape[1]
    if not 1 <= n_coefficients <= n_filters:
        raise ValueError(f"cannot take {n_coefficients} coefficients from {n_filters} filters")
    # scipy's unnormalized DCT-II is 2 * sum(...); sqrt(2/N) / 2 = 1 / sqrt(2N).
    return dct(values, type=2, axis=1, norm=None)[:, :n_coefficients] / math.sqrt(2.0 * n_filters)


def sinusoidal_lifter(n_ceps: int, lifter: int) -> NDArray[np.float64]:
    """Weights :math:`1 + (L/2)\\sin(\\pi m / L)` for :math:`c_1 \\dots c_M`.

    Parameters
    ----------
    n_ceps : int
        Number of cepstra ``M`` (``c0`` excluded).
    lifter : int
        Lifter length ``L``; ``0`` returns unit weights.

    Returns
    -------
    numpy.ndarray, shape (n_ceps,)
        Lifter weights.

    References
    ----------
    .. [1] B.-H. Juang, L. R. Rabiner, and J. G. Wilpon, "On the use of bandpass
       liftering in speech recognition," IEEE Transactions on Acoustics, Speech,
       and Signal Processing, vol. 35, no. 7, pp. 947-954, 1987.

    Examples
    --------
    >>> weights = sinusoidal_lifter(12, 22)
    >>> round(float(weights[0]), 4), round(float(weights[10]), 4)
    (2.5655, 12.0)
    """
    if lifter < 0:
        raise ValueError(f"frontend.lifter must be non-negative, got {lifter!r}")
    if lifter == 0:
        return np.ones(n_ceps, dtype=np.float64)
    order = np.arange(1, n_ceps + 1, dtype=np.float64)
    return 1.0 + (lifter / 2.0) * np.sin(np.pi * order / lifter)


def cepstral_features(
    signal: ArrayLike, sample_rate: int, frontend: FrontendConfig
) -> NDArray[np.float64]:
    """Static cepstral features of one signal, energy term last.

    Parameters
    ----------
    signal : array_like, shape (n_samples,)
        Mono waveform.
    sample_rate : int
        Sampling rate in Hz; frame sizes and mel edges follow it.
    frontend : FrontendConfig
        Front-end settings.

    Returns
    -------
    numpy.ndarray, shape (n_frames, n_static)
        Columns ``c1 .. cM`` followed by ``c0`` or ``log_e`` (see
        :func:`feature_layout`). Zero rows when the signal is shorter than
        one frame.

    Raises
    ------
    ValueError
        If the mel range does not fit the sampling rate.

    References
    ----------
    .. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric
       representations for monosyllabic word recognition in continuously spoken
       sentences," IEEE Transactions on Acoustics, Speech, and Signal Processing,
       vol. 28, no. 4, pp. 357-366, 1980.
    .. [2] S. Young et al., The HTK Book (for HTK Version 3.4). Cambridge, U.K.:
       Cambridge University Engineering Department, 2006.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig
    >>> rate = 16_000
    >>> tone = np.sin(2 * np.pi * 220.0 * np.arange(rate) / rate)
    >>> cepstral_features(tone, rate, FrontendConfig()).shape
    (65, 13)
    >>> compact = FrontendConfig(frame_ms=25.0, hop_ms=10.0, n_mels=26, energy_term="none")
    >>> cepstral_features(tone, rate, compact).shape
    (98, 12)
    """
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {samples.shape}")
    frame_length, hop, n_fft = frame_sizes(sample_rate, frontend)
    n_static = frontend.n_ceps + (frontend.energy_term != "none")
    if speechdsp.num_frames(samples.size, frame_length, hop) == 0:
        return np.zeros((0, n_static), dtype=np.float64)

    nyquist = sample_rate / 2.0
    fmax = nyquist if frontend.fmax_hz is None else float(frontend.fmax_hz)
    if fmax > nyquist or frontend.fmin_hz >= fmax:
        raise ValueError(
            f"mel range frontend.fmin_hz={frontend.fmin_hz} .. frontend.fmax_hz={fmax} "
            f"does not fit a {sample_rate} Hz signal (Nyquist {nyquist:g} Hz)"
        )

    emphasized = speechdsp.preemphasis(samples, frontend.preemphasis)
    frames = speechdsp.enframe(emphasized, frame_length, hop)
    windowed = frames * analysis_window(frontend.window, frame_length)
    power = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) ** 2
    filterbank = speechdsp.mel_filterbank(
        sample_rate, n_fft, n_mels=frontend.n_mels, fmin=frontend.fmin_hz, fmax=fmax
    )
    log_mel = np.log(np.maximum(power @ filterbank.T, frontend.log_floor))
    coefficients = htk_dct(log_mel, frontend.n_ceps + 1)
    cepstra = coefficients[:, 1:] * sinusoidal_lifter(frontend.n_ceps, frontend.lifter)

    if frontend.energy_term == "c0":
        energy = coefficients[:, :1]
    elif frontend.energy_term == "log_energy":
        # Energy of the raw frame, before pre-emphasis and windowing, which is
        # where the ETSI front end takes its logE and HTK its default raw energy.
        raw = speechdsp.enframe(samples, frame_length, hop)
        energy = np.log(np.maximum(np.sum(raw**2, axis=1), frontend.log_floor))[:, None]
    else:
        return cepstra
    return np.hstack([cepstra, energy])


def append_dynamics(
    static: ArrayLike, order: int, widths: tuple[int, int] = (7, 5)
) -> NDArray[np.float64]:
    """Append regression deltas (order 1) and delta-deltas (order 2).

    The delta of frame :math:`t` is
    :math:`d_t = \\sum_{n=1}^{N} n (c_{t+n} - c_{t-n}) / (2 \\sum_n n^2)` with
    the edges replicated, computed by :func:`speechdsp.delta`; the
    delta-delta is the delta of the delta with the second window.

    Parameters
    ----------
    static : array_like, shape (n_frames, n_static)
        Static features.
    order : {0, 1, 2}
        How many derivative blocks to append.
    widths : tuple of int, optional
        Odd window lengths ``2N + 1`` of the delta and of the delta-delta;
        ``(7, 5)`` means :math:`N = 3` and :math:`N = 2`.

    Returns
    -------
    numpy.ndarray, shape (n_frames, n_static * (order + 1))
        ``[static | delta | delta-delta]``.

    References
    ----------
    .. [1] S. Furui, "Speaker-independent isolated word recognition using
       dynamic features of speech spectrum," IEEE Transactions on Acoustics,
       Speech, and Signal Processing, vol. 34, no. 1, pp. 52-59, 1986.

    Examples
    --------
    The delta of a linear ramp is its slope away from the edges:

    >>> ramp = np.arange(20.0)[:, None] * 0.5
    >>> features = append_dynamics(ramp, 2)
    >>> features.shape, float(features[10, 1]), float(features[10, 2])
    ((20, 3), 0.5, 0.0)
    """
    values = np.asarray(static, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"static features must be 2-D, got shape {values.shape}")
    if order not in DELTA_ORDERS:
        raise ValueError(f"delta order must be one of {list(DELTA_ORDERS)}, got {order!r}")
    blocks = [values]
    if order >= 1:
        blocks.append(speechdsp.delta(values, width=int(widths[0])))
    if order == 2:
        blocks.append(speechdsp.delta(blocks[1], width=int(widths[1])))
    return np.hstack(blocks)


def frame_features(
    signal: ArrayLike, sample_rate: int, frontend: FrontendConfig
) -> NDArray[np.float64]:
    """Static features plus the dynamics that ``frontend.delta_order`` asks for.

    Parameters
    ----------
    signal : array_like, shape (n_samples,)
        Mono waveform.
    sample_rate : int
        Sampling rate in Hz.
    frontend : FrontendConfig
        Front-end settings.

    Returns
    -------
    numpy.ndarray, shape (n_frames, frontend.n_dims)
        Frame-level feature matrix whose columns are :func:`feature_layout`.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig
    >>> rate = 16_000
    >>> noise = np.random.default_rng(0).standard_normal(rate // 2)
    >>> [frame_features(noise, rate, FrontendConfig(delta_order=k)).shape[1] for k in (0, 1, 2)]
    [13, 26, 39]
    """
    static = cepstral_features(signal, sample_rate, frontend)
    return append_dynamics(static, frontend.delta_order, frontend.delta_widths)


def feature_layout(frontend: FrontendConfig) -> tuple[str, ...]:
    """Column names of :func:`frame_features`; the pooled feature names are built from them.

    Parameters
    ----------
    frontend : FrontendConfig
        Front-end settings.

    Returns
    -------
    tuple of str
        ``c1 .. cM``, then ``c0`` or ``log_e``, repeated with the ``d_`` and
        ``dd_`` prefixes for the dynamic blocks.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig
    >>> names = feature_layout(FrontendConfig(n_ceps=2, delta_order=2))
    >>> names
    ('c1', 'c2', 'c0', 'd_c1', 'd_c2', 'd_c0', 'dd_c1', 'dd_c2', 'dd_c0')
    """
    static = [f"c{index}" for index in range(1, frontend.n_ceps + 1)]
    if frontend.energy_term == "c0":
        static.append("c0")
    elif frontend.energy_term == "log_energy":
        static.append("log_e")
    return tuple(
        prefix + name for prefix in _BLOCK_PREFIXES[: frontend.delta_order + 1] for name in static
    )


def htk_c0_preset() -> FrontendConfig:
    """The default front end: HTK-style ``c0`` layout, 13 static dimensions.

    30 ms frames every 15 ms, symmetric Hamming window, 23 mel filters from
    64 Hz to Nyquist, pre-emphasis 0.97, 12 cepstra liftered with
    :math:`L = 22` plus ``c0`` in the HTK scaling, deltas and delta-deltas.
    The filterbank, window and pre-emphasis follow the ETSI distributed speech
    recognition front end; the framing, the power spectrum and the liftered
    HTK cepstra do not (see the module documentation).

    Returns
    -------
    FrontendConfig
        Equal to the default ``FrontendConfig()``.

    References
    ----------
    .. [1] ETSI ES 201 108 V1.1.3, "Speech Processing, Transmission and Quality
       Aspects (STQ); Distributed speech recognition; Front-end feature
       extraction algorithm; Compression algorithms," ETSI, 2003.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig
    >>> htk_c0_preset() == FrontendConfig()
    True
    """
    from .config import FrontendConfig

    return FrontendConfig()


def compact_preset() -> FrontendConfig:
    """A common 25 ms / 10 ms recipe: 26 filters, periodic Hamming, log energy.

    Returns
    -------
    FrontendConfig
        The same 12 cepstra and dynamics as :func:`htk_c0_preset` with the
        shorter framing and the log frame energy instead of ``c0``.

    Examples
    --------
    >>> preset = compact_preset()
    >>> preset.frame_ms, preset.hop_ms, preset.n_mels, preset.energy_term
    (25.0, 10.0, 26, 'log_energy')
    """
    from .config import FrontendConfig

    return FrontendConfig(
        frame_ms=25.0,
        hop_ms=10.0,
        window="hamming",
        n_mels=26,
        fmin_hz=0.0,
        energy_term="log_energy",
    )
