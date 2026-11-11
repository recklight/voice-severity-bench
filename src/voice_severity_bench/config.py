"""Run configuration: frozen dataclasses validated on construction, stored as YAML.

There is one dataclass per YAML section, gathered in :class:`Config`.
Each section validates itself in ``__post_init__`` and raises
:class:`~voice_severity_bench.errors.ConfigError` naming the dotted key, the value it got and
what is allowed. Unknown keys are rejected rather than ignored, so a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .association import P_VALUE_CORRECTIONS
from .cepstral_frontend import DELTA_ORDERS, ENERGY_TERMS, WINDOWS
from .design_matrix import POLYNOMIAL_BASES
from .errors import ConfigError
from .evaluation import metric_names
from .models import ACTIVATIONS, CLASS_WEIGHTS, IMPUTERS, MODEL_NAMES, REDUCERS, models_for_task
from .polynomial_order import ORDER_CRITERIA
from .pooling import FRAME_NORMALIZATIONS, POOLING_STATISTICS, constant_statistics
from .splitting import TASKS
from .stepwise import STEPWISE_STARTS

#: Image formats accepted by ``output.figure_format``.
FIGURE_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def _choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ConfigError(f"{name} must be strictly positive, got {value!r}")


def _number(name: str, value: object, *, minimum: float | None = None) -> None:
    """Check for a finite real number (not a bool), optionally bounded below (inclusive)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value!r}")


def _integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value!r}")


def _boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")


def _sequence(name: str, value: object) -> tuple[Any, ...]:
    """Return a list or tuple as a tuple, and a single scalar as a one-element tuple."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None or isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a list, got {value!r}")
    return (value,)


def _unique(name: str, values: tuple[Any, ...]) -> None:
    if len(set(values)) != len(values):
        raise ConfigError(f"{name} lists a value twice: {list(values)!r}")


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and what is predicted.

    Attributes
    ----------
    dataset : str
        Dataset index (CSV, see :mod:`voice_severity_bench.dataset`). A placeholder until you
        point it at your own data.
    task : str
        ``regression`` (continuous severity) or ``classification`` (labels).
    target : str
        Column of the index holding the target: a numeric rating such as
        ``severity`` for regression, a class name column such as ``label``
        for classification.
    """

    dataset: str = "data/dataset.csv"
    task: str = "regression"
    target: str = "severity"

    def __post_init__(self) -> None:
        _choice("data.task", self.task, TASKS)
        if not isinstance(self.target, str) or not self.target.strip():
            raise ConfigError(f"data.target must name a column, got {self.target!r}")
        if self.target in {"path", "group"}:
            raise ConfigError(f"data.target={self.target!r} is a reserved column name")


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic sustained vowels whose perturbations grow with a severity rating.

    Attributes
    ----------
    n_speakers : int
        Number of synthetic speakers (cross-validation groups).
    n_recordings_per_speaker : int
        Recordings per speaker.
    duration_s : float
        Length of each recording in seconds.
    sample_rate : int
        Sampling rate of the written WAV files in Hz.
    f0_range_hz : tuple of float
        Range of the speakers' mean fundamental frequency.
    within_speaker_sd : float
        Standard deviation of a recording's severity around its speaker's
        severity, in rating points (0-100 scale).
    jitter_range : tuple of float
        Relative cycle-to-cycle period perturbation at severity 0 and 100.
    shimmer_range : tuple of float
        Relative cycle-to-cycle amplitude perturbation at severity 0 and 100.
    hnr_range_db : tuple of float
        Harmonics-to-noise ratio at severity 0 and 100, in dB.
    label_threshold : float
        Recordings at or above this severity are labeled ``dysphonic``,
        the others ``healthy``.
    """

    n_speakers: int = 20
    n_recordings_per_speaker: int = 3
    duration_s: float = 1.0
    sample_rate: int = 16_000
    f0_range_hz: tuple[float, float] = (90.0, 240.0)
    within_speaker_sd: float = 6.0
    jitter_range: tuple[float, float] = (0.002, 0.03)
    shimmer_range: tuple[float, float] = (0.02, 0.3)
    hnr_range_db: tuple[float, float] = (30.0, 6.0)
    label_threshold: float = 35.0

    def __post_init__(self) -> None:
        _integer("synthetic.n_speakers", self.n_speakers, minimum=2)
        _integer("synthetic.n_recordings_per_speaker", self.n_recordings_per_speaker, minimum=1)
        _number("synthetic.duration_s", self.duration_s, minimum=0.1)
        _integer("synthetic.sample_rate", self.sample_rate, minimum=8000)
        _number("synthetic.within_speaker_sd", self.within_speaker_sd, minimum=0.0)
        _number("synthetic.label_threshold", self.label_threshold, minimum=0.0)
        if not 0.0 < self.label_threshold < 100.0:
            raise ConfigError(
                f"synthetic.label_threshold must lie inside (0, 100), got {self.label_threshold!r}"
            )
        bounds = {
            "synthetic.f0_range_hz": (self.f0_range_hz, 20.0, self.sample_rate / 4.0),
            "synthetic.jitter_range": (self.jitter_range, 0.0, 0.2),
            "synthetic.shimmer_range": (self.shimmer_range, 0.0, 1.0),
            "synthetic.hnr_range_db": (self.hnr_range_db, -20.0, 80.0),
        }
        for name, (pair, low, high) in bounds.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ConfigError(f"{name} must be a pair of numbers, got {pair!r}")
            for value in pair:
                _number(name, value)
                if not low <= value <= high:
                    raise ConfigError(
                        f"{name} values must lie in [{low:g}, {high:g}], got {pair!r}"
                    )
        if not self.f0_range_hz[0] < self.f0_range_hz[1]:
            raise ConfigError(
                "synthetic.f0_range_hz must be (low, high) with low < high, "
                f"got {self.f0_range_hz!r}"
            )

    @property
    def n_recordings(self) -> int:
        """Total number of recordings."""
        return self.n_speakers * self.n_recordings_per_speaker


@dataclass(frozen=True)
class AudioConfig:
    """How recordings are read before feature extraction.

    Attributes
    ----------
    sample_rate : int or None
        Resample every file to this rate (Hz). ``None`` keeps each file's own
        rate; frame sizes then follow it, and a warning is logged when the
        dataset mixes rates.
    trim_s : float
        Seconds removed from both ends of every recording (0 keeps it whole).
    trim_silence : bool
        Cut leading and trailing silence with :func:`speechdsp.trim_silence`.
    rms_dbfs : float or None
        Scale every recording, after cropping and trimming, to this RMS level
        in dB relative to full scale (``None`` keeps the recorded level). Level
        differences between recording sessions then cannot reach ``c0``.
    """

    sample_rate: int | None = None
    trim_s: float = 0.0
    trim_silence: bool = False
    rms_dbfs: float | None = None

    def __post_init__(self) -> None:
        if self.sample_rate is not None:
            _integer("audio.sample_rate", self.sample_rate, minimum=1000)
        _number("audio.trim_s", self.trim_s, minimum=0.0)
        _boolean("audio.trim_silence", self.trim_silence)
        if self.rms_dbfs is not None:
            _number("audio.rms_dbfs", self.rms_dbfs)
            if self.rms_dbfs >= 0.0:
                raise ConfigError(f"audio.rms_dbfs must be below 0 dBFS, got {self.rms_dbfs!r}")


@dataclass(frozen=True)
class FrontendConfig:
    """Cepstral front end (see :mod:`voice_severity_bench.cepstral_frontend`).

    The defaults use the filterbank of the ETSI distributed speech recognition
    front end (23 filters from 64 Hz), 30 ms frames every 15 ms, 12 HTK-scaled
    cepstra plus ``c0`` and both dynamic blocks: 39 dimensions per frame.

    Attributes
    ----------
    frame_ms, hop_ms : float
        Frame length and advance in milliseconds, converted with the file's
        own sampling rate.
    n_fft : int or None
        FFT size; ``None`` uses the next power of two that holds a frame.
    window : str
        ``hamming_symmetric``, ``hamming`` (periodic), ``hann`` or ``rectangular``.
    preemphasis : float
        Pre-emphasis coefficient (0 disables it).
    n_mels : int
        Number of triangular mel filters.
    fmin_hz, fmax_hz : float
        Edges of the filterbank; ``fmax_hz = None`` means the Nyquist frequency.
    n_ceps : int
        Cepstral coefficients ``c1 .. cM`` kept (``c0`` excluded).
    energy_term : str
        ``c0``, ``log_energy`` or ``none``; appended after the cepstra.
    lifter : int
        Sinusoidal lifter length (0 disables liftering).
    log_floor : float
        Floor applied to filterbank energies before the logarithm.
    delta_order : int
        0 static only, 1 adds deltas, 2 adds deltas and delta-deltas.
    delta_widths : tuple of int
        Odd regression window lengths of the delta and the delta-delta.
    """

    frame_ms: float = 30.0
    hop_ms: float = 15.0
    n_fft: int | None = None
    window: str = "hamming_symmetric"
    preemphasis: float = 0.97
    n_mels: int = 23
    fmin_hz: float = 64.0
    fmax_hz: float | None = None
    n_ceps: int = 12
    energy_term: str = "c0"
    lifter: int = 22
    log_floor: float = 1e-10
    delta_order: int = 2
    delta_widths: tuple[int, int] = (7, 5)

    def __post_init__(self) -> None:
        _number("frontend.frame_ms", self.frame_ms)
        _positive("frontend.frame_ms", self.frame_ms)
        _number("frontend.hop_ms", self.hop_ms)
        _positive("frontend.hop_ms", self.hop_ms)
        if self.n_fft is not None:
            _integer("frontend.n_fft", self.n_fft, minimum=16)
        _choice("frontend.window", self.window, WINDOWS)
        _number("frontend.preemphasis", self.preemphasis, minimum=0.0)
        if self.preemphasis >= 1.0:
            raise ConfigError(f"frontend.preemphasis must be below 1, got {self.preemphasis!r}")
        _integer("frontend.n_mels", self.n_mels, minimum=2)
        _number("frontend.fmin_hz", self.fmin_hz, minimum=0.0)
        if self.fmax_hz is not None:
            _number("frontend.fmax_hz", self.fmax_hz)
            if self.fmax_hz <= self.fmin_hz:
                raise ConfigError(
                    f"frontend.fmax_hz={self.fmax_hz!r} must exceed "
                    f"frontend.fmin_hz={self.fmin_hz!r}"
                )
        _integer("frontend.n_ceps", self.n_ceps, minimum=1)
        if self.n_ceps >= self.n_mels:
            raise ConfigError(
                f"frontend.n_ceps={self.n_ceps} must be smaller than frontend.n_mels={self.n_mels} "
                "(c0 and c1..cM come from n_mels filters)"
            )
        _choice("frontend.energy_term", self.energy_term, ENERGY_TERMS)
        _integer("frontend.lifter", self.lifter, minimum=0)
        _number("frontend.log_floor", self.log_floor)
        _positive("frontend.log_floor", self.log_floor)
        if self.delta_order not in DELTA_ORDERS or isinstance(self.delta_order, bool):
            raise ConfigError(
                f"frontend.delta_order={self.delta_order!r} is not supported; "
                f"choose one of {list(DELTA_ORDERS)}"
            )
        widths = self.delta_widths
        if (
            not isinstance(widths, tuple)
            or len(widths) != 2
            or any(
                isinstance(w, bool) or not isinstance(w, int) or w < 3 or w % 2 == 0 for w in widths
            )
        ):
            raise ConfigError(
                f"frontend.delta_widths must be two odd integers >= 3, got {widths!r}"
            )

    @property
    def n_static(self) -> int:
        """Static dimensions: the cepstra plus the energy term, if any."""
        return self.n_ceps + (self.energy_term != "none")

    @property
    def n_dims(self) -> int:
        """Dimensions per frame after the dynamic blocks (13, 26 or 39 by default)."""
        return self.n_static * (self.delta_order + 1)


@dataclass(frozen=True)
class PoolingConfig:
    """Utterance-level statistics (see :mod:`voice_severity_bench.pooling`).

    Attributes
    ----------
    statistics : tuple of str
        Any of ``mean``, ``std``, ``min``, ``max``, ``percentile``, ``skew``
        and ``kurtosis``, in output order.
    percentiles : tuple of float
        Levels in percent used by the ``percentile`` statistic.
    normalization : str
        ``none``, ``cmn`` or ``cmvn``, applied per recording before pooling.
    """

    statistics: tuple[str, ...] = ("mean", "std")
    percentiles: tuple[float, ...] = (10.0, 50.0, 90.0)
    normalization: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistics", _sequence("pooling.statistics", self.statistics))
        object.__setattr__(self, "percentiles", _sequence("pooling.percentiles", self.percentiles))
        _check_statistics("pooling.statistics", self.statistics)
        if not self.percentiles:
            raise ConfigError("pooling.percentiles must list at least one level")
        for value in self.percentiles:
            _number("pooling.percentiles", value)
            if not 0.0 <= value <= 100.0:
                raise ConfigError(f"pooling.percentiles must lie in [0, 100], got {value!r}")
        _unique("pooling.percentiles", self.percentiles)
        _choice("pooling.normalization", self.normalization, FRAME_NORMALIZATIONS)
        constant = constant_statistics(self.normalization, self.statistics)
        if constant:
            raise ConfigError(
                f"pooling.statistics {constant} would be identical for every recording after "
                f"pooling.normalization={self.normalization!r}"
            )


def _check_statistics(name: str, statistics: tuple[Any, ...]) -> None:
    if not statistics:
        raise ConfigError(f"{name} must list at least one statistic")
    for value in statistics:
        _choice(name, value, POOLING_STATISTICS)
    _unique(name, statistics)


@dataclass(frozen=True)
class AnalysisConfig:
    """Statistical analysis: inference, stepwise selection, polynomial order, correlations.

    Attributes
    ----------
    ci_level : float
        Confidence level of coefficient and correlation intervals.
    correction : str
        Multiple-comparison adjustment of p-values: ``holm`` or ``fdr_bh``.
    p_enter, p_remove : float
        Partial F-test thresholds of stepwise selection (``p_enter <= p_remove``).
    stepwise_start : str
        ``empty`` (forward start) or ``full`` (backward start).
    max_degree : int
        Highest polynomial degree of the order sweep.
    polynomial_basis : str
        ``orthogonal``, ``centered`` or ``raw``.
    order_criterion : str
        How the order sweep recommends a degree: ``kfold_rmse``,
        ``loocv_rmse``, ``bic`` or ``aic``.
    n_resamples : int
        Bootstrap resamples of the confidence intervals in ``compare``.
    """

    ci_level: float = 0.95
    correction: str = "holm"
    p_enter: float = 0.05
    p_remove: float = 0.10
    stepwise_start: str = "empty"
    max_degree: int = 8
    polynomial_basis: str = "orthogonal"
    order_criterion: str = "kfold_rmse"
    n_resamples: int = 10_000

    def __post_init__(self) -> None:
        for name, value in (
            ("analysis.ci_level", self.ci_level),
            ("analysis.p_enter", self.p_enter),
            ("analysis.p_remove", self.p_remove),
        ):
            _number(name, value)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{name} must lie inside (0, 1), got {value!r}")
        if self.p_enter > self.p_remove:
            raise ConfigError(
                f"analysis.p_enter={self.p_enter!r} must not exceed "
                f"analysis.p_remove={self.p_remove!r}, or stepwise selection can cycle"
            )
        _choice("analysis.correction", self.correction, P_VALUE_CORRECTIONS)
        _choice("analysis.stepwise_start", self.stepwise_start, STEPWISE_STARTS)
        _integer("analysis.max_degree", self.max_degree, minimum=1)
        _choice("analysis.polynomial_basis", self.polynomial_basis, POLYNOMIAL_BASES)
        _choice("analysis.order_criterion", self.order_criterion, ORDER_CRITERIA)
        _integer("analysis.n_resamples", self.n_resamples, minimum=100)


@dataclass(frozen=True)
class AblationConfig:
    """The feature-setting grid compared by ``ablate``.

    Each field is one dimension of the grid; an empty list keeps the value of
    the ``frontend``, ``pooling`` or ``model`` section. The first value of
    every list forms the baseline that the other settings are tested against.

    Attributes
    ----------
    n_ceps, n_mels, delta_order, energy_term, lifter : tuple
        Front-end values to compare.
    frame_hop_ms : tuple of (float, float)
        ``(frame_ms, hop_ms)`` pairs to compare.
    statistics : tuple of tuple of str
        Pooling statistic sets to compare.
    reducer : tuple of str
        Dimensionality reductions to compare (``none``, ``pca``, ``lda``).
    models : tuple of str
        Estimators to compare.
    n_repeats : int
        Repetitions of the k-fold split (different shuffles, same for every setting).
    metric : str or None
        Metric that ranks settings and is tested against the baseline;
        ``None`` means ``rmse`` for regression and ``uar`` for classification.
    """

    n_ceps: tuple[int, ...] = (12, 6, 18)
    n_mels: tuple[int, ...] = ()
    delta_order: tuple[int, ...] = (0, 1, 2)
    energy_term: tuple[str, ...] = ()
    frame_hop_ms: tuple[tuple[float, float], ...] = ()
    lifter: tuple[int, ...] = ()
    statistics: tuple[tuple[str, ...], ...] = ()
    reducer: tuple[str, ...] = ()
    models: tuple[str, ...] = ("ridge", "svr")
    n_repeats: int = 3
    metric: str | None = None

    def __post_init__(self) -> None:
        for key in (
            "n_ceps",
            "n_mels",
            "delta_order",
            "energy_term",
            "frame_hop_ms",
            "lifter",
            "statistics",
            "reducer",
            "models",
        ):
            object.__setattr__(self, key, _sequence(f"ablation.{key}", getattr(self, key)))
        for value in self.n_ceps:
            _integer("ablation.n_ceps", value, minimum=1)
        for value in self.n_mels:
            _integer("ablation.n_mels", value, minimum=2)
        for value in self.delta_order:
            if isinstance(value, bool) or value not in DELTA_ORDERS:
                raise ConfigError(
                    f"ablation.delta_order values must be in {list(DELTA_ORDERS)}, got {value!r}"
                )
        for value in self.energy_term:
            _choice("ablation.energy_term", value, ENERGY_TERMS)
        pairs = []
        for pair in self.frame_hop_ms:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ConfigError(
                    f"ablation.frame_hop_ms entries must be [frame_ms, hop_ms] pairs, got {pair!r}"
                )
            for value in pair:
                _number("ablation.frame_hop_ms", value)
                _positive("ablation.frame_hop_ms", value)
            pairs.append(tuple(float(v) for v in pair))
        object.__setattr__(self, "frame_hop_ms", tuple(pairs))
        for value in self.lifter:
            _integer("ablation.lifter", value, minimum=0)
        sets = []
        for entry in self.statistics:
            stats = entry if isinstance(entry, tuple) else (entry,)
            _check_statistics("ablation.statistics", stats)
            sets.append(stats)
        object.__setattr__(self, "statistics", tuple(sets))
        for value in self.reducer:
            _choice("ablation.reducer", value, REDUCERS)
        for value in self.models:
            _choice("ablation.models", value, MODEL_NAMES)
        for key in ("n_ceps", "n_mels", "delta_order", "energy_term", "frame_hop_ms", "lifter"):
            _unique(f"ablation.{key}", getattr(self, key))
        for key in ("statistics", "reducer", "models"):
            _unique(f"ablation.{key}", getattr(self, key))
        _integer("ablation.n_repeats", self.n_repeats, minimum=1)
        if self.metric is not None:
            _choice(
                "ablation.metric",
                self.metric,
                metric_names("regression", binary=True)
                + metric_names("classification", binary=True),
            )


@dataclass(frozen=True)
class ModelConfig:
    """Estimator, dimensionality reduction and hyperparameters.

    See :mod:`voice_severity_bench.models` for the pipelines.

    Attributes
    ----------
    name : str
        Regression: ``linear``, ``ridge``, ``svr``, ``random_forest``, ``mlp``,
        ``torch_mlp``. Classification: ``logistic``, ``svc``,
        ``random_forest``, ``mlp``, ``torch_mlp``.
    reducer : str
        ``none``, ``pca`` or ``lda`` (classification only), fitted inside each
        training fold after standardization.
    n_components : int or None
        Components kept by the reducer; ``None`` keeps 95 % of the variance
        (PCA) or ``n_classes - 1`` directions (LDA).
    tune : bool
        Choose hyperparameters by an inner cross-validation on each training
        fold (nested cross-validation).
    impute : str
        ``none``, or ``median``: missing feature values are replaced by the
        training-fold median and every column that had gaps gets a 0/1
        missing-value indicator. Needed for measures that can be undefined,
        such as jitter of a recording with too few voiced periods.
    class_weight : str or None
        ``balanced`` weights the classes inversely to their frequency in the
        training fold (``logistic``, ``svc`` and ``random_forest`` only).
    block_selection : bool
        For feature files divided into blocks (for example one block per
        network layer), let the inner loop of ``model.tune`` choose the block
        together with the other hyperparameters. Without blocks it does nothing.
    ridge_alpha : float
        Ridge penalty.
    ridge_alphas : tuple of float
        Ridge penalties searched by ``model.tune``; empty keeps the built-in
        grid (0.01 to 100).
    svr_c, svr_epsilon : float
        Support vector regression penalty and insensitive-tube width (in
        standardized target units).
    logistic_c, svc_c : float
        Inverse regularization strengths of the classifiers.
    n_estimators : int
        Trees of the random forest.
    max_depth : int or None
        Depth limit of the trees (``None``: unlimited).
    hidden_units : tuple of int
        Hidden layer sizes of the multilayer perceptrons.
    activation : str
        ``logistic``, ``relu`` or ``tanh``.
    mlp_alpha : float
        L2 penalty of the perceptrons: ``alpha`` of the scikit-learn ones,
        Adam's ``weight_decay`` for the PyTorch one.
    max_iter : int
        Iteration limit of the iterative scikit-learn estimators.
    early_stopping : bool
        Let the scikit-learn perceptrons hold out 10 % of each training fold
        and stop when that validation score stops improving. Off by default:
        with a few dozen recordings the held-out score is too coarse and stops
        training far too early; regularization (``mlp_alpha``) is tuned instead.
    patience : int
        Epochs without improvement before the scikit-learn perceptrons stop:
        improvement of the validation score with ``early_stopping``, of the
        training loss without it (``n_iter_no_change``). The PyTorch
        perceptron ignores this and ``early_stopping``; it runs ``n_epochs``.
    learning_rate : float
        Initial learning rate of both perceptrons.
    batch_size : int
        Mini-batch size of the PyTorch perceptron.
    n_epochs : int
        Training epochs of the PyTorch perceptron.
    """

    name: str = "ridge"
    reducer: str = "none"
    n_components: int | None = None
    tune: bool = False
    impute: str = "none"
    class_weight: str | None = None
    block_selection: bool = False
    ridge_alpha: float = 1.0
    ridge_alphas: tuple[float, ...] = ()
    svr_c: float = 1.0
    svr_epsilon: float = 0.1
    logistic_c: float = 1.0
    svc_c: float = 1.0
    n_estimators: int = 200
    max_depth: int | None = None
    hidden_units: tuple[int, ...] = (100,)
    activation: str = "logistic"
    mlp_alpha: float = 1e-4
    max_iter: int = 2000
    early_stopping: bool = False
    patience: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 200

    def __post_init__(self) -> None:
        _choice("model.name", self.name, MODEL_NAMES)
        _choice("model.reducer", self.reducer, REDUCERS)
        if self.n_components is not None:
            _integer("model.n_components", self.n_components, minimum=1)
        _boolean("model.tune", self.tune)
        _choice("model.impute", self.impute, IMPUTERS)
        if self.class_weight is not None:
            _choice("model.class_weight", self.class_weight, CLASS_WEIGHTS)
        _boolean("model.block_selection", self.block_selection)
        if self.block_selection and not self.tune:
            raise ConfigError(
                "model.block_selection chooses the block in the inner loop; set model.tune: true"
            )
        _number("model.ridge_alpha", self.ridge_alpha, minimum=0.0)
        object.__setattr__(self, "ridge_alphas", _sequence("model.ridge_alphas", self.ridge_alphas))
        for value in self.ridge_alphas:
            _number("model.ridge_alphas", value)
            _positive("model.ridge_alphas", value)
        _unique("model.ridge_alphas", self.ridge_alphas)
        for name in ("svr_c", "logistic_c", "svc_c", "learning_rate"):
            _number(f"model.{name}", getattr(self, name))
            _positive(f"model.{name}", getattr(self, name))
        _number("model.svr_epsilon", self.svr_epsilon, minimum=0.0)
        _number("model.mlp_alpha", self.mlp_alpha, minimum=0.0)
        _integer("model.n_estimators", self.n_estimators, minimum=1)
        if self.max_depth is not None:
            _integer("model.max_depth", self.max_depth, minimum=1)
        object.__setattr__(self, "hidden_units", _sequence("model.hidden_units", self.hidden_units))
        if not self.hidden_units:
            raise ConfigError("model.hidden_units must list at least one layer size")
        for value in self.hidden_units:
            _integer("model.hidden_units", value, minimum=1)
        _choice("model.activation", self.activation, ACTIVATIONS)
        _integer("model.max_iter", self.max_iter, minimum=1)
        _boolean("model.early_stopping", self.early_stopping)
        _integer("model.patience", self.patience, minimum=1)
        _integer("model.batch_size", self.batch_size, minimum=1)
        _integer("model.n_epochs", self.n_epochs, minimum=1)


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation.

    Attributes
    ----------
    n_splits : int
        Number of folds; stratified for classification, and grouped when the
        index has a ``group`` column.
    n_inner_splits : int
        Folds of the inner loop that tunes hyperparameters (``model.tune``).
    n_repeats : int
        Repetitions of the k-fold split in ``compare``.
    positive_class : str
        Class whose recall is the sensitivity and whose score gives the
        ROC curve in binary classification.
    """

    n_splits: int = 5
    n_inner_splits: int = 3
    n_repeats: int = 3
    positive_class: str = "dysphonic"

    def __post_init__(self) -> None:
        _integer("evaluation.n_splits", self.n_splits, minimum=2)
        _integer("evaluation.n_inner_splits", self.n_inner_splits, minimum=2)
        _integer("evaluation.n_repeats", self.n_repeats, minimum=1)
        if not isinstance(self.positive_class, str) or not self.positive_class:
            raise ConfigError(
                f"evaluation.positive_class must be a class name, got {self.positive_class!r}"
            )


@dataclass(frozen=True)
class OutputConfig:
    """Where and how results are written.

    Attributes
    ----------
    root : str
        Run folders are created here when ``--out`` is not given.
    figure_format : str
        ``png``, ``pdf`` or ``svg``.
    dpi : int
        Resolution of raster figures.
    """

    root: str = "outputs"
    figure_format: str = "png"
    dpi: int = 150

    def __post_init__(self) -> None:
        _choice("output.figure_format", self.figure_format, FIGURE_FORMATS)
        _integer("output.dpi", self.dpi, minimum=1)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "synthetic": SyntheticConfig,
    "audio": AudioConfig,
    "frontend": FrontendConfig,
    "pooling": PoolingConfig,
    "analysis": AnalysisConfig,
    "ablation": AblationConfig,
    "model": ModelConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
}

#: Sections whose values change what ``prepare`` writes.
FEATURE_SECTIONS: tuple[str, ...] = ("audio", "frontend", "pooling")


def _tuplify(value: Any) -> Any:
    """YAML lists become tuples, so configurations stay hashable."""
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value


def _listify(value: Any) -> Any:
    """Tuples become lists again, for YAML and JSON."""
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_listify(item) for item in value]
    return value


def _build_section(name: str, cls: type, raw: Any) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"section {name!r} must be a mapping, got {type(raw).__name__}")
    known = sorted(f.name for f in dataclasses.fields(cls))
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in section {name!r}; valid keys are {known}")
    try:
        return cls(**{key: _tuplify(value) for key, value in raw.items()})
    except TypeError as exc:
        raise ConfigError(f"section {name!r}: {exc}") from None


@dataclass(frozen=True)
class Config:
    """Complete configuration of a run.

    Attributes
    ----------
    seed : int
        Master seed: splits, model initialization and synthetic data derive from it.
    data, synthetic, audio, frontend, pooling, analysis, ablation, model, evaluation, output
        See the section classes.

    Examples
    --------
    >>> config = Config().override({"evaluation.n_splits": 3})
    >>> config.evaluation.n_splits, Config.from_dict(config.to_dict()) == config
    (3, True)
    """

    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {self.seed!r}")
        task = self.data.task
        allowed = models_for_task(task)
        if self.model.name not in allowed:
            raise ConfigError(
                f"model.name={self.model.name!r} does not fit data.task={task!r}; "
                f"choose one of {list(allowed)}"
            )
        if task == "regression" and self.model.reducer == "lda":
            raise ConfigError(
                "model.reducer='lda' needs class labels; "
                "it only works with data.task='classification'"
            )
        if self.model.name == "torch_mlp" and self.model.reducer != "none":
            raise ConfigError(
                "model.name='torch_mlp' standardizes inside the network and needs "
                f"model.reducer='none', got {self.model.reducer!r}"
            )
        # How the ablation grid combines with the other sections is checked by
        # ablation_problems() when 'ablate' expands the grid: a grid written for
        # another task or front end must not stop prepare, evaluate or train.

    def ablation_problems(self) -> list[str]:
        """Ways in which the ``ablation`` grid does not fit the other sections.

        Returns
        -------
        list of str
            One message per problem, each naming the dotted keys involved;
            empty when every grid point is a valid configuration.

        Examples
        --------
        >>> Config().ablation_problems()
        []
        >>> problems = Config().override({"frontend.n_mels": 16}).ablation_problems()
        >>> len(problems), "ablation.n_ceps" in problems[0]
        (1, True)
        """
        task = self.data.task
        allowed = models_for_task(task)
        problems = [
            f"ablation.models lists {name!r}, which does not fit data.task={task!r}; "
            f"choose from {list(allowed)}"
            for name in self.ablation.models
            if name not in allowed
        ]
        if task == "regression" and "lda" in self.ablation.reducer:
            problems.append(
                "ablation.reducer 'lda' needs class labels; "
                "it only works with data.task='classification'"
            )
        if self.ablation.metric is not None:
            names = metric_names(task, binary=True)
            if self.ablation.metric not in names:
                problems.append(
                    f"ablation.metric={self.ablation.metric!r} does not fit data.task={task!r}; "
                    f"choose one of {list(names)}"
                )
        for stats in self.ablation.statistics:
            constant = constant_statistics(self.pooling.normalization, stats)
            if constant:
                problems.append(
                    f"ablation.statistics {list(stats)} includes {constant}, which would be "
                    f"identical for every recording after pooling.normalization="
                    f"{self.pooling.normalization!r}"
                )
        largest_ceps = max(self.ablation.n_ceps or (self.frontend.n_ceps,))
        fewest_mels = min(self.ablation.n_mels or (self.frontend.n_mels,))
        if largest_ceps >= fewest_mels:
            problems.append(
                f"ablation.n_ceps / frontend.n_ceps up to {largest_ceps} needs more filters than "
                f"ablation.n_mels / frontend.n_mels down to {fewest_mels}"
            )
        return problems

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Config:
        """Build a configuration from a nested mapping, rejecting unknown keys."""
        data = dict(raw or {})
        known = sorted(f.name for f in dataclasses.fields(cls))
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ConfigError(f"unknown setting(s) {unknown}; valid top-level keys are {known}")
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name in _SECTIONS:
                kwargs[name] = _build_section(name, _SECTIONS[name], value)
            else:
                kwargs[name] = _tuplify(value)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file; keys it leaves out keep their defaults."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"configuration file not found: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{source} is not valid YAML: {exc}") from None
        if raw is not None and not isinstance(raw, Mapping):
            raise ConfigError(f"{source} must contain a mapping of settings")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested mapping (lists instead of tuples), ready for YAML or JSON."""
        return _listify(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration as YAML and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
        return destination

    def override(self, updates: Mapping[str, Any]) -> Config:
        """Return a validated copy with dotted keys replaced, e.g. ``{"output.dpi": 300}``."""
        tree = self.to_dict()
        for dotted, value in updates.items():
            *parents, leaf = dotted.split(".")
            node = tree
            for part in parents:
                child = node.get(part)
                if not isinstance(child, dict):
                    valid = sorted(key for key, item in node.items() if isinstance(item, dict))
                    raise ConfigError(
                        f"unknown configuration section in {dotted!r}; valid sections are {valid}"
                    )
                node = child
            if leaf not in node:
                raise ConfigError(
                    f"unknown configuration key {dotted!r}; valid keys here are {sorted(node)}"
                )
            node[leaf] = value
        return Config.from_dict(tree)
