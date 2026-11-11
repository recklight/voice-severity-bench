"""High-level steps shared by the command line and the examples.

``prepare`` reads every recording of a dataset index, computes its frame-level
cepstral features (:mod:`voice_severity_bench.cepstral_frontend`) and pools
them into one vector (:mod:`voice_severity_bench.pooling`); ``train`` fits the
configured pipeline on all items; ``predict`` repeats exactly the feature
extraction stored in a model file on new recordings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .cepstral_frontend import cepstral_features, frame_features
from .config import Config
from .corpus import read_audio
from .dataset import FeatureSet, index_intervals, load_dataset_index
from .errors import DatasetError, ModelFileError
from .evaluation import class_scores
from .models import fit_model
from .pooling import pool_frames
from .splitting import has_groups

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

_LOGGER = logging.getLogger(__name__)


def utterance_features(signal: ArrayLike, sample_rate: int, config: Config) -> NDArray[np.float64]:
    """Pooled feature vector of one (already loaded) recording.

    Parameters
    ----------
    signal : array_like, shape (n_samples,)
        Mono waveform.
    sample_rate : int
        Its sampling rate in Hz.
    config : Config
        ``frontend`` and ``pooling`` settings.

    Returns
    -------
    numpy.ndarray, shape (n_features,)
        The pooled vector, named by :func:`~voice_severity_bench.pooling.pooled_feature_names`.

    Raises
    ------
    ValueError
        If the recording is shorter than one analysis frame.

    Examples
    --------
    >>> rate = 16_000
    >>> tone = np.sin(2 * np.pi * 150.0 * np.arange(rate // 2) / rate)
    >>> utterance_features(tone, rate, Config()).shape
    (78,)
    """
    frames = frame_features(signal, sample_rate, config.frontend)
    if frames.shape[0] == 0:
        raise ValueError(
            f"{np.asarray(signal).size} samples at {sample_rate} Hz are shorter than one "
            f"{config.frontend.frame_ms:g} ms frame"
        )
    return pool_frames(
        frames,
        config.pooling.statistics,
        percentiles=config.pooling.percentiles,
        normalization=config.pooling.normalization,
    )


def recording_frames(
    paths: Sequence[str | Path],
    config: Config,
    *,
    static_only: bool = False,
    intervals: ArrayLike | None = None,
) -> list[NDArray[np.float64]]:
    """Frame-level features of several recordings; any unusable file stops the run.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        WAV files.
    config : Config
        ``audio`` and ``frontend`` settings.
    static_only : bool, optional
        Return only the static block (no deltas), which the ablation study
        caches once and extends with every requested delta order.
    intervals : array_like, shape (n_items, 2), optional
        Start and end in seconds of the part of each file to analyse (NaN for
        an open end), as returned by
        :func:`~voice_severity_bench.dataset.index_intervals`.

    Returns
    -------
    list of numpy.ndarray
        One ``(n_frames, n_dims)`` matrix per recording.

    Raises
    ------
    DatasetError
        If any recording is unreadable or shorter than one frame. The message
        gives the number of such recordings and names the first five.
    """
    frames: list[NDArray[np.float64]] = []
    problems: list[str] = []
    rates: set[int] = set()
    spans = (
        np.full((len(paths), 2), np.nan)
        if intervals is None
        else np.asarray(intervals, dtype=np.float64)
    )
    if spans.shape != (len(paths), 2):
        raise ValueError(f"intervals must have shape ({len(paths)}, 2), got {spans.shape}")
    for index, (path, (start, end)) in enumerate(zip(paths, spans, strict=True), start=1):
        try:
            signal, rate = read_audio(path, config.audio, start_s=start, end_s=end)
            if static_only:
                matrix = cepstral_features(signal, rate, config.frontend)
            else:
                matrix = frame_features(signal, rate, config.frontend)
            if matrix.shape[0] == 0:
                raise ValueError("shorter than one analysis frame")
        except (ValueError, OSError) as exc:
            where = "" if np.isnan(start) and np.isnan(end) else f" [{start:g}-{end:g} s]"
            problems.append(f"{Path(path).name}{where}: {exc}")
            continue
        rates.add(rate)
        frames.append(matrix)
        if index % 200 == 0:
            _LOGGER.info("%d/%d recordings analyzed", index, len(paths))
    if problems:
        shown = "; ".join(problems[:5]) + (
            f" ... and {len(problems) - 5} more" if len(problems) > 5 else ""
        )
        raise DatasetError(f"{len(problems)} recording(s) cannot be analyzed: {shown}")
    if len(rates) > 1:
        _LOGGER.warning(
            "recordings use %d sampling rates %s; set audio.sample_rate to resample them to one",
            len(rates),
            sorted(rates),
        )
    return frames


def pool_recordings(frames: Sequence[NDArray[np.float64]], config: Config) -> NDArray[np.float64]:
    """Stack the pooled vectors of several frame matrices into ``(n_items, n_features)``."""
    return np.vstack(
        [
            pool_frames(
                matrix,
                config.pooling.statistics,
                percentiles=config.pooling.percentiles,
                normalization=config.pooling.normalization,
            )
            for matrix in frames
        ]
    )


def prepare_features(dataset: str | Path, config: Config) -> FeatureSet:
    """Read a dataset index and compute one pooled feature vector per recording.

    Parameters
    ----------
    dataset : str or pathlib.Path
        ``dataset.csv`` (see :mod:`voice_severity_bench.dataset`).
    config : Config
        ``data``, ``audio``, ``frontend`` and ``pooling`` settings.

    Returns
    -------
    FeatureSet
        Features, targets, groups, paths and the configuration used.

    Raises
    ------
    FileNotFoundError, DatasetError
        For a missing index, missing files or recordings that cannot be analyzed.
    """
    index = load_dataset_index(dataset, target=config.data.target, task=config.data.task)
    _LOGGER.info("%d recordings listed in %s", len(index), Path(dataset).as_posix())
    intervals = index_intervals(index)
    frames = recording_frames(index["path"].tolist(), config, intervals=intervals)
    X = pool_recordings(frames, config)
    y = index[config.data.target].to_numpy()
    y = y.astype(np.float64) if config.data.task == "regression" else y.astype(str)
    _LOGGER.info("%d recordings x %d features", X.shape[0], X.shape[1])
    return FeatureSet(
        X=X,
        y=y,
        groups=index["group"].to_numpy().astype(str),
        paths=index["path"].to_numpy().astype(str),
        config=config.to_dict(),
        intervals=intervals,
    )


def train_model(features: FeatureSet, config: Config) -> BaseEstimator:
    """Fit the configured pipeline on every item (tuned by inner cross-validation if asked).

    Parameters
    ----------
    features : FeatureSet
        Output of :func:`prepare_features`.
    config : Config
        Run configuration.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The trained pipeline; store it with :func:`~voice_severity_bench.models.save_model`.
    """
    groups = np.asarray(features.groups).astype(str)
    model = fit_model(
        features.X,
        features.y,
        config,
        groups=groups if has_groups(groups) else None,
    )
    _LOGGER.info("trained %s on %d items", config.model.name, len(features.y))
    return model


def predict_inputs(inputs: Sequence[str | Path], bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Predict new recordings with the settings stored in a model file.

    Parameters
    ----------
    inputs : sequence of str or pathlib.Path
        WAV files.
    bundle : mapping
        Output of :func:`~voice_severity_bench.models.load_model`.

    Returns
    -------
    pandas.DataFrame
        ``path`` and ``predicted``, plus ``score_<class>`` columns for
        classification, one row per input in the given order.
    """
    if not inputs:
        raise ValueError("no input files given")
    if "estimator" not in bundle or "config" not in bundle:
        raise ModelFileError("the model bundle lacks its estimator or configuration")
    config = Config.from_dict(bundle["config"])
    missing = [str(path) for path in inputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"input file(s) not found: {', '.join(missing[:5])}")
    frames = recording_frames([Path(path) for path in inputs], config)
    X = pool_recordings(frames, config)
    estimator = bundle["estimator"]
    table = pd.DataFrame({"path": [Path(path).as_posix() for path in inputs]})
    if config.data.task == "regression":
        table["predicted"] = np.asarray(estimator.predict(X), dtype=np.float64)
    else:
        classes = [str(name) for name in bundle["classes"]]
        scores = class_scores(estimator, X, classes)
        table["predicted"] = np.asarray(estimator.predict(X)).astype(str)
        for column, name in enumerate(classes):
            table[f"score_{name}"] = scores[:, column]
    return table
