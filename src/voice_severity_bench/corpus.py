"""Reading recordings and exporting frame-level features for a whole folder tree.

:func:`read_audio` cuts out an optional interval at the file's own rate and
then applies the ``audio`` settings: resampling to one rate (polyphase
filtering), trimming a fixed number of *seconds* from both ends,
endpoint-based silence trimming from :mod:`speechdsp`, and scaling to a fixed
RMS level.

:func:`extract_corpus` walks a folder, keeps only audio files, computes the
frame-level features of each and mirrors the folder structure in the output
folder. Every file gets a row in the returned table with status ``ok``,
``too_short`` or ``error`` and the reason, so no file is skipped silently.
The output format (``fmt``) is one of:

* ``npz``: the matrix, its column names and the front-end settings;
* ``htk``: the binary HTK parameter file (big-endian header and data, the
  frame period in 100 ns units and ``parmKind`` derived from the actual
  layout, e.g. ``MFCC_0_D_A`` = 8966; Young et al., 2006);
* ``text``: a whitespace-separated matrix with nine significant digits (``%.8e``),
  readable by :func:`read_text_matrix` or any spreadsheet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import speechdsp
from numpy.typing import NDArray
from scipy.signal import resample_poly

from .cepstral_frontend import feature_layout, frame_features, frame_sizes

if TYPE_CHECKING:
    from .config import AudioConfig, Config, FrontendConfig

_LOGGER = logging.getLogger(__name__)

#: File suffixes treated as audio (compared case-insensitively).
AUDIO_SUFFIXES: tuple[str, ...] = (".wav",)
#: Output formats accepted by :func:`extract_corpus`.
FEATURE_FORMATS: tuple[str, ...] = ("npz", "htk", "text")
#: Suffix of each output format.
_FORMAT_SUFFIX: dict[str, str] = {"npz": ".npz", "htk": ".htk", "text": ".txt"}

#: HTK base kind of cepstral features and its qualifier bits.
_HTK_MFCC = 6
_HTK_ENERGY = 0o100
_HTK_DELTA = 0o400
_HTK_ACCELERATION = 0o1000
_HTK_ZEROTH = 0o20000


def iter_audio_files(root: str | Path) -> list[Path]:
    """Audio files below ``root``, sorted by relative POSIX path.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     for name in ("b/two.WAV", "a/one.wav", "a/notes.txt"):
    ...         target = Path(folder) / name
    ...         target.parent.mkdir(parents=True, exist_ok=True)
    ...         _ = target.write_bytes(b"")
    ...     [p.relative_to(folder).as_posix() for p in iter_audio_files(folder)]
    ['a/one.wav', 'b/two.WAV']
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"audio folder not found: {base}")
    files = [
        path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.relative_to(base).as_posix())


def read_audio(
    path: str | Path,
    audio: AudioConfig,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Read a recording as mono float64 and apply the ``audio`` settings.

    Parameters
    ----------
    path : str or pathlib.Path
        WAV file.
    audio : AudioConfig
        Resampling, trimming and level settings.
    start_s, end_s : float, optional
        Interval to keep, in seconds of the file; ``None`` (or NaN) leaves
        that end open. The cut is made before resampling, at the file's own
        rate, so the resampling filter sees no edge inside the recording.

    Returns
    -------
    signal : numpy.ndarray, shape (n_samples,)
        Samples in ``[-1, 1]``.
    sample_rate : int
        Rate of ``signal`` in Hz.

    Raises
    ------
    ValueError
        If the file holds NaN or infinite samples (possible in floating-point
        WAV), the interval lies outside the recording, the recording is
        shorter than the requested trimming, or it is silent and
        ``audio.rms_dbfs`` asks for a level.

    Examples
    --------
    >>> import tempfile
    >>> from voice_severity_bench.config import AudioConfig
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = Path(folder) / "tone.wav"
    ...     speechdsp.write_wav(target, 0.5 * np.sin(np.arange(8_000) / 5.0), 8_000)
    ...     signal, rate = read_audio(target, AudioConfig(sample_rate=16_000, trim_s=0.1))
    >>> signal.shape, rate
    ((12800,), 16000)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = Path(folder) / "tone.wav"
    ...     speechdsp.write_wav(target, 0.5 * np.sin(np.arange(8_000) / 5.0), 8_000)
    ...     audio = AudioConfig(sample_rate=16_000, rms_dbfs=-26.0)
    ...     signal, rate = read_audio(target, audio, start_s=0.25, end_s=0.75)
    >>> signal.shape, round(float(20 * np.log10(np.sqrt(np.mean(signal**2)))), 6)
    ((8000,), -26.0)
    """
    signal, rate = speechdsp.read_wav(path)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{Path(path).name} contains NaN or infinite samples")
    signal = _cut_interval(signal, rate, start_s, end_s, Path(path).name)
    if audio.sample_rate is not None and audio.sample_rate != rate:
        ratio = Fraction(int(audio.sample_rate), int(rate))
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
        rate = int(audio.sample_rate)
    if audio.trim_s > 0.0:
        cut = round(audio.trim_s * rate)
        if signal.size <= 2 * cut:
            raise ValueError(
                f"{Path(path).name} lasts {signal.size / rate:.3f} s, not longer than "
                f"2 x audio.trim_s = {2 * audio.trim_s:g} s"
            )
        signal = signal[cut : signal.size - cut]
    if audio.trim_silence:
        signal = speechdsp.trim_silence(signal, rate)
    if audio.rms_dbfs is not None:
        rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
        if rms == 0.0:
            raise ValueError(f"{Path(path).name} is silent; it cannot be scaled to audio.rms_dbfs")
        signal = signal * (10.0 ** (audio.rms_dbfs / 20.0) / rms)
    return np.asarray(signal, dtype=np.float64), int(rate)


def _cut_interval(
    signal: NDArray[np.float64], rate: int, start_s: float | None, end_s: float | None, name: str
) -> NDArray[np.float64]:
    def given(value: float | None) -> bool:
        return value is not None and not np.isnan(value)

    if not given(start_s) and not given(end_s):
        return signal
    first = round(float(start_s) * rate) if given(start_s) else 0
    last = round(float(end_s) * rate) if given(end_s) else signal.size
    if not 0 <= first < last <= signal.size:
        raise ValueError(
            f"{name}: interval {start_s}-{end_s} s does not fit the recording "
            f"(0-{signal.size / rate:.3f} s)"
        )
    return signal[first:last]


def htk_parameter_kind(frontend: FrontendConfig) -> int:
    """HTK ``parmKind`` code that describes the frame layout.

    Base kind ``MFCC`` (6) plus ``_0`` for ``c0``, ``_E`` for log energy,
    ``_D`` for deltas and ``_A`` for delta-deltas.

    Examples
    --------
    >>> from voice_severity_bench.config import FrontendConfig
    >>> htk_parameter_kind(FrontendConfig()), htk_parameter_kind(FrontendConfig(delta_order=0))
    (8966, 8198)
    """
    kind = _HTK_MFCC
    if frontend.energy_term == "c0":
        kind |= _HTK_ZEROTH
    elif frontend.energy_term == "log_energy":
        kind |= _HTK_ENERGY
    if frontend.delta_order >= 1:
        kind |= _HTK_DELTA
    if frontend.delta_order == 2:
        kind |= _HTK_ACCELERATION
    return kind


def read_text_matrix(path: str | Path) -> NDArray[np.float64]:
    """Read a whitespace-separated numeric matrix (one frame per line).

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = Path(folder) / "frames.txt"
    ...     _ = target.write_text("1.0 2.0\\n3.0 4.0\\n", encoding="utf-8")
    ...     read_text_matrix(target).tolist()
    [[1.0, 2.0], [3.0, 4.0]]
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"feature file not found: {source}")
    return np.loadtxt(source, dtype=np.float64, ndmin=2)


def load_frame_features(path: str | Path) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Read a frame-level feature file written by :func:`extract_corpus` (any format).

    Returns
    -------
    features : numpy.ndarray, shape (n_frames, n_dims)
        The matrix.
    details : dict
        ``feature_names``, ``frontend`` and ``sample_rate`` for ``.npz``
        files, the HTK header for ``.htk`` files, empty for text files.

    Examples
    --------
    >>> import tempfile
    >>> from voice_severity_bench.config import Config
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     rate = 16_000
    ...     tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(rate // 2) / rate)
    ...     speechdsp.write_wav(Path(folder) / "in" / "a.wav", tone, rate)
    ...     _ = extract_corpus(Path(folder) / "in", Path(folder) / "out", Config(), fmt="npz")
    ...     frames, details = load_frame_features(Path(folder) / "out" / "a.npz")
    >>> frames.shape, details["feature_names"][:2], details["sample_rate"]
    ((32, 39), ['c1', 'c2'], 16000)
    """
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            return archive["features"].astype(np.float64), {
                "feature_names": [str(name) for name in archive["feature_names"]],
                "frontend": json.loads(str(archive["frontend_json"])),
                "sample_rate": int(archive["sample_rate"]),
            }
    if source.suffix == ".htk":
        return speechdsp.read_htk(source)
    return read_text_matrix(source), {}


def _write_frames(
    destination: Path, frames: NDArray[np.float64], rate: int, frontend: FrontendConfig, fmt: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        np.savez_compressed(
            destination,
            features=frames,
            feature_names=np.asarray(feature_layout(frontend)),
            frontend_json=np.asarray(json.dumps(_section_dict(frontend), sort_keys=True)),
            sample_rate=np.asarray(rate),
        )
    elif fmt == "htk":
        _, hop, _ = frame_sizes(rate, frontend)
        period = round(hop * 10_000_000 / rate)
        speechdsp.write_htk(
            destination, frames, period_100ns=period, kind=htk_parameter_kind(frontend)
        )
    else:
        np.savetxt(destination, frames, fmt="%.8e", newline="\n")


def _section_dict(section: Any) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(section).items()
    }


def extract_corpus(
    root: str | Path, out_dir: str | Path, config: Config, *, fmt: str = "npz"
) -> pd.DataFrame:
    """Frame-level features of every audio file below ``root``, mirrored under ``out_dir``.

    Parameters
    ----------
    root : str or pathlib.Path
        Folder searched recursively for audio files.
    out_dir : str or pathlib.Path
        Output folder; ``root/a/b.wav`` becomes ``out_dir/a/b.<suffix>``.
    config : Config
        ``audio`` and ``frontend`` settings.
    fmt : {"npz", "htk", "text"}, optional
        Output format.

    Returns
    -------
    pandas.DataFrame
        One row per audio file: ``path``, ``output``, ``sample_rate``,
        ``n_samples``, ``n_frames``, ``n_dims``, ``status`` (``ok``,
        ``too_short`` or ``error``) and ``message``.

    Examples
    --------
    >>> import tempfile
    >>> from voice_severity_bench.config import Config
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     rate = 16_000
    ...     tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(rate // 2) / rate)
    ...     speechdsp.write_wav(Path(folder) / "in" / "spk1" / "a.wav", tone, rate)
    ...     speechdsp.write_wav(Path(folder) / "in" / "blip.wav", tone[:100], rate)
    ...     table = extract_corpus(Path(folder) / "in", Path(folder) / "out", Config(), fmt="htk")
    ...     table[["path", "n_frames", "n_dims", "status"]].values.tolist()
    [['blip.wav', 0, 39, 'too_short'], ['spk1/a.wav', 32, 39, 'ok']]
    """
    if fmt not in FEATURE_FORMATS:
        raise ValueError(f"unknown feature format {fmt!r}; choose one of {list(FEATURE_FORMATS)}")
    source = Path(root)
    target = Path(out_dir)
    files = iter_audio_files(source)
    if not files:
        raise FileNotFoundError(
            f"no audio files ({', '.join(AUDIO_SUFFIXES)}) found under {source}"
        )
    rows = []
    for index, file in enumerate(files, start=1):
        relative = file.relative_to(source)
        output = relative.with_suffix(_FORMAT_SUFFIX[fmt])
        row: dict[str, Any] = {
            "path": relative.as_posix(),
            "output": output.as_posix(),
            "sample_rate": 0,
            "n_samples": 0,
            "n_frames": 0,
            "n_dims": config.frontend.n_dims,
            "status": "ok",
            "message": "",
        }
        try:
            signal, rate = read_audio(file, config.audio)
            frames = frame_features(signal, rate, config.frontend)
            row.update(sample_rate=rate, n_samples=signal.size, n_frames=frames.shape[0])
            if frames.shape[0] == 0:
                row.update(status="too_short", output="", message="shorter than one frame")
            else:
                _write_frames(target / output, frames, rate, config.frontend, fmt)
        except (ValueError, OSError) as exc:
            row.update(status="error", output="", message=str(exc))
            _LOGGER.warning("%s: %s", relative.as_posix(), exc)
        rows.append(row)
        if index % 500 == 0:
            _LOGGER.info("%d/%d files processed", index, len(files))
    table = pd.DataFrame(rows)
    _LOGGER.info(
        "%d of %d files written (%d too short, %d failed)",
        int((table["status"] == "ok").sum()),
        len(table),
        int((table["status"] == "too_short").sum()),
        int((table["status"] == "error").sum()),
    )
    return table
