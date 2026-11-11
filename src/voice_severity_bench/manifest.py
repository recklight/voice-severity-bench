"""Run folders and the ``manifest.json`` that records how a result was produced.

Every command except ``index`` (which only writes ``dataset.csv``) also writes
``manifest.json`` beside its results, recording the command, a UTC time stamp,
the package and library versions, the git commit of the working tree when
there is one, the master seed, the SHA-256 digest and size of every input
file, the complete configuration and any command-specific details. All JSON
goes through :func:`write_json`, so it is reproducible byte for byte: UTF-8,
LF line endings, a final newline, NumPy values as plain numbers and
non-finite floats as ``null``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_LOGGER = logging.getLogger(__name__)

#: Distributions whose versions a manifest records, when they are installed.
_REPORTED_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "speechdsp",
    "torch",
)
_CHUNK_BYTES = 1 << 20


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     sample = Path(folder) / "sample.bin"
    ...     _ = sample.write_bytes(b"abc")
    ...     file_sha256(sample)[:12]
    'ba7816bf8f01'
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: ArrayLike) -> str:
    """SHA-256 of an array's contents, its dtype and shape included."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(directory: str | Path | None = None) -> str | None:
    """Commit hash of the working tree, or ``None`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if directory is None else str(directory),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _library_versions() -> dict[str, str]:
    """Python, platform and installed library versions, without importing anything."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in _REPORTED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _jsonable(value: Any) -> Any:
    """Plain Python values for :mod:`json`; NaN and infinities become ``None``."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON document reproducibly and return its path.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     scores = {"uar": np.float64("nan"), "n_items": np.int64(3)}
    ...     target = write_json(Path(folder) / "scores.json", scores)
    ...     print(target.read_text(encoding="utf-8"), end="")
    {
      "uar": null,
      "n_items": 3
    }
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
    destination.write_text(text + "\n", encoding="utf-8", newline="\n")
    return destination


def build_manifest(
    *,
    command: str,
    config: Mapping[str, Any],
    inputs: Iterable[str | Path] = (),
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest of one run.

    Parameters
    ----------
    command : str
        The command that produced the results.
    config : mapping
        Complete configuration, as returned by ``Config.to_dict()``.
    inputs : iterable of str or pathlib.Path, optional
        Input files to fingerprint; a missing file is recorded as missing.
    seed : int, optional
        Master seed.
    extra : mapping, optional
        Command-specific details, stored under ``details``.

    Returns
    -------
    dict
        JSON-ready manifest whose keys are always in the same order.
    """
    from . import __version__  # deferred: the package __init__ imports the modules

    files = []
    for item in inputs:
        path = Path(item)
        entry: dict[str, Any] = {"path": path.as_posix()}
        if path.is_file():
            entry.update(sha256=file_sha256(path), bytes=path.stat().st_size)
        else:
            entry["missing"] = True
        files.append(entry)
    return _jsonable(
        {
            "command": command,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_version": __version__,
            "git_commit": git_commit(Path.cwd()),
            "seed": seed,
            "versions": _library_versions(),
            "inputs": files,
            "config": dict(config),
            "details": dict(extra or {}),
        }
    )


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest built by :func:`build_manifest` and return its path."""
    destination = write_json(path, manifest)
    _LOGGER.debug("manifest written to %s", destination)
    return destination


def run_directory(root: str | Path, label: str) -> Path:
    """Create ``<root>/<label>_<UTC time stamp>``, adding ``_1``, ``_2`` ... if it exists."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{label}_{stamp}"
    candidate, counter = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{counter}")
        counter += 1
    candidate.mkdir(parents=True)
    return candidate
