"""Exceptions raised by the package.

Every expected failure is a ``ValueError`` (or ``FileNotFoundError`` for a
missing input, ``ImportError`` for a missing optional dependency), so the
command line catches exactly those and turns them into a one-line message.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class ConfigError(ValueError):
    """Raised when a setting is unknown, missing or out of range."""


class DatasetError(ValueError):
    """Raised when a dataset index or one of its files cannot be used."""


class ModelFileError(ValueError):
    """Raised when a model file is incomplete or was written in another format."""


class DesignMatrixError(ValueError):
    """Raised when a regression design cannot be estimated (rank deficient or too few rows)."""


class MissingDependencyError(ImportError):
    """Raised when a feature needs an optional dependency that is not installed."""


def require_module(name: str, *, extra: str, feature: str) -> ModuleType:
    """Import an optional dependency or explain which extra provides it."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(
            f'{feature} needs {name}: pip install -e ".[{extra}]" in the project folder, '
            f"or pip install {name}"
        ) from exc
