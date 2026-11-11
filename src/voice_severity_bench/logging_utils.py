"""Logging setup for the command line and the example scripts.

The modules only call ``logging.getLogger(__name__)``. Where the records go is
decided by :func:`configure_logging`, which only the command line and the
example scripts call, so a program that imports the package keeps its own
logging setup.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#: Layout of every record written by the handlers installed here.
LOG_FORMAT: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
#: Time stamp layout used by :data:`LOG_FORMAT`.
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

#: Attribute that marks the handlers installed by :func:`configure_logging`.
_OWN_HANDLER = "_voice_severity_bench_handler"


def configure_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Send log records to standard error, and optionally to a file.

    Calling it again replaces the handlers of the previous call instead of
    stacking duplicates; handlers installed by anything else are left alone.

    Parameters
    ----------
    level : str or int, optional
        ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` or ``"CRITICAL"``
        (any case), or a numeric level.
    log_file : str or pathlib.Path, optional
        Also append every record to this file (UTF-8).

    Raises
    ------
    ValueError
        For an unknown level name.
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(
                f"unknown logging level {level!r}; use DEBUG, INFO, WARNING, ERROR or CRITICAL"
            )
    else:
        resolved = int(level)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _OWN_HANDLER, False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _OWN_HANDLER, True)
        root.addHandler(handler)
    root.setLevel(resolved)
