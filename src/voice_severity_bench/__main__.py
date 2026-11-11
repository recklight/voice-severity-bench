"""Run the command-line interface with ``python -m voice_severity_bench``.

Equivalent to the ``voice-severity-bench`` console script, and usable straight from a
checkout without installing (``PYTHONPATH=src``).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess test
    main()
