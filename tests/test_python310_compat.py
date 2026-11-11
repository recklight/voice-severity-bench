"""The Python 3.10 gate, plus the packaging rules every release keeps.

The package declares ``requires-python = ">=3.10"`` but is developed on a newer
interpreter, so a passing test suite proves nothing about 3.10 on its own.
Every module is parsed with the 3.10 grammar, and executable code (not
comments or strings) is scanned for names that only exist in Python 3.11+, in
NumPy 2.x, or in releases newer than the declared dependency floors.
"""

from __future__ import annotations

import ast
import importlib
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

PACKAGE = "voice_severity_bench"
#: Requirements whose floors must stay installable on Python 3.10.
FLOORS = (
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "pandas>=1.5",
    "matplotlib>=3.6",
    "pyyaml>=6.0",
    "typer>=0.12",
    "joblib>=1.2",
)
#: Modules that ``import <package>`` and its command line must not load.
OPTIONAL_MODULES = ("torch",)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
_PROSE_TOKENS = frozenset({"COMMENT", "STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"})

FORBIDDEN_STDLIB: dict[str, str] = {
    "tomllib": "a YAML reader (3.11+)",
    "typing.Self": "the class name as a string (3.11+)",
    "LiteralString": "str (3.11+)",
    "assert_never": "raise AssertionError (3.11+)",
    "typing.Unpack": "explicit parameters (3.11+)",
    "TypeVarTuple": "a plain TypeVar (3.11+)",
    "StrEnum": "class Foo(str, Enum) (3.11+)",
    "ReprEnum": "Enum (3.11+)",
    "datetime.UTC": "datetime.timezone.utc (3.11+)",
    "TaskGroup": "asyncio.gather (3.11+)",
    "ExceptionGroup": "a single exception (3.11+)",
    "contextlib.chdir": "os.chdir in try/finally (3.11+)",
    "file_digest": "hash the file in chunks (3.11+)",
    "math.cbrt": "np.cbrt (3.11+)",
    "getmembers_static": "inspect.getmembers (3.11+)",
    "typing.override": "a plain method (3.12+)",
    "itertools.batched": "an explicit slice loop (3.12+)",
    "Path.walk": "os.walk (3.12+)",
    "ReadOnly": "a plain annotation (3.13+)",
    "copy.replace": "dataclasses.replace (3.13+)",
}

FORBIDDEN_NUMPY: dict[str, str] = {
    "np.float_": "np.float64",
    "np.int0": "np.intp",
    "np.NaN": "np.nan",
    "np.Inf": "np.inf",
    "np.product": "np.prod",
    "np.alltrue": "np.all",
    "np.round_": "np.round",
    "np.msort": "np.sort",
    "np.trapz": "scipy.integrate.trapezoid",
    "np.in1d": "np.isin",
    "np.row_stack": "np.vstack",
    "np.asfarray": "np.asarray(x, dtype=np.float64)",
    "np.cast": "ndarray.astype",
    "np.find_common_type": "np.result_type",
    "np.trapezoid": "scipy.integrate.trapezoid (NumPy 2.0+ only)",
    "np.concat": "np.concatenate (NumPy 2.0+ only)",
    "np.unique_values": "np.unique (NumPy 2.0+ only)",
    "np.unique_counts": "np.unique(..., return_counts=True) (NumPy 2.0+ only)",
    "np.unique_inverse": "np.unique(..., return_inverse=True) (NumPy 2.0+ only)",
    "np.unique_all": "np.unique with return flags (NumPy 2.0+ only)",
}

#: Names that ``from <module> import <name>`` cannot find on Python 3.10.
FORBIDDEN_FROM_IMPORTS: dict[str, frozenset[str]] = {
    "typing": frozenset(
        {"Self", "LiteralString", "assert_never", "Unpack", "TypeVarTuple", "override", "ReadOnly"}
    ),
    "enum": frozenset({"StrEnum", "ReprEnum"}),
    "datetime": frozenset({"UTC"}),
    "contextlib": frozenset({"chdir"}),
    "hashlib": frozenset({"file_digest"}),
    "itertools": frozenset({"batched"}),
    "copy": frozenset({"replace"}),
    "asyncio": frozenset({"TaskGroup"}),
}

FORBIDDEN_LIBRARY: dict[str, str] = {
    "root_mean_squared_error": "np.sqrt(mean_squared_error(...)) (scikit-learn 1.4+)",
    "squared=False": "np.sqrt(mean_squared_error(...)) (removed in scikit-learn 1.6)",
    "TargetEncoder": "a plain encoder (scikit-learn 1.3+)",
    "ShortTimeFFT": "scipy.signal.stft (SciPy 1.12+)",
}


def code_only(source: str) -> str:
    """Strip comments and string literals so documentation is not mistaken for code."""
    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if tokenize.tok_name.get(token.type, "") not in _PROSE_TOKENS:
            pieces.append(token.string)
    return " ".join(pieces)


def python_files() -> list[Path]:
    files = sorted(PROJECT_ROOT.glob("*.py"))
    for folder in ("src", "tests", "examples"):
        files.extend(sorted((PROJECT_ROOT / folder).rglob("*.py")))
    return [path for path in files if "__pycache__" not in path.parts]


def files_to_scan() -> list[Path]:
    """Every module except this one, which spells out the forbidden names on purpose."""
    this_file = Path(__file__).resolve()
    return [path for path in python_files() if path.resolve() != this_file]


def offences(path: Path, table: dict[str, str]) -> list[str]:
    source = code_only(path.read_text(encoding="utf-8"))
    return [
        f"{name} -> {fix}"
        for name, fix in table.items()
        if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", source.replace(" . ", "."))
    ]


def test_there_is_something_to_check():
    assert len(python_files()) > 10
    assert len(files_to_scan()) == len(python_files()) - 1


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_module_parses_under_the_python_310_grammar(path):
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
    except SyntaxError as exc:
        pytest.fail(f"{path.name} does not parse as Python 3.10: {exc.msg} (line {exc.lineno})")


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_standard_library_name_newer_than_310(path):
    found = offences(path, FORBIDDEN_STDLIB)
    assert not found, f"{path.name}: " + "; ".join(found)


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_from_import_of_a_name_newer_than_310(path):
    found = [
        f"from {node.module} import {alias.name}"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_FROM_IMPORTS
        for alias in node.names
        if alias.name in FORBIDDEN_FROM_IMPORTS[node.module]
    ]
    assert not found, f"{path.name}: " + "; ".join(found)


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_numpy_name_outside_the_1_24_to_2_x_overlap(path):
    found = offences(path, FORBIDDEN_NUMPY)
    source = code_only(path.read_text(encoding="utf-8")).replace(" . ", ".")
    if re.search(r"(?<!\bnp)(?<!\bnumpy)\.ptp\s*\(", source):
        found.append("ndarray.ptp() -> np.ptp(x)")
    assert not found, f"{path.name}: " + "; ".join(found)


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_library_api_newer_than_the_declared_floors(path):
    source = code_only(path.read_text(encoding="utf-8")).replace(" ", "")
    found = [
        f"{name} -> {fix}"
        for name, fix in FORBIDDEN_LIBRARY.items()
        if name.replace(" ", "") in source
    ]
    assert not found, f"{path.name}: " + "; ".join(found)


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_pep695_generics_or_type_aliases(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        assert getattr(node, "type_params", []) == [], f"{path.name}: PEP 695 type parameters"
        assert type(node).__name__ != "TypeAlias", f"{path.name}: 'type X = ...' statement"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_module_defers_annotation_evaluation(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    ), f"{path.name} lacks 'from __future__ import annotations'"


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "src").rglob("*.py")), ids=lambda p: p.name)
def test_no_slotted_dataclasses(path):
    """Frozen dataclasses with slots do not unpickle on early 3.10 releases."""
    source = code_only(path.read_text(encoding="utf-8")).replace(" ", "")
    assert "slots=True" not in source, f"{path.name}: dataclass with slots=True"


def test_pyproject_targets_python_310_with_a_pep_639_license():
    assert 'requires-python = ">=3.10"' in PYPROJECT
    assert 'target-version = "py310"' in PYPROJECT
    assert 'license = "MIT"' in PYPROJECT
    assert 'license-files = ["LICENSE"]' in PYPROJECT
    assert '"setuptools>=77"' in PYPROJECT
    assert "License ::" not in PYPROJECT
    assert 'authors = [{ name = "RL" }]' in PYPROJECT
    assert "Copyright (c) RL" in (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")


def test_dependency_floors_are_installable_on_python_310():
    for requirement in FLOORS:
        assert f'"{requirement}"' in PYPROJECT, requirement
    assert "numpy>=2" not in PYPROJECT


def test_speechdsp_is_a_plain_pypi_requirement():
    assert '"speechdsp>=0.2.0"' in PYPROJECT
    for banned in ("tool.uv", "git+", "file://", "../"):
        assert banned not in PYPROJECT, banned


def test_version_is_the_same_everywhere():
    declared = re.search(r'^version = "([^"]+)"$', PYPROJECT, re.MULTILINE).group(1)
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cited = re.search(r'^version: "([^"]+)"$', citation, re.MULTILINE).group(1)
    released = re.search(r"^date-released: (\S+)$", citation, re.MULTILINE).group(1)
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert importlib.import_module(PACKAGE).__version__ == declared == cited
    assert f"## [{declared}] - {released}" in changelog


def test_importing_the_package_loads_no_optional_dependency():
    probe = (
        f"import sys, {PACKAGE}, {PACKAGE}.cli;"
        f"print(','.join(m for m in {OPTIONAL_MODULES!r} if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src")),
        check=True,
    )
    assert completed.stdout.strip() == ""
