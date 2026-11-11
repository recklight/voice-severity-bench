"""Dataset index files, utterance-level feature files and analysis tables.

A dataset is described by one CSV index, ``dataset.csv``:

=========  ========  ======================================================
column     required  meaning
=========  ========  ======================================================
path       yes       audio file, relative to the folder holding the CSV
<target>   yes       the rating (regression) or class name (classification)
group      no        speaker or session; folds never split a group
start_s    no        start of the analysed interval in seconds (empty: 0)
end_s      no        end of the interval in seconds (empty: end of file)
=========  ========  ======================================================

Targets and groups are read only from these columns, never parsed from file names.
With ``start_s``/``end_s`` one recording can appear in several rows, for
example its sustained vowel and its sentences; give those rows the same
``group`` so they never end up on both sides of a fold.

``prepare`` turns the index into ``features.npz`` (one pooled feature vector
per row); ``regress``, ``stepwise``, ``order-sweep`` and ``correlate``
read either such a file or any CSV table of numeric columns. Features from
other front ends (clinical voice measures, layers of a speech representation
network) come in through :func:`feature_set_from_table`; such files carry
their own column names, a ``kind`` and optionally column ``blocks``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .config import FEATURE_SECTIONS, Config
from .corpus import iter_audio_files
from .errors import ConfigError, DatasetError
from .pooling import pooled_feature_names

_LOGGER = logging.getLogger(__name__)

#: File name of a dataset index.
DATASET_INDEX_NAME: str = "dataset.csv"
#: ``data`` keys that shape ``features.npz`` besides the feature sections.
_TARGET_KEYS: tuple[str, ...] = ("task", "target")
#: Items an error message lists before "... and N more".
_LISTED = 5
#: Optional interval columns of a dataset index, in seconds.
INTERVAL_COLUMNS: tuple[str, str] = ("start_s", "end_s")
#: ``kind`` of the feature files written by ``prepare``.
MFCC_KIND: str = "mfcc"


def _listing(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_LISTED])
    extra = len(items) - _LISTED
    return shown + (f" ... and {extra} more" if extra > 0 else "")


def load_dataset_index(
    path: str | Path, *, target: str | None = None, task: str = "regression"
) -> pd.DataFrame:
    """Read and validate a dataset index.

    Parameters
    ----------
    path : str or pathlib.Path
        The CSV file.
    target : str, optional
        Column that must be present and complete; numeric for regression.
    task : {"regression", "classification"}, optional
        How the target column is checked.

    Returns
    -------
    pandas.DataFrame
        The index with ``path`` resolved to absolute POSIX paths and a
        ``group`` column (empty strings when the file has none). Interval
        columns, when present, are floats with NaN for an open end.

    Raises
    ------
    FileNotFoundError
        If the index itself does not exist.
    DatasetError
        For missing columns, empty or non-numeric targets, missing audio files,
        invalid intervals, or a recording listed more than once (one recording
        in two rows could end up on both sides of a fold). With interval
        columns a file may repeat, but not with the same interval.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"dataset index not found: {source}. Point data.dataset (or --dataset) at your own "
            "dataset.csv, or run 'voice-severity-bench synthesize' first"
        )
    try:
        table = pd.read_csv(source, dtype={"path": str, "group": str}, keep_default_na=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{source} is not a readable CSV file: {exc}") from None
    required = ["path"] + ([target] if target else [])
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise DatasetError(f"{source} lacks column(s) {missing}; found {list(table.columns)}")
    if table.empty:
        raise DatasetError(f"{source} lists no recordings")
    if table["path"].isna().any():
        raise DatasetError(f"{source} has rows without a path")
    if target:
        empty = table.index[table[target].isna()].tolist()
        if empty:
            raise DatasetError(
                f"{source}: column {target!r} is empty in row(s) "
                f"{_listing([str(i + 2) for i in empty])}"
            )
        if task == "regression":
            numeric = pd.to_numeric(table[target], errors="coerce")
            bad = table.index[numeric.isna()].tolist()
            if bad:
                raise DatasetError(
                    f"{source}: column {target!r} must be numeric for regression; "
                    f"bad value(s) in row(s) {_listing([str(i + 2) for i in bad])}"
                )
            table[target] = numeric.astype(np.float64)
        else:
            table[target] = table[target].astype(str)
    table["group"] = table["group"].fillna("").astype(str) if "group" in table else ""
    _check_intervals(table, source)
    folder = source.resolve().parent
    resolved = [(folder / Path(str(item))).resolve() for item in table["path"]]
    absent = [
        str(item) for item, file in zip(table["path"], resolved, strict=True) if not file.is_file()
    ]
    if absent:
        raise DatasetError(
            f"{len(absent)} file(s) listed in {source} do not exist: {_listing(absent)}"
        )
    intervals = index_intervals(table)
    rows_of: dict[tuple[Path, float | None, float | None], list[int]] = {}
    for row, file in enumerate(resolved, start=2):
        start, end = (None, None) if intervals is None else _interval_key(intervals[row - 2])
        rows_of.setdefault((file, start, end), []).append(row)
    repeated = [
        f"{table['path'].iloc[rows[0] - 2]} (rows {', '.join(map(str, rows))})"
        for rows in rows_of.values()
        if len(rows) > 1
    ]
    if repeated:
        what = "audio file" if intervals is None else "audio file and interval"
        raise DatasetError(f"{source} lists the same {what} more than once: {_listing(repeated)}")
    table["path"] = [file.as_posix() for file in resolved]
    return table


def _check_intervals(table: pd.DataFrame, source: Path) -> None:
    """Validate and convert the optional interval columns in place."""
    for column in INTERVAL_COLUMNS:
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        bad = table.index[values.isna() & table[column].notna()].tolist()
        if bad:
            raise DatasetError(
                f"{source}: column {column!r} must hold seconds; bad value(s) in row(s) "
                f"{_listing([str(i + 2) for i in bad])}"
            )
        negative = table.index[values < 0].tolist()
        if negative:
            raise DatasetError(
                f"{source}: column {column!r} is negative in row(s) "
                f"{_listing([str(i + 2) for i in negative])}"
            )
        table[column] = values.astype(np.float64)
    if all(column in table for column in INTERVAL_COLUMNS):
        start, end = table["start_s"], table["end_s"]
        reversed_rows = table.index[(end <= start.fillna(0.0)) & end.notna()].tolist()
        if reversed_rows:
            raise DatasetError(
                f"{source}: end_s does not lie after start_s in row(s) "
                f"{_listing([str(i + 2) for i in reversed_rows])}"
            )


def index_intervals(table: pd.DataFrame) -> NDArray[np.float64] | None:
    """``(n_rows, 2)`` start and end times of a dataset index; ``None`` without interval columns.

    Examples
    --------
    >>> table = pd.DataFrame({"path": ["a.wav", "a.wav"], "start_s": [0.5, 2.0]})
    >>> index_intervals(table).tolist()
    [[0.5, nan], [2.0, nan]]
    >>> index_intervals(table.drop(columns="start_s")) is None
    True
    """
    if not any(column in table for column in INTERVAL_COLUMNS):
        return None
    columns = [
        table[column].to_numpy(dtype=np.float64) if column in table else np.full(len(table), np.nan)
        for column in INTERVAL_COLUMNS
    ]
    return np.column_stack(columns)


def _interval_key(row: ArrayLike) -> tuple[float | None, float | None]:
    start, end = (float(value) for value in np.asarray(row, dtype=np.float64))
    return (None if math.isnan(start) else start, None if math.isnan(end) else end)


@dataclass(frozen=True, eq=False)
class FeatureSet:
    """Utterance-level model inputs, one row per recording (or recording interval).

    Attributes
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
        Pooled features.
    y : numpy.ndarray, shape (n_items,)
        Ratings (float) or class names (str).
    groups : numpy.ndarray, shape (n_items,)
        Speaker of each recording (empty strings when ungrouped).
    paths : numpy.ndarray, shape (n_items,)
        Audio file of each row.
    config : dict
        The complete configuration ``prepare`` used (``Config.to_dict()``);
        for other front ends at least ``data.task`` and ``data.target``.
    names : tuple of str, optional
        Column names; ``None`` derives them from ``config`` (MFCC files).
    kind : str
        ``mfcc`` for files from ``prepare``; any other label for external
        front ends, whose settings ``load_features`` does not check.
    blocks : dict of str to (int, int), optional
        Half-open column range of every block (e.g. one per network layer).
    intervals : numpy.ndarray, shape (n_items, 2), optional
        Start and end of the analysed interval in seconds, NaN for an open end.
    extractor : dict, optional
        Provenance of an external front end (versions, settings, checksums).
    """

    X: NDArray[np.float64]
    y: NDArray[Any]
    groups: NDArray[np.str_]
    paths: NDArray[np.str_]
    config: dict[str, Any]
    names: tuple[str, ...] | None = None
    kind: str = MFCC_KIND
    blocks: dict[str, tuple[int, int]] | None = None
    intervals: NDArray[np.float64] | None = None
    extractor: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        n_items, n_features = np.shape(self.X)
        for name in ("y", "groups", "paths"):
            if len(getattr(self, name)) != n_items:
                raise DatasetError(
                    f"FeatureSet.{name} has {len(getattr(self, name))} rows, X has {n_items}"
                )
        if self.names is not None:
            object.__setattr__(self, "names", tuple(str(name) for name in self.names))
            if len(self.names) != n_features:
                raise DatasetError(f"{len(self.names)} column names for {n_features} columns")
        if self.blocks is not None:
            blocks = {str(key): (int(a), int(b)) for key, (a, b) in self.blocks.items()}
            bad = [key for key, (a, b) in blocks.items() if not 0 <= a < b <= n_features]
            if bad:
                raise DatasetError(f"block(s) {bad} lie outside the {n_features} columns")
            object.__setattr__(self, "blocks", blocks)
        if self.intervals is not None:
            intervals = np.asarray(self.intervals, dtype=np.float64)
            if intervals.shape != (n_items, 2):
                raise DatasetError(
                    f"intervals must have shape ({n_items}, 2), got {intervals.shape}"
                )
            object.__setattr__(self, "intervals", intervals)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Column names of ``X``, stored or derived from the stored configuration."""
        if self.names is not None:
            return self.names
        if self.kind != MFCC_KIND:
            return tuple(f"x{column}" for column in range(np.shape(self.X)[1]))
        stored = Config.from_dict(self.config)
        return pooled_feature_names(stored.frontend, stored.pooling)

    def keys(self) -> list[tuple[str, float | None, float | None]]:
        """``(path, start_s, end_s)`` of every row, ``None`` for an open end.

        Rows of two feature files describe the same item when their keys are
        equal; :func:`~voice_severity_bench.comparison.align_feature_sets` relies on it.
        """
        if self.intervals is None:
            return [(str(path), None, None) for path in self.paths]
        return [
            (str(path), *_interval_key(row))
            for path, row in zip(self.paths, self.intervals, strict=True)
        ]

    def take(self, rows: ArrayLike) -> FeatureSet:
        """The rows ``rows`` (indices or a boolean mask), in that order.

        Examples
        --------
        >>> features = feature_set_from_table(
        ...     pd.DataFrame({"path": ["a", "b", "c"], "score": [1.0, 2.0, 3.0], "f": [4, 5, 6]}),
        ...     target="score",
        ... )
        >>> features.take([2, 0]).paths.tolist(), features.take([2, 0]).X.ravel().tolist()
        (['c', 'a'], [6.0, 4.0])
        """
        index = np.asarray(rows)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        return replace(
            self,
            X=self.X[index],
            y=np.asarray(self.y)[index],
            groups=np.asarray(self.groups)[index],
            paths=np.asarray(self.paths)[index],
            intervals=None if self.intervals is None else self.intervals[index],
        )

    def select_blocks(self, names: str | Sequence[str]) -> FeatureSet:
        """Only the columns of the named blocks, side by side in the given order.

        Examples
        --------
        >>> table = pd.DataFrame({"path": ["a", "b"], "y": [1.0, 2.0],
        ...                       "l0": [1, 2], "l1": [3, 4], "l2": [5, 6]})
        >>> features = feature_set_from_table(
        ...     table, target="y", blocks={"first": ["l0"], "rest": ["l1", "l2"]}
        ... )
        >>> picked = features.select_blocks(["rest"])
        >>> picked.feature_names, picked.blocks
        (('l1', 'l2'), {'rest': (0, 2)})
        """
        wanted = [names] if isinstance(names, str) else list(names)
        if self.blocks is None:
            raise DatasetError("this feature file has no blocks")
        unknown = [name for name in wanted if name not in self.blocks]
        if unknown:
            raise DatasetError(f"unknown block(s) {unknown}; available: {list(self.blocks)}")
        columns: list[int] = []
        blocks: dict[str, tuple[int, int]] = {}
        for name in wanted:
            start, stop = self.blocks[name]
            blocks[name] = (len(columns), len(columns) + stop - start)
            columns.extend(range(start, stop))
        all_names = self.feature_names
        return replace(
            self,
            X=self.X[:, columns],
            names=tuple(all_names[column] for column in columns),
            blocks=blocks,
        )


def feature_set_from_table(
    table: pd.DataFrame,
    *,
    target: str,
    feature_columns: Sequence[str] | None = None,
    blocks: Mapping[str, Sequence[str]] | None = None,
    task: str = "regression",
    kind: str = "external",
    extractor: Mapping[str, Any] | None = None,
) -> FeatureSet:
    """Wrap a table of precomputed features in a :class:`FeatureSet`.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per item with ``path``, the ``target`` column, and optionally
        ``group``, ``start_s``/``end_s`` and the feature columns. Paths are
        kept as given, so files to be compared must spell them the same way.
    target : str
        Rating or class column.
    feature_columns : sequence of str, optional
        Feature columns, in order; default the block columns, or else every
        numeric column that is not ``path``, ``group``, an interval or the target.
    blocks : mapping of str to sequence of str, optional
        Column names of every block; the columns are laid out block after block.
    task : {"regression", "classification"}, optional
        Converts the target to float or to class names.
    kind : str, optional
        Label of the front end; anything but ``mfcc``.
    extractor : mapping, optional
        Provenance stored with the file.

    Returns
    -------
    FeatureSet

    Raises
    ------
    DatasetError
        For missing or non-numeric columns, empty targets, or a column in two blocks.

    Examples
    --------
    >>> table = pd.DataFrame(
    ...     {"path": ["a.wav", "b.wav"], "group": ["s1", "s2"], "severity": [10, 60],
    ...      "hnr_db": [21.5, 8.0], "jitter": [0.004, None]}
    ... )
    >>> features = feature_set_from_table(table, target="severity", kind="clinical")
    >>> features.feature_names, features.y.tolist(), features.kind
    (('hnr_db', 'jitter'), [10.0, 60.0], 'clinical')
    """
    if kind == MFCC_KIND:
        raise ConfigError("kind='mfcc' is reserved for files written by 'prepare'")
    for column in ("path", target):
        if column not in table:
            raise DatasetError(f"the table lacks column {column!r}")
    if table[target].isna().any():
        raise DatasetError(f"column {target!r} has empty cells")
    layout: dict[str, tuple[int, int]] | None = None
    if blocks is not None:
        ordered = [str(column) for columns in blocks.values() for column in columns]
        repeated = sorted({column for column in ordered if ordered.count(column) > 1})
        if repeated:
            raise DatasetError(f"column(s) {repeated} belong to more than one block")
        if feature_columns is not None and list(feature_columns) != ordered:
            raise DatasetError("feature_columns must list the block columns block after block")
        feature_columns = ordered
        layout, position = {}, 0
        for name, columns in blocks.items():
            layout[str(name)] = (position, position + len(columns))
            position += len(columns)
    if feature_columns is None:
        reserved = {"path", "group", target, *INTERVAL_COLUMNS}
        feature_columns = [
            str(column)
            for column in table.select_dtypes(include=[np.number]).columns
            if column not in reserved
        ]
    missing = [column for column in feature_columns if column not in table]
    if missing:
        raise DatasetError(f"the table lacks feature column(s) {_listing(missing)}")
    if not feature_columns:
        raise DatasetError("the table has no feature columns")
    values = table[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    text = [
        column
        for column in feature_columns
        if (values[column].isna() & table[column].notna()).any()
    ]
    if text:
        raise DatasetError(f"feature column(s) {_listing(text)} are not numeric")
    y = table[target].to_numpy()
    y = y.astype(np.float64) if task == "regression" else y.astype(str)
    groups = table["group"].fillna("").astype(str) if "group" in table else [""] * len(table)
    return FeatureSet(
        X=values.to_numpy(dtype=np.float64),
        y=y,
        groups=np.asarray(groups, dtype=str),
        paths=table["path"].astype(str).to_numpy(),
        config={"data": {"task": task, "target": target}},
        names=tuple(feature_columns),
        kind=kind,
        blocks=layout,
        intervals=index_intervals(table),
        extractor=None if extractor is None else dict(extractor),
    )


def save_features(path: str | Path, features: FeatureSet) -> Path:
    """Write ``features.npz``.

    Keys ``X``, ``y``, ``groups``, ``paths``, ``config_json`` and ``kind``, and
    when present ``names``, ``blocks_json``, ``intervals`` and ``extractor_json``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "X": np.asarray(features.X, dtype=np.float64),
        "y": np.asarray(features.y),
        "groups": np.asarray(features.groups).astype(str),
        "paths": np.asarray(features.paths).astype(str),
        "config_json": np.asarray(json.dumps(features.config, sort_keys=True)),
        "kind": np.asarray(features.kind),
    }
    if features.names is not None:
        arrays["names"] = np.asarray(features.names, dtype=str)
    if features.blocks is not None:
        arrays["blocks_json"] = np.asarray(
            json.dumps({k: list(v) for k, v in features.blocks.items()})
        )
    if features.intervals is not None:
        arrays["intervals"] = np.asarray(features.intervals, dtype=np.float64)
    if features.extractor is not None:
        arrays["extractor_json"] = np.asarray(json.dumps(features.extractor, sort_keys=True))
    np.savez_compressed(destination, **arrays)
    return destination


def _differences(stored: dict[str, Any], current: dict[str, Any], kind: str) -> list[str]:
    keys = []
    if kind == MFCC_KIND:
        for section in FEATURE_SECTIONS:
            old, new = stored.get(section, {}), current.get(section, {})
            keys.extend(
                f"{section}.{key}"
                for key in sorted(set(old) | set(new))
                if old.get(key) != new.get(key)
            )
    old_data, new_data = stored.get("data", {}), current.get("data", {})
    keys.extend(
        f"data.{key}"
        for key in _TARGET_KEYS
        if (kind == MFCC_KIND or key in old_data) and old_data.get(key) != new_data.get(key)
    )
    return keys


def load_features(path: str | Path, config: Config | None = None) -> FeatureSet:
    """Read ``features.npz``; with ``config``, insist on the same feature settings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ConfigError
        If ``config`` differs from the settings ``prepare`` used in
        :data:`~voice_severity_bench.config.FEATURE_SECTIONS` or in
        ``data.task`` / ``data.target``. For external front ends only the
        stored ``data`` keys are compared.
    DatasetError
        If the file is not a feature file of this package.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"feature file not found: {source}. Run 'voice-severity-bench prepare' first"
        )
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{source} is not a readable .npz file: {exc}") from None
    needed = {"X", "y", "groups", "paths", "config_json"}
    if not needed <= set(arrays):
        raise DatasetError(
            f"{source} lacks {sorted(needed - set(arrays))}; was it written by 'prepare'?"
        )
    stored = json.loads(str(arrays["config_json"]))
    kind = str(arrays["kind"]) if "kind" in arrays else MFCC_KIND
    if config is not None:
        changed = _differences(stored, config.to_dict(), kind)
        if changed:
            raise ConfigError(
                f"feature settings differ from the ones 'prepare' used: {', '.join(changed)}; "
                "run 'prepare' again or use the same --config"
            )
    blocks = None
    if "blocks_json" in arrays:
        blocks = {
            key: (int(a), int(b)) for key, (a, b) in json.loads(str(arrays["blocks_json"])).items()
        }
    return FeatureSet(
        X=arrays["X"].astype(np.float64),
        y=arrays["y"],
        groups=arrays["groups"].astype(str),
        paths=arrays["paths"].astype(str),
        config=stored,
        names=tuple(arrays["names"].astype(str)) if "names" in arrays else None,
        kind=kind,
        blocks=blocks,
        intervals=arrays["intervals"].astype(np.float64) if "intervals" in arrays else None,
        extractor=json.loads(str(arrays["extractor_json"])) if "extractor_json" in arrays else None,
    )


def build_dataset_index(
    audio_root: str | Path,
    targets: str | Path,
    *,
    id_column: str = "id",
) -> Path:
    """Write ``<audio_root>/dataset.csv`` by matching audio files to a ratings table.

    Every row of ``targets`` is matched to the audio file whose path relative
    to ``audio_root`` without the suffix, or whose bare file stem, equals the
    row's ``id``. All other columns of ``targets`` (ratings, ``label``,
    ``group``, ...) are copied.

    Parameters
    ----------
    audio_root : str or pathlib.Path
        Folder searched recursively for ``.wav`` files.
    targets : str or pathlib.Path
        CSV with an ``id`` column and the ratings.
    id_column : str, optional
        Name of the identifier column.

    Returns
    -------
    pathlib.Path
        The written index.

    Raises
    ------
    DatasetError
        For duplicate identifiers, ambiguous stems, unmatched rows, or two
        identifiers that match the same file (one by relative path, the other
        by stem).
    """
    root = Path(audio_root)
    if not root.is_dir():
        raise FileNotFoundError(f"audio folder not found: {root}")
    table_path = Path(targets)
    if not table_path.is_file():
        raise FileNotFoundError(f"ratings table not found: {table_path}")
    table = pd.read_csv(table_path, dtype={id_column: str})
    if id_column not in table.columns:
        raise DatasetError(
            f"{table_path} lacks the {id_column!r} column; found {list(table.columns)}"
        )
    duplicated = table[id_column][table[id_column].duplicated()].tolist()
    if duplicated:
        raise DatasetError(f"{table_path} repeats identifier(s) {_listing(duplicated)}")
    files = iter_audio_files(root)
    by_relative = {file.relative_to(root).with_suffix("").as_posix(): file for file in files}
    by_stem: dict[str, list[Path]] = {}
    for file in files:
        by_stem.setdefault(file.stem, []).append(file)
    matched, unmatched = [], []
    claimed: dict[Path, list[str]] = {}
    for identifier in table[id_column]:
        key = str(identifier)
        if key in by_relative:
            matched.append(by_relative[key])
        elif len(by_stem.get(key, [])) == 1:
            matched.append(by_stem[key][0])
        elif key in by_stem:
            raise DatasetError(f"identifier {key!r} matches several files: {by_stem[key]}")
        else:
            unmatched.append(key)
            continue
        claimed.setdefault(matched[-1], []).append(key)
    if unmatched:
        raise DatasetError(
            f"{len(unmatched)} identifier(s) in {table_path} have no audio file under {root}: "
            f"{_listing(unmatched)}"
        )
    shared = [
        f"{file.relative_to(root).as_posix()} ({', '.join(keys)})"
        for file, keys in claimed.items()
        if len(keys) > 1
    ]
    if shared:
        raise DatasetError(
            f"identifiers in {table_path} match the same audio file: {_listing(shared)}"
        )
    unused = len(files) - len(set(matched))
    if unused:
        _LOGGER.warning("%d audio file(s) under %s have no row in %s", unused, root, table_path)
    index = table.drop(columns=[id_column])
    index.insert(0, "path", [file.relative_to(root).as_posix() for file in matched])
    destination = root / DATASET_INDEX_NAME
    index.to_csv(destination, index=False, lineterminator="\n")
    return destination


def load_analysis_table(
    path: str | Path,
    *,
    target: str | None = None,
    predictors: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Predictors and response for the regression-analysis commands.

    Parameters
    ----------
    path : str or pathlib.Path
        ``features.npz`` from ``prepare`` (the response is its target), or a
        CSV table (``target`` names the response column).
    target : str, optional
        Response column of a CSV table; ignored for ``.npz`` files.
    predictors : sequence of str, optional
        Columns to use; default every numeric column except the response.

    Returns
    -------
    X : pandas.DataFrame
        Predictor columns.
    y : pandas.Series
        Numeric response.

    Raises
    ------
    DatasetError
        For unknown columns, a non-numeric response or too few complete rows.
    """
    source = Path(path)
    if source.suffix == ".npz":
        features = load_features(source)
        frame = pd.DataFrame(features.X, columns=list(features.feature_names))
        response_name = str(features.config.get("data", {}).get("target", "target"))
        try:
            response = pd.Series(features.y.astype(np.float64), name=response_name)
        except ValueError:
            raise DatasetError(
                f"{source} holds class labels; regression analysis needs a numeric rating"
            ) from None
    else:
        if not source.is_file():
            raise FileNotFoundError(f"table not found: {source}")
        frame = pd.read_csv(source)
        if not target:
            raise ConfigError("a CSV table needs --target to name the response column")
        if target not in frame.columns:
            raise DatasetError(f"{source} has no column {target!r}; found {list(frame.columns)}")
        response = pd.to_numeric(frame.pop(target), errors="coerce").rename(target)
    if predictors:
        unknown = [name for name in predictors if name not in frame.columns]
        if unknown:
            raise DatasetError(
                f"unknown predictor(s) {unknown}; available: {list(frame.columns)[:20]}"
            )
        frame = frame[list(predictors)]
    else:
        frame = frame.select_dtypes(include=[np.number])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    complete = frame.notna().all(axis=1) & response.notna()
    dropped = int((~complete).sum())
    if dropped:
        _LOGGER.warning("%d row(s) with missing or non-numeric values were dropped", dropped)
    frame, response = (
        frame[complete].reset_index(drop=True),
        response[complete].reset_index(drop=True),
    )
    if frame.shape[1] == 0:
        raise DatasetError(f"{source} has no numeric predictor columns")
    if len(response) < 3:
        raise DatasetError(f"{source} has only {len(response)} complete row(s)")
    return frame, response
