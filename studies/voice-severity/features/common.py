"""Shared data access for the feature extractors (S4).

Every extractor reads recordings through :func:`load_interval`, which applies
the frozen preprocessing of analysis_plan.md section 5 with
voice_severity_bench.read_audio: one channel per D3, the interval cut at the
file's own rate, resampled to 16 kHz, scaled to -26 dBFS RMS.

Item keys are ``(path, start_s, end_s)`` with ``path`` exactly as
``load_dataset_index`` resolves it, so every feature file of one material
lines up with every other one.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.io import wavfile

from voice_severity_bench import AudioConfig, load_dataset_index, read_audio

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data" / "pvqd" / "audio"
DERIVED = ROOT / "data" / "derived"
INDEX = ROOT / "features" / "index"
OUT = ROOT / "features" / "out"
REPO = ROOT.parent / "voice-severity-bench"
MATERIALS = ("a", "i", "sentences", "whole")
PRIMARY_TARGET = "capev_severity"
#: The D3 channel choice; files not listed here are averaged by read_audio.
CHANNEL_CHOICE = {"NYU1015": 1}
#: S1 rounds interval ends to 1 ms; an end this far past the file is clipped.
END_TOLERANCE_S = 0.002
#: Seed of the file permutation behind every dry run (harness dry, analyze --dry).
DRY_SEED = 20261105


@lru_cache(maxsize=1)
def plan() -> dict[str, Any]:
    with open(ROOT / "analysis_plan.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def audio_config(native: bool = False) -> AudioConfig:
    """16 kHz and -26 dBFS (plan section 5); ``native`` keeps each file's rate (E5f)."""
    settings = plan()["audio"]
    rate = None if native else int(settings["sample_rate"])
    return AudioConfig(sample_rate=rate, rms_dbfs=float(settings["rms_dbfs"]))


def ensure_derived() -> dict[str, Path]:
    """Write the single-channel copies D3 asks for (data/derived/<id>.wav, original rate and format)."""
    ids = pd.read_csv(ROOT / "s0_audit" / "out" / "id_map.csv").set_index("file_id")
    DERIVED.mkdir(parents=True, exist_ok=True)
    written = {}
    for file_id, channel in CHANNEL_CHOICE.items():
        target = DERIVED / f"{file_id}.wav"
        if not target.is_file():
            rate, data = wavfile.read(AUDIO / ids.loc[file_id, "wav_file"])
            wavfile.write(target, rate, np.ascontiguousarray(data[:, channel]))
        written[file_id] = target
    return written


def audio_path(file_id: str, wav_file: str) -> Path:
    if file_id in CHANNEL_CHOICE:
        return DERIVED / f"{file_id}.wav"
    return AUDIO / wav_file


def cv_groups(merged: bool = True) -> dict[str, str]:
    """Cross-validation group of every file: D4 pairs share one group unless ``merged`` is false."""
    ids = pd.read_csv(ROOT / "s0_audit" / "out" / "id_map.csv")["file_id"]
    groups = {file_id: file_id for file_id in ids}
    if merged:
        for pair in plan()["data"]["merged_groups"]:
            label = "+".join(pair)
            for file_id in pair:
                groups[file_id] = label
    return groups


def dry_permutation() -> dict[str, str]:
    """file -> the file whose ratings it carries in a dry run."""
    ids = pd.read_csv(ROOT / "s0_audit" / "out" / "targets.csv")["file_id"].to_numpy()
    shuffled = np.random.default_rng(DRY_SEED).permutation(ids)
    return dict(zip(ids, shuffled, strict=True))


@lru_cache(maxsize=2)
def targets(dry: bool = False) -> pd.DataFrame:
    """The S0 targets; ``dry`` hands every file the ratings of another one."""
    table = pd.read_csv(ROOT / "s0_audit" / "out" / "targets.csv").set_index("file_id")
    if dry:
        mapping = dry_permutation()
        moved = table.loc[[mapping[f] for f in table.index]]
        moved.index = table.index
        table = moved
    return table


@lru_cache(maxsize=1)
def recordings() -> pd.DataFrame:
    return pd.read_csv(ROOT / "s0_audit" / "out" / "recordings.csv").set_index("file_id")


def index(material: str) -> pd.DataFrame:
    """Dataset index of one material: valid S1 intervals only, ``path`` resolved.

    ``whole`` gets open intervals (NaN), so the full file is read without
    rounding; ends that S1 rounded past the file by at most END_TOLERANCE_S
    are clipped to the file length.
    """
    if material not in MATERIALS:
        raise ValueError(f"unknown material {material!r}")
    ensure_derived()
    intervals = pd.read_csv(ROOT / "s1_segments" / "analysis_intervals.csv")
    rows = intervals[(intervals["material"] == material) & (intervals["status"] == "valid")]
    ids = pd.read_csv(ROOT / "s0_audit" / "out" / "id_map.csv").set_index("file_id")
    durations = recordings()["duration_s"]
    groups = cv_groups()
    table = pd.DataFrame(
        {
            "path": [audio_path(f, ids.loc[f, "wav_file"]).as_posix() for f in rows["file_id"]],
            PRIMARY_TARGET: targets().loc[rows["file_id"], PRIMARY_TARGET].to_numpy(),
            "group": [groups[f] for f in rows["file_id"]],
            "file_id": rows["file_id"].to_numpy(),
            "recording_group": rows["file_id"].str.extract(r"^([A-Z]+)")[0].to_numpy(),
        }
    )
    if material == "whole":
        table["start_s"], table["end_s"] = np.nan, np.nan
    else:
        end = rows["end_s"].to_numpy(dtype=float)
        limit = durations.loc[rows["file_id"]].to_numpy(dtype=float)
        over = end - limit
        if np.any(over > END_TOLERANCE_S):
            bad = rows["file_id"].to_numpy()[over > END_TOLERANCE_S].tolist()
            raise ValueError(f"{material}: interval ends beyond the recording for {bad}")
        table["start_s"] = rows["start_s"].to_numpy(dtype=float)
        table["end_s"] = np.minimum(end, limit)
    INDEX.mkdir(parents=True, exist_ok=True)
    destination = INDEX / f"{material}.csv"
    table.to_csv(destination, index=False, lineterminator="\n")
    return load_dataset_index(destination, target=PRIMARY_TARGET)


def load_interval(row: Any, native: bool = False) -> tuple[np.ndarray, int]:
    """The preprocessed signal of one index row."""
    return read_audio(row["path"], audio_config(native), start_s=row["start_s"], end_s=row["end_s"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_state() -> dict[str, Any]:
    """Commit of voice-severity-bench and whether src/ has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src")),
    }


def provenance(script: str | Path, settings: dict[str, Any]) -> dict[str, Any]:
    """What an extractor records in extractor_json."""
    import voice_severity_bench

    return {
        "script": Path(script).name,
        "script_sha256": sha256(Path(script)),
        "common_sha256": sha256(Path(__file__)),
        "analysis_plan_sha256": {
            name: sha256(ROOT / name) for name in ("analysis_plan.md", "analysis_plan.yaml")
        },
        "voice_severity_bench": {"version": voice_severity_bench.__version__, **repo_state()},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "settings": settings,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
