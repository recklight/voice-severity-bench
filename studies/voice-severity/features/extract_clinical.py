"""A3: clinical voice measures with Praat through parselmouth (plan section 6).

    .venv/Scripts/python.exe features/extract_clinical.py --materials a
    .venv/Scripts/python.exe features/extract_clinical.py --materials a --native   # E5f

Pitch range in two passes (De Looze & Hirst, 2008): To Pitch (cc) at
50-700 Hz, then again between 0.75 x Q1 and 1.5 x Q3 of the first pass. The
second pass drives every other measure. A measure Praat leaves undefined is
stored as NaN and its reason goes to the log; the item is kept, and the
model imputes it inside the training folds.

parselmouth is GPL-licensed, which is one reason this script stays in the
paper folder and out of the MIT package.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from voice_severity_bench import feature_set_from_table, save_features  # noqa: E402

SPEC = common.plan()["feature_sets"]["A3"]
COLUMNS = list(SPEC["columns"])
JITTER = tuple(SPEC["jitter_args"])
SHIMMER = tuple(SPEC["shimmer_args"])
HARM = SPEC["harmonicity_cc"]
CPPS = SPEC["cpps"]
FIRST_PASS = tuple(SPEC["pitch"]["first_pass_hz"])
#: Praat defaults of To Pitch (cc) after the time step and floor, ceiling last.
PITCH_CC_DEFAULTS = (15, "no", 0.03, 0.45, 0.01, 0.35, 0.14)
#: Fewer voiced frames than this in the first pass leaves the pitch range undefined.
MIN_VOICED_FRAMES = 3


def _value(x: float) -> float:
    return float(x) if x is not None and math.isfinite(float(x)) else math.nan


def pitch_cc(sound: parselmouth.Sound, floor: float, ceiling: float) -> parselmouth.Pitch:
    return call(sound, "To Pitch (cc)", 0.0, floor, *PITCH_CC_DEFAULTS, ceiling)


def voice_breaks(sound, pitch, points, floor: float, ceiling: float) -> float:
    report = call(
        [sound, pitch, points], "Voice report", 0.0, 0.0, floor, ceiling,
        JITTER[2], SHIMMER[3], 0.03, 0.45,
    )  # fmt: skip
    # Praat prints a zero as "0" and anything else as a percentage, e.g. "12.3%".
    match = re.search(r"Degree of voice breaks:\s*(--undefined--|[-\d.]+)(%?)", report)
    if match is None or match.group(1) == "--undefined--":
        return math.nan
    value = float(match.group(1))
    return value / 100.0 if match.group(2) else value


def measures(signal: np.ndarray, rate: int) -> tuple[dict[str, float], list[str]]:
    """The 18 A3 columns of one interval, and the reasons for any NaN."""
    out = dict.fromkeys(COLUMNS, math.nan)
    notes: list[str] = []
    sound = parselmouth.Sound(signal, sampling_frequency=rate)
    first = pitch_cc(sound, *FIRST_PASS)
    f0 = first.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if voiced.size < MIN_VOICED_FRAMES:
        notes.append(f"first pitch pass found {voiced.size} voiced frames")
        return out, notes
    q1, q3 = np.percentile(voiced, [25, 75])
    floor = SPEC["pitch"]["floor_factor"] * q1
    ceiling = SPEC["pitch"]["ceiling_factor"] * q3
    pitch = pitch_cc(sound, floor, ceiling)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    out["voiced_fraction"] = voiced.size / f0.size if f0.size else math.nan
    if voiced.size >= MIN_VOICED_FRAMES:
        semitones = 12.0 * np.log2(voiced / 100.0)
        out["f0_mean_st"] = float(semitones.mean())
        out["f0_sd_st"] = float(semitones.std(ddof=1))
        low, high = np.percentile(semitones, [5, 95])
        out["f0_range_st"] = float(high - low)
    else:
        notes.append(f"second pitch pass found {voiced.size} voiced frames")
    points = call([sound, pitch], "To PointProcess (cc)")
    for column, command in (
        ("jitter_local", "Get jitter (local)"),
        ("jitter_local_abs", "Get jitter (local, absolute)"),
        ("jitter_rap", "Get jitter (rap)"),
        ("jitter_ppq5", "Get jitter (ppq5)"),
        ("jitter_ddp", "Get jitter (ddp)"),
    ):
        out[column] = _value(call(points, command, 0.0, 0.0, *JITTER))
    for column, command in (
        ("shimmer_local", "Get shimmer (local)"),
        ("shimmer_local_db", "Get shimmer (local_dB)"),
        ("shimmer_apq3", "Get shimmer (apq3)"),
        ("shimmer_apq5", "Get shimmer (apq5)"),
        ("shimmer_apq11", "Get shimmer (apq11)"),
        ("shimmer_dda", "Get shimmer (dda)"),
    ):
        out[column] = _value(call([sound, points], command, 0.0, 0.0, *SHIMMER))
    out["voice_breaks_degree"] = voice_breaks(sound, pitch, points, floor, ceiling)
    harmonicity = call(
        sound, "To Harmonicity (cc)", HARM["time_step"], floor,
        HARM["silence_threshold"], HARM["periods_per_window"],
    )  # fmt: skip
    out["hnr_mean_db"] = _value(call(harmonicity, "Get mean", 0.0, 0.0))
    out["cpps_db"] = cpps(sound)
    notes += [f"{column} undefined" for column, value in out.items() if math.isnan(value)]
    return out, notes


def cpps(sound: parselmouth.Sound) -> float:
    pc = CPPS["power_cepstrogram"]
    cepstrogram = call(
        sound, "To PowerCepstrogram", pc["pitch_floor"], pc["time_step"],
        pc["max_frequency"], pc["preemphasis_from"],
    )  # fmt: skip
    trend = CPPS["trend_line"]
    value = call(
        cepstrogram, "Get CPPS", "no", CPPS["time_averaging_window"],
        CPPS["quefrency_averaging_window"], *CPPS["peak_search_hz"], CPPS["tolerance"],
        CPPS["interpolation"].capitalize(), *trend["quefrency_range"],
        trend["type"].capitalize(), trend["fit"].capitalize(),
    )  # fmt: skip
    return _value(value)


def run(material: str, native: bool, limit: int | None) -> Path | None:
    index = common.index(material)
    if limit is not None:
        index = index.iloc[:limit]
    rows, log = [], []
    start = time.perf_counter()
    for _, row in index.iterrows():
        signal, rate = common.load_interval(row, native=native)
        values, notes = measures(signal, rate)
        rows.append(values)
        log += [{"file_id": row["file_id"], "note": note} for note in notes]
    elapsed = time.perf_counter() - start
    name = "A3_native" if native else "A3"
    table = index[["path", "group", "start_s", "end_s", common.PRIMARY_TARGET]].reset_index(drop=True)
    table = table.join(pd.DataFrame(rows, columns=COLUMNS))
    nan_counts = table[COLUMNS].isna().sum()
    print(f"{name} {material}: {len(table)} items in {elapsed:.1f} s; NaN per column:")
    print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "  none")
    if limit is not None:
        return None
    settings = {
        "praat_version": parselmouth.PRAAT_VERSION,
        "parselmouth_version": parselmouth.__version__,
        "sample_rate": "native" if native else common.plan()["audio"]["sample_rate"],
        "pitch_cc_defaults": PITCH_CC_DEFAULTS,
        "spec": SPEC,
    }
    features = feature_set_from_table(
        table, target=common.PRIMARY_TARGET, feature_columns=COLUMNS, kind="clinical",
        extractor=common.provenance(__file__, settings),
    )  # fmt: skip
    destination = common.OUT / material / f"{name}.npz"
    save_features(destination, features)
    pd.DataFrame(log, columns=["file_id", "note"]).to_csv(
        common.OUT / material / f"{name}_log.csv", index=False, lineterminator="\n"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--materials", nargs="+", default=["a"], choices=common.MATERIALS)
    parser.add_argument("--native", action="store_true", help="Each file's own sampling rate (E5f).")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    for material in args.materials:
        written = run(material, args.native, args.limit)
        if written:
            print("written", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
