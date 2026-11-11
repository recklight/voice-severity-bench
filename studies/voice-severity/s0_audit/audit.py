"""S0 data audit of PVQD: identifiers, demographics, ratings, recordings, reliability.

Reads ``data/pvqd`` without modifying it and writes every output to
``s0_audit/out``. Nothing here fits a model. Runs on Python 3.9 or newer
with numpy, pandas, scipy, soundfile and openpyxl.

Usage::

    python audit.py            # all steps
    python audit.py --no-audio # skip the recording statistics (fast)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "pvqd"
OUT = HERE / "out"

# Characteristic label in Ratings_both_scales.xlsx -> (scale, dimension, *_only.xlsx file)
DIMENSIONS = {
    "CAPE-V Severity": ("capev", "severity", "cape_v_severity_only.xlsx"),
    "CAPE-V Roughness": ("capev", "roughness", "cape_v_roughness_only.xlsx"),
    "CAPE-V Breathiness": ("capev", "breathiness", "cape_v_breathiness_only.xlsx"),
    "CAPE-V Strain": ("capev", "strain", "cape_v_strain_only.xlsx"),
    "CAPE-V Pitch": ("capev", "pitch", "cape_v_pitch_only.xlsx"),
    "CAPE-V Loudness": ("capev", "loudness", "cape_v_loudness_only.xlsx"),
    "GRBAS Grade": ("grbas", "grade", "grbas_grade_only.xlsx"),
    "GRBAS Roughness": ("grbas", "roughness", "grbas_roughness_only.xlsx"),
    "GRBAS Breathiness": ("grbas", "breathiness", "grbas_breathiness_only.xlsx"),
    "GRBAS Asthenia": ("grbas", "asthenia", "grbas_asthenia_only.xlsx"),
    "GRBAS Strain": ("grbas", "strain", "grbas_strain_only.xlsx"),
}

# Published values, "database overview.pdf" (identical in the v1 and v4 text on these numbers).
OFFICIAL = {
    "capev_severity": {"inter": 0.918, "intra": 0.943, "pearson_trials": 0.890},
    "capev_roughness": {"inter": 0.789, "intra": 0.896, "pearson_trials": 0.814},
    "capev_breathiness": {"inter": 0.827, "intra": 0.911, "pearson_trials": 0.833},
    "capev_strain": {"inter": 0.829, "intra": 0.908, "pearson_trials": 0.828},
    "capev_pitch": {"inter": 0.856, "intra": 0.878, "pearson_trials": 0.772},
    "capev_loudness": {"inter": 0.870, "intra": 0.905, "pearson_trials": 0.824},
    "grbas_grade": {"inter": 0.911, "intra": 0.905, "pearson_trials": 0.827},
    "grbas_roughness": {"inter": 0.787, "intra": 0.846, "pearson_trials": 0.734},
    "grbas_breathiness": {"inter": 0.844, "intra": 0.884, "pearson_trials": 0.793},
    "grbas_asthenia": {"inter": 0.843, "intra": 0.892, "pearson_trials": 0.804},
    "grbas_strain": {"inter": 0.845, "intra": 0.862, "pearson_trials": 0.757},
}

# Rating-sheet identifiers with no audio file of that name; the audio and the
# demographics both use the SJ form, so ST is read as a typo in the rating sheets.
MANUAL_RATING_MAP = {"ST1001": "SJ1001", "ST2004": "SJ2004"}

SUMMARY_ROWS = {"MEAN", "MEDIAN", "MIN", "MAX", "RANGE", "SD", "STD"}
RATER_COLUMN = re.compile(r"rater\s*(\d)\s*time\s*(\d)", re.IGNORECASE)

SHORT_S = 8.0
LONG_S = 45.0
HIGH_SD = 20.0  # SD of all severity ratings; twice the 10-point agreement margin (Nagle, 2016)
CLIP_LEVEL = 0.999
CLIP_FRACTION = 1e-3


# --------------------------------------------------------------------------- ids
def normalize_id(raw: str) -> Tuple[str, str]:
    """Normalize a file or participant identifier and name the rules applied."""
    text = str(raw).strip().upper()
    rules = ["R1"]
    stripped = re.sub(r"[\s_]*E_?NSS$", "", text)
    if stripped != text:
        rules.append("R2")
        text = stripped
    stripped = text.rstrip("._ ")
    if stripped != text:
        rules.append("R3")
        text = stripped
    return text, "+".join(rules)


def build_id_map(wav_files: List[Path], rating_ids: List[str], demog_ids: List[str]) -> pd.DataFrame:
    rows = []
    rating_norm = {}
    for raw in rating_ids:
        key, rule = normalize_id(raw)
        if key in MANUAL_RATING_MAP:
            key, rule = MANUAL_RATING_MAP[key], rule + "+R4"
        rating_norm[key] = (raw, rule)
    demog_norm = {}
    for raw in demog_ids:
        key, rule = normalize_id(raw)
        demog_norm.setdefault(key, []).append(raw)
    seen = set()
    for path in wav_files:
        key, rule = normalize_id(path.stem)
        if key in seen:
            raise SystemExit(f"two audio files normalize to {key}")
        seen.add(key)
        rating = rating_norm.get(key)
        demog = demog_norm.get(key)
        notes = []
        if rating is None:
            notes.append("no rating row")
        if demog is None:
            notes.append("no demographics row")
        elif len(demog) > 1:
            notes.append(f"{len(demog)} demographics rows")
        rows.append(
            {
                "file_id": key,
                "wav_file": path.name,
                "wav_rule": rule,
                "rating_id": rating[0] if rating else "",
                "rating_rule": rating[1] if rating else "",
                "demog_id": demog[0] if demog else "",
                "note": "; ".join(notes),
            }
        )
    table = pd.DataFrame(rows).sort_values("file_id").reset_index(drop=True)
    orphans = sorted(set(rating_norm) - seen)
    return table, orphans, sorted(set(demog_norm) - seen)


# ------------------------------------------------------------------ demographics
def clean_demographics(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    raw = raw.rename(columns=lambda c: str(c).strip())
    counts = {"rows_read": int(len(raw))}
    ids = raw["Participant ID"].astype(str).str.strip()
    gender = raw["Gender"].astype(str).str.strip()
    summary = gender.str.upper().isin(SUMMARY_ROWS) | ids.str.upper().isin(SUMMARY_ROWS)
    empty = raw["Participant ID"].isna() | (ids == "") | (ids.str.lower() == "nan")
    counts["summary_rows_dropped"] = int(summary.sum())
    counts["empty_id_rows_dropped"] = int((empty & ~summary).sum())
    table = raw[~summary & ~empty].copy()
    sex_map = {"f": "F", "female": "F", "m": "M", "male": "M"}
    table["sex"] = table["Gender"].astype(str).str.strip().str.lower().map(sex_map)
    table["age"] = pd.to_numeric(table["Age"], errors="coerce")
    diagnosis = table["Diagnosis"].astype(str).str.strip()
    lower = diagnosis.str.lower()
    flag = pd.Series(np.nan, index=table.index)
    flag[lower.isin(["y", "yes"])] = 1
    flag[lower.isin(["n", "no"])] = 0
    named = ~lower.isin(["y", "yes", "n", "no", "nan", ""])
    flag[named] = 1
    table["dysphonia_flag"] = flag
    table["diagnosis_text"] = np.where(named, diagnosis, "")
    table["demog_key"] = [normalize_id(v)[0] for v in table["Participant ID"]]
    counts["duplicate_ids"] = int(table["demog_key"].duplicated(keep=False).sum())
    counts["sex_unparsed"] = int(table["sex"].isna().sum())
    out = table[["demog_key", "Participant ID", "sex", "age", "dysphonia_flag", "diagnosis_text"]]
    out = out.rename(columns={"Participant ID": "raw_id"})
    return out.reset_index(drop=True), counts


# ----------------------------------------------------------------------- ratings
def parse_ratings(id_lookup: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Long rating table plus the sheet's own summary columns, with cross-checks."""
    problems: List[str] = []
    long_rows = []
    sheet_rows = []
    book = DATA / "ratings" / "Ratings_both_scales.xlsx"
    for sheet in ("CAPE-V", "GRBAS"):
        frame = pd.read_excel(book, sheet_name=sheet)
        frame.columns = [str(c).strip() for c in frame.columns]
        rater_cols = {c: RATER_COLUMN.search(c) for c in frame.columns}
        rater_cols = {c: (int(m.group(1)), int(m.group(2))) for c, m in rater_cols.items() if m}
        avg_col = next(c for c in frame.columns if c.lower().startswith("average"))
        min_col = next(c for c in frame.columns if c.lower().startswith("minimum"))
        max_col = next(c for c in frame.columns if c.lower().startswith("maximum"))
        sd_col = next(c for c in frame.columns if c.lower().startswith("standard"))
        for row_number, row in frame.iterrows():
            label = str(row["Characteristics"]).strip()
            label = label.replace("GRBAS: ", "GRBAS ")
            if label not in DIMENSIONS:
                problems.append(f"{sheet} row {row_number + 2}: unknown characteristic {label!r}")
                continue
            scale, dimension, _ = DIMENSIONS[label]
            key = normalize_id(row["File"])[0]
            key = MANUAL_RATING_MAP.get(key, key)
            if key not in id_lookup:
                problems.append(f"{sheet} row {row_number + 2}: {row['File']!r} has no audio")
            values = []
            for col, (slot, session) in rater_cols.items():
                value = row[col]
                if pd.notna(value):
                    long_rows.append(
                        {
                            "file_id": key,
                            "scale": scale,
                            "dimension": dimension,
                            "target": f"{scale}_{dimension}",
                            "slot": slot,
                            "session": session,
                            "value": float(value),
                            "sheet": sheet,
                            "row": row_number + 2,
                        }
                    )
                    values.append(float(value))
            sheet_rows.append(
                {
                    "file_id": key,
                    "target": f"{scale}_{dimension}",
                    "sheet_average": float(row[avg_col]),
                    "sheet_min": float(row[min_col]),
                    "sheet_max": float(row[max_col]),
                    "sheet_sd": float(row[sd_col]),
                    "our_mean": float(np.mean(values)) if values else np.nan,
                    "our_min": float(np.min(values)) if values else np.nan,
                    "our_max": float(np.max(values)) if values else np.nan,
                    "our_sd_pop": float(np.std(values, ddof=0)) if values else np.nan,
                    "our_sd_sample": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "n_values": len(values),
                }
            )
    ratings = pd.DataFrame(long_rows)
    checks = pd.DataFrame(sheet_rows)
    return ratings, checks, problems


def check_single_sheets(ratings: pd.DataFrame) -> List[str]:
    """The per-dimension *_only.xlsx files must repeat the raw values of the combined book."""
    problems = []
    for label, (scale, dimension, filename) in DIMENSIONS.items():
        frame = pd.read_excel(DATA / "ratings" / filename)
        frame.columns = [str(c).strip() for c in frame.columns]
        rater_cols = {c: RATER_COLUMN.search(c) for c in frame.columns}
        rater_cols = {c: (int(m.group(1)), int(m.group(2))) for c, m in rater_cols.items() if m}
        ours = ratings[ratings["target"] == f"{scale}_{dimension}"]
        ours = ours.set_index(["file_id", "slot", "session"])["value"]
        theirs = {}
        for _, row in frame.iterrows():
            key = MANUAL_RATING_MAP.get(normalize_id(row["File"])[0], normalize_id(row["File"])[0])
            for col, (slot, session) in rater_cols.items():
                if pd.notna(row[col]):
                    theirs[(key, slot, session)] = float(row[col])
        theirs = pd.Series(theirs)
        if set(theirs.index) != set(ours.index):
            problems.append(f"{filename}: rating cells differ from Ratings_both_scales.xlsx")
            continue
        diff = (theirs.sort_index() - ours.sort_index()).abs().max()
        if diff > 0:
            problems.append(f"{filename}: values differ by up to {diff}")
    return problems


def slot_means(ratings: pd.DataFrame, target: str) -> pd.DataFrame:
    """Files x slots matrix of each rater's two-session mean (NaN where a slot is absent)."""
    sub = ratings[ratings["target"] == target]
    means = sub.groupby(["file_id", "slot"])["value"].mean().unstack("slot")
    return means.sort_index()


def build_targets(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    problems = []
    frames = []
    for target, sub in ratings.groupby("target"):
        sessions = sub.groupby(["file_id", "slot"])["session"].nunique()
        incomplete = sessions[sessions != 2]
        if len(incomplete):
            problems.append(f"{target}: {len(incomplete)} rater slots without both sessions")
        means = slot_means(ratings, target)
        grouped = sub.groupby("file_id")["value"]
        frame = pd.DataFrame(
            {
                f"{target}": grouped.mean(),
                f"{target}__session1": sub[sub["session"] == 1].groupby("file_id")["value"].mean(),
                f"{target}__slotmean": means.mean(axis=1),
                f"{target}__n_ratings": grouped.size(),
                f"{target}__n_slots": means.notna().sum(axis=1),
                f"{target}__sd": grouped.std(ddof=1),
                f"{target}__sd_slots": means.std(axis=1, ddof=1),
            }
        )
        frames.append(frame)
    targets = pd.concat(frames, axis=1).sort_index()
    targets.index.name = "file_id"
    return targets, problems


# ------------------------------------------------------------------- reliability
def icc_oneway(matrix: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    """One-way random ICC(1,1) and ICC(1,k) for unbalanced items x ratings (NaN = missing).

    k0 replaces k when items carry different numbers of ratings (McGraw & Wong, 1996).
    """
    rows = [r[~np.isnan(r)] for r in matrix]
    rows = [r for r in rows if r.size >= 2]
    n = len(rows)
    sizes = np.array([r.size for r in rows], dtype=float)
    total = sizes.sum()
    grand = np.concatenate(rows).mean()
    means = np.array([r.mean() for r in rows])
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(sum(np.sum((r - r.mean()) ** 2) for r in rows))
    df_b, df_w = n - 1, total - n
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    k0 = (total - np.sum(sizes**2) / total) / (n - 1)
    icc1 = (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)
    icck = (ms_b - ms_w) / ms_b
    f_obs = ms_b / ms_w
    alpha = 1 - level
    f_l = f_obs / stats.f.ppf(1 - alpha / 2, df_b, df_w)
    f_u = f_obs * stats.f.ppf(1 - alpha / 2, df_w, df_b)
    return {
        "n_items": n,
        "k0": float(k0),
        "icc1": float(icc1),
        "icc1_low": float((f_l - 1) / (f_l + k0 - 1)),
        "icc1_high": float((f_u - 1) / (f_u + k0 - 1)),
        "icck": float(icck),
        "icck_low": float(1 - 1 / f_l),
        "icck_high": float(1 - 1 / f_u),
        "f": float(f_obs),
        "df_between": float(df_b),
        "df_within": float(df_w),
        "p": float(stats.f.sf(f_obs, df_b, df_w)),
    }


def icc_twoway(matrix: np.ndarray) -> Dict[str, float]:
    """Shrout-Fleiss ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k) on a complete items x raters matrix."""
    n, k = matrix.shape
    grand = matrix.mean()
    ss_rows = k * np.sum((matrix.mean(axis=1) - grand) ** 2)
    ss_cols = n * np.sum((matrix.mean(axis=0) - grand) ** 2)
    ss_total = np.sum((matrix - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    return {
        "icc2_1": float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)),
        "icc2_k": float((msr - mse) / (msr + (msc - mse) / n)),
        "icc3_1": float((msr - mse) / (msr + (k - 1) * mse)),
        "icc3_k": float((msr - mse) / msr),
    }


def ccc(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(), b.mean()
    cov = np.mean((a - ma) * (b - mb))
    return float(2 * cov / (a.var() + b.var() + (ma - mb) ** 2))


def reliability(ratings: pd.DataFrame, rng_seed: int = 20261102) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ceiling = []
    for target in OFFICIAL:
        sub = ratings[ratings["target"] == target]
        means = slot_means(ratings, target).to_numpy(dtype=float)
        per_rating = sub.pivot_table(index="file_id", columns=["slot", "session"], values="value")
        oneway_slots = icc_oneway(means)
        oneway_all = icc_oneway(per_rating.to_numpy(dtype=float))
        first3 = means[:, :3]
        twoway = icc_twoway(first3)
        # Intra-rater: every (file, slot) with both sessions, and the published variant
        # (three slots per file, a random three of four where there are four).
        wide = sub.pivot_table(index=["file_id", "slot"], columns="session", values="value").dropna()
        t1, t2 = wide[1].to_numpy(dtype=float), wide[2].to_numpy(dtype=float)
        rng = np.random.default_rng([rng_seed, sum(map(ord, target))])
        picked = []
        for file_id, group in wide.groupby(level="file_id"):
            slots = group.index.get_level_values("slot").to_numpy()
            if slots.size > 3:
                slots = np.sort(rng.choice(slots, size=3, replace=False))
            picked.extend((file_id, s) for s in slots)
        wide3 = wide.loc[picked]
        diff = t2 - t1
        intra_pairs = np.column_stack([t1, t2])
        intra = icc_oneway(intra_pairs)
        intra_tw = icc_twoway(intra_pairs)
        intra3_tw = icc_twoway(wide3[[1, 2]].to_numpy(dtype=float))
        official = OFFICIAL[target]
        rows.append(
            {
                "target": target,
                "n_files": oneway_slots["n_items"],
                "k0_slots": oneway_slots["k0"],
                "inter_icc1_slotmeans": oneway_slots["icc1"],
                "inter_icc1_low": oneway_slots["icc1_low"],
                "inter_icc1_high": oneway_slots["icc1_high"],
                "inter_icck_slotmeans": oneway_slots["icck"],
                "inter_icck_low": oneway_slots["icck_low"],
                "inter_icck_high": oneway_slots["icck_high"],
                "inter_icc1_all_ratings": oneway_all["icc1"],
                "inter_icck_all_ratings": oneway_all["icck"],
                "inter_icc2k_first3": twoway["icc2_k"],
                "inter_icc3k_first3": twoway["icc3_k"],
                "official_inter": official["inter"],
                "intra_n_pairs": int(len(wide)),
                "intra_icc1": intra["icc1"],
                "intra_icc1_avg2": intra["icck"],
                "intra_icc3k": intra_tw["icc3_k"],
                "intra_icc3k_3slots": intra3_tw["icc3_k"],
                "intra_pearson": float(np.corrcoef(t1, t2)[0, 1]),
                "official_intra": official["intra"],
                "official_pearson_trials": official["pearson_trials"],
                "intra_mean_abs_diff": float(np.mean(np.abs(diff))),
                "intra_ba_bias": float(diff.mean()),
                "intra_ba_loa_low": float(diff.mean() - 1.96 * diff.std(ddof=1)),
                "intra_ba_loa_high": float(diff.mean() + 1.96 * diff.std(ddof=1)),
            }
        )
        # Rater ceiling on the slot means; the target is their mean.
        k = np.sum(~np.isnan(means), axis=1)
        var = np.nanvar(means, axis=1, ddof=1)
        floor = float(np.sqrt(np.mean(var / k)))
        loro_rating, loro_reference = [], []
        for row in means:
            present = np.flatnonzero(~np.isnan(row))
            for j in present:
                others = row[present[present != j]]
                loro_rating.append(row[j])
                loro_reference.append(others.mean())
        a, b = np.array(loro_rating), np.array(loro_reference)
        ceiling.append(
            {
                "target": target,
                "noise_floor_rmse": floor,
                "r_max": float(np.sqrt(max(oneway_slots["icck"], 0.0))),
                "target_sd": float(np.nanstd(np.nanmean(means, axis=1), ddof=1)),
                "loro_n_pairs": int(a.size),
                "loro_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                "loro_mae": float(np.mean(np.abs(a - b))),
                "loro_pearson": float(np.corrcoef(a, b)[0, 1]),
                "loro_ccc": ccc(a, b),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(ceiling)


# -------------------------------------------------------------------- recordings
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def recording_stats(id_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in id_map.itertuples():
        path = DATA / "audio" / item.wav_file
        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        mono = audio.mean(axis=1)
        frame = int(round(0.030 * rate))
        hop = int(round(0.015 * rate))
        n_frames = 1 + max(0, (mono.size - frame) // hop)
        energy = np.array([np.mean(mono[i * hop : i * hop + frame] ** 2) for i in range(n_frames)])
        freqs, psd = signal.welch(mono, fs=rate, nperseg=2048)
        cumulative = np.cumsum(psd) / np.sum(psd)
        row = {
            "file_id": item.file_id,
            "wav_file": item.wav_file,
            "sha256": sha256_file(path),
            "prefix": re.match(r"[A-Z]+", item.file_id).group(0),
            "sample_rate": rate,
            "channels": audio.shape[1],
            "n_samples": audio.shape[0],
            "duration_s": audio.shape[0] / rate,
            "rms_dbfs": db(np.sqrt(np.mean(mono**2))),
            "peak_dbfs": db(np.max(np.abs(audio))),
            "clip_fraction": float(np.mean(np.abs(audio) >= CLIP_LEVEL)),
            "dc_offset": float(mono.mean()),
            "noise_floor_dbfs": float(10 * np.log10(max(np.percentile(energy, 10), 1e-24))),
            "rolloff99_hz": float(freqs[np.searchsorted(cumulative, 0.99)]),
            "channel_corr": np.nan,
            "channel_level_diff_db": np.nan,
        }
        if audio.shape[1] == 2:
            left, right = audio[:, 0], audio[:, 1]
            row["channel_corr"] = float(np.corrcoef(left, right)[0, 1])
            row["channel_level_diff_db"] = db(np.sqrt(np.mean(left**2))) - db(
                np.sqrt(np.mean(right**2))
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ report
def write_csv(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    frame.to_csv(OUT / name, index=index, lineterminator="\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-audio", action="store_true", help="skip the recording statistics")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore", category=UserWarning)  # openpyxl styling notices
    OUT.mkdir(exist_ok=True)

    wav_files = sorted((DATA / "audio").glob("*.wav"))
    both = pd.read_excel(DATA / "ratings" / "Ratings_both_scales.xlsx", sheet_name="CAPE-V")
    rating_ids = sorted(set(str(v).strip() for v in both["File"].dropna()))
    demog_raw = pd.read_excel(DATA / "ratings" / "Demographics.xlsx")
    demog, demog_counts = clean_demographics(demog_raw)

    id_map, rating_orphans, demog_orphans = build_id_map(
        wav_files, rating_ids, demog["demog_key"].tolist()
    )
    write_csv(id_map, "id_map.csv")
    lookup = dict(zip(id_map["file_id"], id_map["wav_file"]))

    demog_out = demog.merge(
        id_map[["file_id"]], left_on="demog_key", right_on="file_id", how="left"
    )
    write_csv(demog_out, "demographics_clean.csv")

    ratings, sheet_checks, problems = parse_ratings(lookup)
    problems += check_single_sheets(ratings)
    write_csv(ratings, "ratings_long.csv")
    sheet_checks["mean_diff"] = (sheet_checks["our_mean"] - sheet_checks["sheet_average"]).abs()
    sheet_checks["sd_pop_diff"] = (sheet_checks["our_sd_pop"] - sheet_checks["sheet_sd"]).abs()
    sheet_checks["sd_sample_diff"] = (sheet_checks["our_sd_sample"] - sheet_checks["sheet_sd"]).abs()
    write_csv(sheet_checks, "sheet_checks.csv")

    targets, target_problems = build_targets(ratings)
    problems += target_problems
    write_csv(targets, "targets.csv", index=True)

    reliab, ceiling = reliability(ratings)
    write_csv(reliab, "reliability.csv")
    write_csv(ceiling, "rater_ceiling.csv")

    summary = {
        "n_wav": len(wav_files),
        "n_rating_ids": len(rating_ids),
        "rating_orphans": rating_orphans,
        "demog_orphans_not_in_audio": demog_orphans,
        "demographics": demog_counts,
        "ratings_per_file_severity": targets["capev_severity__n_ratings"].value_counts().to_dict(),
        "max_mean_diff_vs_sheet": float(sheet_checks["mean_diff"].max()),
        "sd_definition_matching_sheet": (
            "population"
            if sheet_checks["sd_pop_diff"].max() < sheet_checks["sd_sample_diff"].max()
            else "sample"
        ),
        "max_sd_diff_vs_sheet": float(
            min(sheet_checks["sd_pop_diff"].max(), sheet_checks["sd_sample_diff"].max())
        ),
        "slotmean_equals_mean_all_max_diff": float(
            max(
                (targets[t] - targets[f"{t}__slotmean"]).abs().max()
                for t in OFFICIAL
            )
        ),
        "problems": problems,
    }

    if not args.no_audio:
        rec = recording_stats(id_map)
        manifest = json.loads((ROOT / "data" / "pvqd_manifest.json").read_text(encoding="utf-8"))
        expected = {Path(f["path"]).name: f["sha256"] for f in manifest["files"]}
        rec["sha256_matches_manifest"] = [expected.get(w) == h for w, h in zip(rec["wav_file"], rec["sha256"])]
        write_csv(rec, "recordings.csv")

        merged = rec.merge(targets[["capev_severity", "capev_severity__sd"]], left_on="file_id", right_index=True)
        merged = merged.merge(demog_out[["file_id", "sex", "age", "dysphonia_flag", "diagnosis_text"]], on="file_id", how="left")
        site = merged.groupby("prefix").agg(
            n=("file_id", "size"),
            severity_mean=("capev_severity", "mean"),
            severity_sd=("capev_severity", "std"),
            severity_median=("capev_severity", "median"),
            share_25k=("sample_rate", lambda s: float(np.mean(s == 25000))),
            duration_median=("duration_s", "median"),
            rms_median=("rms_dbfs", "median"),
            noise_floor_median=("noise_floor_dbfs", "median"),
            share_female=("sex", lambda s: float(np.mean(s == "F"))),
            age_median=("age", "median"),
        )
        write_csv(site, "site_severity.csv", index=True)
        crosstab = pd.crosstab(merged["prefix"], merged["sample_rate"])
        write_csv(crosstab, "site_samplerate_crosstab.csv", index=True)
        groups = [g["capev_severity"].dropna().to_numpy() for _, g in merged.groupby("prefix")]
        kw = stats.kruskal(*groups)
        rate_groups = [g["capev_severity"].to_numpy() for _, g in merged.groupby("sample_rate")]
        mw = stats.mannwhitneyu(*rate_groups) if len(rate_groups) == 2 else None
        summary["kruskal_severity_by_prefix"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}
        if mw is not None:
            summary["mannwhitney_severity_by_rate"] = {"U": float(mw.statistic), "p": float(mw.pvalue)}
        summary["sha256_all_match_manifest"] = bool(rec["sha256_matches_manifest"].all())

        # Speaker screen: duplicated audio, and identical demographic tuples within a group.
        dup_hash = merged[merged["sha256"].duplicated(keep=False)]
        tuples = merged.dropna(subset=["sex", "age"])
        tuple_key = tuples["prefix"] + "|" + tuples["sex"] + "|" + tuples["age"].astype(int).astype(str) + "|" + tuples["diagnosis_text"].fillna("")
        dup_tuple = tuples[tuple_key.duplicated(keep=False)].assign(tuple_key=tuple_key)
        screen = pd.concat(
            [dup_hash.assign(reason="DUP_AUDIO"), dup_tuple.assign(reason="SAME_DEMOGRAPHICS")],
            ignore_index=True,
        )
        write_csv(screen[["reason", "file_id", "prefix", "sex", "age", "diagnosis_text", "duration_s", "sha256"] + (["tuple_key"] if "tuple_key" in screen else [])], "speaker_screen.csv")

        rows = []
        for item in merged.itertuples():
            codes = []
            if item.file_id not in targets.index:
                codes.append("NO_RATING")
            if item.channels == 2:
                codes.append("STEREO")
            if item.clip_fraction > CLIP_FRACTION:
                codes.append("CLIPPING")
            if item.duration_s < SHORT_S:
                codes.append("SHORT")
            if item.duration_s > LONG_S:
                codes.append("LONG_EXTRA_SPEECH")
            if item.capev_severity__sd >= HIGH_SD:
                codes.append("HIGH_RATER_SD")
            if item.file_id in set(dup_hash["file_id"]):
                codes.append("DUP_AUDIO")
            if codes:
                hard = {"NO_RATING", "NO_AUDIO", "UNREADABLE", "DUP_AUDIO"}
                rows.append(
                    {
                        "file_id": item.file_id,
                        "reason_codes": "+".join(codes),
                        "evidence": f"dur={item.duration_s:.1f}s ch={item.channels} clip={item.clip_fraction:.4f} sev_sd={item.capev_severity__sd:.1f}",
                        "proposed": "exclude" if hard & set(codes) else "keep; sensitivity flag",
                    }
                )
        write_csv(pd.DataFrame(rows), "exclusion_candidates.csv")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
