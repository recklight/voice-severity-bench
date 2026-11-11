"""S1 automatic segmentation of PVQD recordings into /a/, /i/, sentences and the whole file.

Rating-blind: reads ``s0_audit/out/id_map.csv`` and the audio, never the ratings.
Runs on Python 3.9 or newer with numpy, scipy, soundfile, librosa and kudio.

Method, per recording, on a 16 kHz mono signal with 10 ms frames:

1. Speech spans from ``kudio.vad``.
2. Envelope flux: RMS change over a 50 ms lag of cepstral coefficients c1-c6
   of the 40-band log-mel spectrum (5-frame smoothing, 0.25 s median filter).
   Sustained vowels stay low; running speech changes every phoneme. Six
   coefficients keep the formant envelope and drop the harmonics, so a vowel
   whose pitch wobbles (tremor, vibrato, a glide, a rough voice) still counts
   as steady. A flux on the log-mel bands themselves rose to 3-4.5 dB on the
   tremor vowels of LA9017 and LA9020 and missed them, and those are exactly
   the voices that must not drop out of a severity study.
3. Vowel-quality score (dB): ``[E(550-1100) - E(200-450)] - [E(1800-3000) - E(900-1500)]``.
   /a/ has its F1 in the first band and its F2 below 1500 Hz, /i/ the opposite,
   so /a/ scores high and /i/ low; median-filtered over 0.25 s.
4. Steady stretches are speech with flux below ``flux_max`` (gaps up to
   ``bridge_s`` bridged), at least ``min_token_s`` long. A stretch is split
   where the score crosses ``ai_boundary_db`` and stays across it for
   ``side_min_s``; each piece is /a/ when its median score is above the
   boundary and /i/ below it, flagged low-confidence within ``ai_margin_db``.
   The boundary sits in the empty gap of the bimodal score distribution of all
   steady pieces (-25 to -10 dB on the first full run). With no prior on the
   CAPE-V order, recordings that put the sentences first work too, and an /a/
   that runs straight into an /i/ is split.
5. Sentences: the speech spans that do not overlap a token, grouped into
   blocks that no token interrupts; the block with the most speech is taken.
6. Analysis intervals: the longest token of each vowel that is not a pitch
   glide (slow F0 range above ``glide_st`` semitones) and is at least
   ``steady_min_s`` long; failing that, the longest token, flagged when it is
   a glide. ``trim_s`` is trimmed
   from each end and at most the central ``max_interval_s``, valid when at
   least ``min_interval_s`` remains (flagged under 1.0 s); the sentence block,
   valid with at least ``sentence_min_speech_s`` of speech; the whole file.

Usage::

    python segment.py                 # every file, writes segments.csv etc.
    python segment.py --files SJ1001  # selected files (dev runs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import medfilt, resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIO = ROOT / "data" / "pvqd" / "audio"
CACHE = HERE / "cache"

PARAMS: Dict[str, object] = {
    "sr": 16000,
    "n_fft": 512,
    "win": 400,
    "hop": 160,
    "n_mels": 40,
    "mel_fmin": 60.0,
    "mel_fmax": 7600.0,
    "vad": {"threshold_db": 8.0, "flatness_max": 0.6, "min_speech": 0.3, "min_silence": 0.3, "pad": 0.05},
    "flux_n_ceps": 6,
    "flux_smooth_frames": 5,
    "flux_lag_frames": 5,
    "track_median_frames": 25,
    "flux_max": 5.0,
    "ai_boundary_db": -17.5,
    "ai_margin_db": 7.5,
    "bridge_s": 0.10,
    "side_min_s": 0.3,
    "min_token_s": 0.8,
    "trim_s": 0.10,
    "max_interval_s": 3.0,
    "min_interval_s": 0.6,
    "glide_st": 3.0,
    "glide_smooth_s": 0.5,
    "steady_min_s": 1.2,
    "sentence_min_speech_s": 3.0,
    "sentence_many_spans": 12,
    "sentence_long_gap_s": 3.0,
    "sentence_token_margin_s": 0.2,
    "f0_fmin": 50.0,
    "f0_fmax": 600.0,
    "lpc_sr": 10000,
    "lpc_order": 12,
}

# D3 (S0): NYU1015's first channel is silent (-84 dBFS against -27 dBFS); use the second.
CHANNEL_CHOICE = {"NYU1015": 1}


# ------------------------------------------------------------------ signal prep
def load_mono_16k(path: Path, file_id: str) -> Tuple[np.ndarray, int, float]:
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if file_id in CHANNEL_CHOICE:
        mono = audio[:, CHANNEL_CHOICE[file_id]]
    else:
        mono = audio.mean(axis=1)
    duration = audio.shape[0] / rate
    sr = int(PARAMS["sr"])
    if rate != sr:
        from math import gcd

        g = gcd(rate, sr)
        mono = resample_poly(mono, sr // g, rate // g)
    return mono, rate, duration


def frame_tracks(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-frame energy, flux, vowel score and a coarse log-mel image (10 ms hop)."""
    import librosa

    sr, n_fft, win, hop = (int(PARAMS[k]) for k in ("sr", "n_fft", "win", "hop"))
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)) ** 2
    mel = librosa.feature.melspectrogram(
        S=power, sr=sr, n_mels=int(PARAMS["n_mels"]),
        fmin=float(PARAMS["mel_fmin"]), fmax=float(PARAMS["mel_fmax"]),
    )
    logmel = 10 * np.log10(mel + 1e-10)
    from scipy.fft import dct

    ceps = dct(logmel, type=2, norm="ortho", axis=0)[1 : int(PARAMS["flux_n_ceps"]) + 1]
    k = int(PARAMS["flux_smooth_frames"])
    kernel = np.ones(k) / k
    smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, ceps)
    lag = int(PARAMS["flux_lag_frames"])
    flux = np.zeros(smooth.shape[1])
    flux[lag:] = np.sqrt(np.mean((smooth[:, lag:] - smooth[:, :-lag]) ** 2, axis=0))
    flux[:lag] = flux[lag] if flux.size > lag else 0.0
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    def band(lo: float, hi: float) -> np.ndarray:
        mask = (freqs >= lo) & (freqs < hi)
        return 10 * np.log10(power[mask].sum(axis=0) + 1e-12)

    score = (band(550, 1100) - band(200, 450)) - (band(1800, 3000) - band(900, 1500))
    m = int(PARAMS["track_median_frames"])
    return {
        "energy_db": 10 * np.log10(power.mean(axis=0) + 1e-12),
        "flux": medfilt(flux, m),
        "score": medfilt(score, m),
        "logmel": logmel,
    }


def spans_to_mask(spans: Sequence[Tuple[float, float]], n_frames: int) -> np.ndarray:
    hop_s = int(PARAMS["hop"]) / int(PARAMS["sr"])
    mask = np.zeros(n_frames, dtype=bool)
    for a, b in spans:
        mask[int(round(a / hop_s)) : int(round(b / hop_s)) + 1] = True
    return mask


def runs(labels: np.ndarray) -> List[Tuple[int, int, str]]:
    """Maximal runs ``(start, stop, label)`` of equal consecutive labels (stop exclusive)."""
    out = []
    start = 0
    for t in range(1, labels.size + 1):
        if t == labels.size or labels[t] != labels[start]:
            out.append((start, t, str(labels[start])))
            start = t
    return out


def absorb_short_runs(values: np.ndarray, min_len: int) -> np.ndarray:
    """Give every run shorter than *min_len* frames the label of its longer neighbour."""
    values = values.copy()
    while True:
        pieces = runs(values)
        short = [p for p in pieces if p[1] - p[0] < min_len]
        if len(pieces) < 2 or not short:
            return values
        start, stop, _ = min(short, key=lambda p: p[1] - p[0])
        idx = [p[:2] for p in pieces].index((start, stop))
        left = pieces[idx - 1] if idx > 0 else None
        right = pieces[idx + 1] if idx + 1 < len(pieces) else None
        donor = max((p for p in (left, right) if p is not None), key=lambda p: p[1] - p[0])
        values[start:stop] = donor[2]


def find_tokens(flux: np.ndarray, score: np.ndarray, speech: np.ndarray) -> List[Tuple[int, int, str]]:
    """Sustained-vowel tokens as frame ranges labelled "a" or "i".

    Steady stretches come first (speech with low flux, short dips bridged). A
    stretch is split only where the score crosses the /a/-/i/ midline and stays
    across it for ``side_min_s``; each piece is then labelled by its median
    score. Labelling frames one by one made a breathy /a/ near a threshold
    flicker into many short pieces.
    """
    hop_s = int(PARAMS["hop"]) / int(PARAMS["sr"])
    bridge = int(round(float(PARAMS["bridge_s"]) / hop_s))
    side_min = int(round(float(PARAMS["side_min_s"]) / hop_s))
    min_frames = int(round(float(PARAMS["min_token_s"]) / hop_s))
    steady = speech & (flux < float(PARAMS["flux_max"]))
    steady_runs = runs(steady)
    for idx in range(1, len(steady_runs) - 1):
        start, stop, value = steady_runs[idx]
        if value == "False" and stop - start <= bridge and steady_runs[idx - 1][2] == "True":
            steady[start:stop] = speech[start:stop].all()
    tokens = []
    for start, stop, value in runs(steady):
        if value != "True" or stop - start < min_frames:
            continue
        side = np.where(score[start:stop] > float(PARAMS["ai_boundary_db"]), "a", "i")
        side = absorb_short_runs(side, side_min)
        for p0, p1, _ in runs(side):
            if p1 - p0 < min_frames:
                continue
            median = float(np.median(score[start + p0 : start + p1]))
            label = "a" if median > float(PARAMS["ai_boundary_db"]) else "i"
            tokens.append((start + p0, start + p1, label))
    return tokens


# ------------------------------------------------------------------ descriptors
def lpc_formants(segment: np.ndarray) -> Tuple[float, float]:
    """Median F1 and F2 (Hz) over 25 ms frames of a 10 kHz, order-12 LPC fit."""
    import librosa

    sr = int(PARAMS["sr"])
    target = int(PARAMS["lpc_sr"])
    x = resample_poly(segment, target // 1000 * 1, sr // 1000 * 1) if sr != target else segment
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    frame, hop = int(0.025 * target), int(0.010 * target)
    window = np.hamming(frame)
    f1s, f2s = [], []
    for start in range(0, x.size - frame, hop * 3):
        chunk = x[start : start + frame] * window
        if np.sum(chunk**2) < 1e-10:
            continue
        try:
            coeffs = librosa.lpc(chunk, order=int(PARAMS["lpc_order"]))
        except Exception:
            continue
        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) > 0]
        freqs = np.angle(roots) * target / (2 * np.pi)
        bandwidths = -target / np.pi * np.log(np.abs(roots))
        keep = np.sort(freqs[(freqs > 90) & (freqs < 4500) & (bandwidths < 400)])
        if keep.size >= 2:
            f1s.append(keep[0])
            f2s.append(keep[1])
    if not f1s:
        return float("nan"), float("nan")
    return float(np.median(f1s)), float(np.median(f2s))


def f0_slow_range_st(f0: np.ndarray, a: int, b: int) -> float:
    """Range of the slow F0 movement across a token, in semitones (NaN under half voiced).

    F0 is folded into one octave around its median (pYIN jumps an octave now
    and then on a rough voice), median-filtered over ``glide_smooth_s`` so that
    tremor and vibrato (4-8 Hz) average out, and the 5th-95th percentile range
    of what is left is returned. A glide moves the pitch slowly over several
    semitones, up, down or up and down; a straight-line trend missed the
    rise-and-fall glide of BL10, whose ends sit at the same pitch.
    """
    seg = f0[a:b]
    ok = np.isfinite(seg)
    if seg.size == 0 or ok.mean() < 0.5:
        return float("nan")
    semis = 12 * np.log2(seg[ok] / np.median(seg[ok]))
    semis = semis - 12 * np.round(semis / 12)
    hop_s = int(PARAMS["hop"]) / int(PARAMS["sr"])
    k = int(round(float(PARAMS["glide_smooth_s"]) / hop_s)) | 1
    if semis.size > k:
        semis = medfilt(semis, k)[k // 2 : -(k // 2)]
    return float(np.percentile(semis, 95) - np.percentile(semis, 5))


def f0_stats(f0: np.ndarray, voiced: np.ndarray, a: int, b: int) -> Tuple[float, float, float]:
    """Voiced ratio, median F0 (Hz) and F0 SD in semitones over the central 80 %."""
    seg_v = voiced[a:b]
    seg_f = f0[a:b]
    ratio = float(np.mean(seg_v)) if seg_v.size else float("nan")
    n = seg_f.size
    core = seg_f[int(0.1 * n) : int(0.9 * n)]
    core = core[np.isfinite(core)]
    if core.size < 10:
        return ratio, float("nan"), float("nan")
    semis = 12 * np.log2(core / np.median(core))
    return ratio, float(np.median(core)), float(np.std(semis))


# ------------------------------------------------------------------- per file
def segment_file(file_id: str, wav_file: str, write_cache: bool = True) -> Dict[str, object]:
    import kudio

    y, native_rate, duration = load_mono_16k(AUDIO / wav_file, file_id)
    sr, hop = int(PARAMS["sr"]), int(PARAMS["hop"])
    hop_s = hop / sr
    tracks = frame_tracks(y)
    n_frames = tracks["flux"].size
    spans = kudio.vad(y, sr, **PARAMS["vad"])  # type: ignore[arg-type]
    speech = spans_to_mask(spans, n_frames)
    track = kudio.f0(y, sr, fmin=float(PARAMS["f0_fmin"]), fmax=float(PARAMS["f0_fmax"]), hop_length=hop)
    f0 = np.full(n_frames, np.nan)
    voiced = np.zeros(n_frames, dtype=bool)
    m = min(n_frames, len(track.f0))
    f0[:m] = track.f0[:m]
    voiced[:m] = track.voiced[:m]

    tokens = find_tokens(tracks["flux"], tracks["score"], speech)
    token_rows = []
    for idx, (a, b, lab) in enumerate(tokens):
        start_s, end_s = a * hop_s, b * hop_s
        ratio, f0_med, f0_sd = f0_stats(f0, voiced, a, b)
        f1, f2 = lpc_formants(y[int(start_s * sr) : int(end_s * sr)])
        token_rows.append(
            {
                "file_id": file_id,
                "seg_idx": idx,
                "material": lab,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "score_median": float(np.median(tracks["score"][a:b])),
                "flux_median": float(np.median(tracks["flux"][a:b])),
                "voiced_ratio": ratio,
                "f0_median_hz": f0_med,
                "f0_sd_st": f0_sd,
                "f1_hz": f1,
                "f2_hz": f2,
                "f0_slow_range_st": f0_slow_range_st(f0, a, b),
                "low_confidence": bool(
                    abs(float(np.median(tracks["score"][a:b])) - float(PARAMS["ai_boundary_db"]))
                    < float(PARAMS["ai_margin_db"])
                ),
            }
        )

    # Sentences: speech with every token cut out (plus a margin), in blocks that no
    # token interrupts. Cutting rather than dropping whole spans matters when the
    # pauses are short: in NYU1015 one VAD span covers the sentences and the /a/.
    token_spans = sorted((r["start_s"], r["end_s"]) for r in token_rows)
    margin = float(PARAMS["sentence_token_margin_s"])
    pieces: List[Tuple[float, float]] = []
    for a, b in spans:
        parts = [(a, b)]
        for ta, tb in token_spans:
            cut_a, cut_b = ta - margin, tb + margin
            parts = [
                piece
                for pa, pb in parts
                for piece in ((pa, min(pb, cut_a)), (max(pa, cut_b), pb))
                if piece[1] - piece[0] > 0
            ]
        pieces.extend(p for p in parts if p[1] - p[0] >= float(PARAMS["vad"]["min_speech"]))  # type: ignore[index]
    pieces.sort()
    blocks: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for a, b in pieces:
        if current and any(ta >= current[-1][1] - margin and tb <= a + margin for ta, tb in token_spans):
            blocks.append(current)
            current = []
        current.append((a, b))
    if current:
        blocks.append(current)
    sentence = None
    if blocks:
        best = max(blocks, key=lambda blk: sum(b - a for a, b in blk))
        gaps = [best[k + 1][0] - best[k][1] for k in range(len(best) - 1)]
        sentence = {
            "start_s": round(best[0][0], 3),
            "end_s": round(best[-1][1], 3),
            "n_spans": len(best),
            "speech_s": round(sum(b - a for a, b in best), 3),
            "max_gap_s": round(max(gaps), 3) if gaps else 0.0,
            "n_blocks": len(blocks),
        }

    intervals = []
    for material in ("a", "i"):
        cands = [r for r in token_rows if r["material"] == material]
        row = {"file_id": file_id, "material": material, "start_s": np.nan, "end_s": np.nan, "status": "invalid", "reason": ""}
        if not cands:
            row["reason"] = "no token"
        else:
            def is_glide(r):
                return np.isfinite(r["f0_slow_range_st"]) and r["f0_slow_range_st"] > float(PARAMS["glide_st"])

            # A steady token wins over a longer glide only when it is long enough to
            # leave a full second after trimming; otherwise a 0.6 s scrap would beat
            # a 2 s vowel whose pitch happens to drift.
            steady = [
                r for r in cands
                if not is_glide(r) and r["duration_s"] >= float(PARAMS["steady_min_s"])
            ]
            tok = max(steady or cands, key=lambda r: r["duration_s"])
            start = tok["start_s"] + float(PARAMS["trim_s"])
            end = tok["end_s"] - float(PARAMS["trim_s"])
            if end - start > float(PARAMS["max_interval_s"]):
                mid = 0.5 * (start + end)
                start, end = mid - 0.5 * float(PARAMS["max_interval_s"]), mid + 0.5 * float(PARAMS["max_interval_s"])
            row.update({"start_s": round(start, 3), "end_s": round(end, 3), "source_seg": tok["seg_idx"]})
            if end - start >= float(PARAMS["min_interval_s"]):
                row["status"] = "valid"
                flags = []
                if end - start < 1.0:
                    flags.append("under_1s")
                if tok["low_confidence"]:
                    flags.append("low_confidence")
                if is_glide(tok):
                    flags.append("glide")
                row["reason"] = "+".join(flags)
            else:
                row["reason"] = "short"
        intervals.append(row)
    srow = {"file_id": file_id, "material": "sentences", "start_s": np.nan, "end_s": np.nan, "status": "invalid", "reason": ""}
    if sentence is None:
        srow["reason"] = "no speech outside tokens"
    else:
        srow.update({"start_s": sentence["start_s"], "end_s": sentence["end_s"]})
        flags = []
        if sentence["n_spans"] > int(PARAMS["sentence_many_spans"]):
            flags.append("many_spans")
        if sentence["max_gap_s"] > float(PARAMS["sentence_long_gap_s"]):
            flags.append("long_gap")
        if sentence["speech_s"] >= float(PARAMS["sentence_min_speech_s"]):
            srow["status"] = "valid"
        else:
            flags.insert(0, "too_little_speech")
        srow["reason"] = "+".join(flags)
    intervals.append(srow)
    intervals.append({"file_id": file_id, "material": "whole", "start_s": 0.0, "end_s": round(duration, 3), "status": "valid", "reason": ""})
    for row in intervals:
        row["duration_s"] = round(row["end_s"] - row["start_s"], 3) if np.isfinite(row["start_s"]) else np.nan

    file_flags = []
    if not token_rows:
        file_flags.append("no_tokens")
    if any(r["low_confidence"] for r in token_rows):
        file_flags.append("low_confidence_token")
    for r in token_rows:
        if r["material"] == "a" and np.isfinite(r["f1_hz"]) and r["f1_hz"] < 450:
            file_flags.append(f"a_low_f1@{r['seg_idx']}")
        if r["material"] == "i" and np.isfinite(r["f1_hz"]) and r["f1_hz"] > 550:
            file_flags.append(f"i_high_f1@{r['seg_idx']}")

    if write_cache:
        CACHE.mkdir(exist_ok=True)
        decim = tracks["logmel"].astype(np.float16)
        np.savez_compressed(
            CACHE / f"{file_id}.npz",
            hop_s=hop_s,
            duration=duration,
            energy_db=tracks["energy_db"].astype(np.float32),
            flux=tracks["flux"].astype(np.float32),
            score=tracks["score"].astype(np.float32),
            f0=f0.astype(np.float32),
            voiced=voiced,
            logmel=decim,
            spans=np.array(spans, dtype=np.float64).reshape(-1, 2),
            waveform=resample_poly(y, 1, 20).astype(np.float32),  # 800 Hz envelope view
        )
    return {
        "file_id": file_id,
        "native_rate": native_rate,
        "duration_s": round(duration, 3),
        "n_spans": len(spans),
        "tokens": token_rows,
        "sentence": sentence,
        "intervals": intervals,
        "flags": "+".join(file_flags),
    }


def _worker(args: Tuple[str, str]) -> Dict[str, object]:
    file_id, wav_file = args
    try:
        return segment_file(file_id, wav_file)
    except Exception as exc:  # keep the row; a failure is a result too
        return {"file_id": file_id, "error": f"{type(exc).__name__}: {exc}"}


def provenance() -> Dict[str, object]:
    import kudio
    import librosa

    kudio_dir = Path(kudio.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(kudio_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = ""
    return {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "params": PARAMS,
        "channel_choice": CHANNEL_CHOICE,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "kudio": kudio.__version__,
        "kudio_commit": commit,
        "soundfile": sf.__version__,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="*", help="file_ids to process (default: all)")
    parser.add_argument("--out", default=str(HERE), help="output folder")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    ids = pd.read_csv(ROOT / "s0_audit" / "out" / "id_map.csv")
    if args.files:
        ids = ids[ids["file_id"].isin(args.files)]
    jobs = list(zip(ids["file_id"], ids["wav_file"]))
    with Pool(max(1, args.workers)) as pool:
        results = pool.map(_worker, jobs, chunksize=1)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    segments = pd.DataFrame([t for r in ok for t in r["tokens"]])
    sentences = pd.DataFrame([{"file_id": r["file_id"], **r["sentence"]} for r in ok if r["sentence"]])
    intervals = pd.DataFrame([row for r in ok for row in r["intervals"]])
    files = pd.DataFrame(
        [{k: r[k] for k in ("file_id", "native_rate", "duration_s", "n_spans", "flags")} for r in ok]
        + [{"file_id": r["file_id"], "flags": "error", "error": r["error"]} for r in errors]
    )
    segments.to_csv(out / "segments.csv", index=False, lineterminator="\n")
    sentences.to_csv(out / "sentences.csv", index=False, lineterminator="\n")
    intervals.to_csv(out / "analysis_intervals.csv", index=False, lineterminator="\n")
    files.to_csv(out / "files.csv", index=False, lineterminator="\n")
    (out / "run_meta.json").write_text(json.dumps(provenance(), indent=1, default=str), encoding="utf-8")
    valid = intervals.pivot_table(index="file_id", columns="material", values="status", aggfunc="first")
    print(f"files {len(ok)} ok, {len(errors)} errors")
    for material in ("a", "i", "sentences", "whole"):
        if material in valid:
            print(f"  {material:9s} valid {int((valid[material] == 'valid').sum())}")
    for r in errors:
        print("  ERROR", r["file_id"], r["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
