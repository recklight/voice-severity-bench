"""S1 review figures: one PNG per recording from the segmentation cache.

Needs matplotlib (the voice-severity-bench environment); reads only ``cache/*.npz``
and the CSV files written by ``segment.py``, never the audio or the ratings.

Usage::

    python figures.py --run dev                      # every file of a run folder
    python figures.py --run . --files SJ1001 PT123   # selected files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
COLORS = {"a": "#d55e00", "i": "#0072b2", "other": "#7f7f7f", "sentences": "#009e73"}
THRESH = {"flux_max": 5.0, "ai_boundary_db": -17.5}


def draw(file_id: str, run: Path, out: Path) -> Path:
    data = np.load(HERE / "cache" / f"{file_id}.npz")
    hop = float(data["hop_s"])
    n = data["flux"].size
    t = np.arange(n) * hop
    segs = pd.read_csv(run / "segments.csv")
    segs = segs[segs["file_id"] == file_id]
    intervals = pd.read_csv(run / "analysis_intervals.csv")
    intervals = intervals[intervals["file_id"] == file_id]
    files = pd.read_csv(run / "files.csv").set_index("file_id")
    flags = files.loc[file_id, "flags"] if file_id in files.index else ""
    flags = "" if pd.isna(flags) else str(flags)

    fig = Figure(figsize=(13, 8), dpi=80)
    axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2.4, 1.2, 1.2]})
    wave = data["waveform"]
    axes[0].plot(np.arange(wave.size) / 800.0, wave, lw=0.4, color="black")
    for a, b in data["spans"]:
        axes[0].axvspan(a, b, color="#cccccc", alpha=0.6, lw=0)
    axes[0].set_ylabel("wave\n+VAD")

    img = data["logmel"].astype(np.float32)
    vmax = np.percentile(img, 99.5)
    axes[1].imshow(img, origin="lower", aspect="auto", extent=[0, n * hop, 0, img.shape[0]],
                   vmin=vmax - 70, vmax=vmax, cmap="magma")
    for row in segs.itertuples():
        axes[1].add_patch(Rectangle((row.start_s, 0.5), row.end_s - row.start_s, img.shape[0] - 1,
                                    fill=False, ec=COLORS.get(row.material, "white"), lw=2))
        axes[1].text(row.start_s, img.shape[0] - 3, row.material, color="white", fontsize=11, weight="bold")
    for row in intervals.itertuples():
        if row.status == "valid" and row.material in ("a", "i", "sentences"):
            axes[1].plot([row.start_s, row.end_s], [1.5, 1.5], color=COLORS[row.material], lw=6)
    axes[1].set_ylabel("log-mel\n(40 bands)")

    axes[2].plot(t, data["score"], lw=0.8, color="black")
    axes[2].axhline(THRESH["ai_boundary_db"], color="purple", ls="--", lw=0.8)
    axes[2].set_ylim(-75, 50)
    axes[2].set_ylabel("vowel score\n(dB)")

    axes[3].plot(t, data["flux"], lw=0.8, color="black", label="flux")
    axes[3].axhline(THRESH["flux_max"], color="red", ls="--", lw=0.8)
    voiced = data["voiced"].astype(float)
    axes[3].fill_between(t, 0, voiced * 0.8, color="#56b4e9", alpha=0.5, step="mid", label="voiced")
    axes[3].set_ylim(0, 12)
    axes[3].set_ylabel("envelope flux\n+voiced")
    axes[3].set_xlabel("time (s)")

    summary = []
    for row in intervals.itertuples():
        if row.material == "whole":
            continue
        if row.status == "valid":
            summary.append(f"{row.material} {row.start_s:.2f}-{row.end_s:.2f}")
        else:
            summary.append(f"{row.material} INVALID({row.reason})")
    fig.suptitle(f"{file_id}   {'  |  '.join(summary)}   {('flags: ' + flags) if flags else ''}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{file_id}.png"
    fig.savefig(path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=".", help="folder holding segments.csv etc.")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    run = (HERE / args.run).resolve()
    ids = args.files or sorted(pd.read_csv(run / "files.csv")["file_id"])
    out = Path(args.out) if args.out else run / "figures"
    for file_id in ids:
        draw(file_id, run, out)
    print(f"{len(ids)} figures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
