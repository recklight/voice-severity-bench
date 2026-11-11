"""The analysis set: one row per recording, which materials are valid and the /a/ flags.

    python analysis_set.py            # reads analysis_intervals.csv next to this script

Built from ``analysis_intervals.csv`` (written by ``segment.py``) and nothing
else, so it never sees a rating. ``primary_set`` is ``a_valid``: the primary
analysis needs only a valid /a/, because requiring all four materials would
drop every PT and NYU recording, which have no /i/. ``a_flags`` is the reason
segment.py gives for the /a/ interval (``glide``, ``under_1s``,
``low_confidence``, or why it is invalid).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def analysis_set(intervals: pd.DataFrame) -> pd.DataFrame:
    valid = intervals.pivot_table(
        index="file_id", columns="material", values="status", aggfunc="first"
    ).eq("valid")
    reason = intervals[intervals["material"] == "a"].set_index("file_id")["reason"]
    table = pd.DataFrame(
        {
            "a_valid": valid["a"],
            "i_valid": valid["i"],
            "sentences_valid": valid["sentences"],
            "whole_valid": valid["whole"],
            "a_flags": reason.reindex(valid.index),
            "primary_set": valid["a"],
        }
    )
    table.index.name = "file_id"
    return table.sort_index()


def main() -> int:
    intervals = pd.read_csv(HERE / "analysis_intervals.csv", keep_default_na=False)
    table = analysis_set(intervals)
    table.to_csv(HERE / "analysis_set.csv", lineterminator="\n")
    print(f"{len(table)} recordings, {int(table['primary_set'].sum())} in the primary set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
