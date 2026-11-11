"""M0 (recording conditions), A1 (MFCC-13) and A2 (MFCC-39) for one or more materials.

    .venv/Scripts/python.exe features/extract_mfcc_metadata.py --materials a i sentences whole

M0 comes from s0_audit/out/recordings.csv and is the same for every
material; it is written once per material so its keys match that material's
other feature files. A1 and A2 are voice-severity-bench's ``prepare`` with
the package's default front end, delta order 0 and 2 (plan section 6).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from voice_severity_bench import (  # noqa: E402
    Config,
    feature_set_from_table,
    prepare_features,
    save_features,
)

M0_COLUMNS = [
    "group_BL",
    "group_LA",
    "group_NYU",
    "group_PT",
    "group_SJ",
    "is_25k",
    "is_stereo",
    "duration_s",
    "rms_dbfs",
    "noise_floor_dbfs",
]


def metadata(material: str, index: pd.DataFrame) -> None:
    rec = common.recordings().loc[index["file_id"]]
    table = index[["path", "group", "start_s", "end_s", common.PRIMARY_TARGET]].copy()
    for name in common.plan()["data"]["recording_groups"]:
        table[f"group_{name}"] = (index["recording_group"] == name).astype(float).to_numpy()
    table["is_25k"] = (rec["sample_rate"] == 25000).astype(float).to_numpy()
    table["is_stereo"] = (rec["channels"] > 1).astype(float).to_numpy()
    for column in ("duration_s", "rms_dbfs", "noise_floor_dbfs"):
        table[column] = rec[column].to_numpy(dtype=float)
    assert list(common.plan()["feature_sets"]["M0"]["columns"]) == M0_COLUMNS
    features = feature_set_from_table(
        table,
        target=common.PRIMARY_TARGET,
        feature_columns=M0_COLUMNS,
        kind="metadata",
        extractor=common.provenance(__file__, {"source": "s0_audit/out/recordings.csv"}),
    )
    save_features(common.OUT / material / "M0.npz", features)


def mfcc(material: str, name: str, delta_order: int) -> None:
    config = Config().override(
        {
            "data.target": common.PRIMARY_TARGET,
            "audio.sample_rate": int(common.plan()["audio"]["sample_rate"]),
            "audio.rms_dbfs": float(common.plan()["audio"]["rms_dbfs"]),
            "frontend.delta_order": delta_order,
        }
    )
    prepared = prepare_features(common.INDEX / f"{material}.csv", config)
    settings = {"audio": config.to_dict()["audio"], "frontend": config.to_dict()["frontend"],
                "pooling": config.to_dict()["pooling"]}  # fmt: skip
    features = dataclasses.replace(prepared, extractor=common.provenance(__file__, settings))
    expected = common.plan()["feature_sets"][name]["n_columns"]
    assert features.X.shape[1] == expected, (name, features.X.shape)
    assert np.isfinite(features.X).all(), f"{name}: non-finite MFCC statistics"
    save_features(common.OUT / material / f"{name}.npz", features)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--materials", nargs="+", default=["a"], choices=common.MATERIALS)
    args = parser.parse_args(argv)
    for material in args.materials:
        start = time.perf_counter()
        index = common.index(material)
        metadata(material, index)
        mfcc(material, "A1", 0)
        mfcc(material, "A2", 2)
        print(f"{material}: {len(index)} items, {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
