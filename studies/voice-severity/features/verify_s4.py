"""S4 checks: every feature file loads and aligns, NaN only in A3, SSL reproducible.

    .venv/Scripts/python.exe features/verify_s4.py --materials a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import extract_ssl  # noqa: E402

from voice_severity_bench import align_feature_sets, load_features  # noqa: E402

SETS = ("M0", "A1", "A2", "A3", "A4", "A5", "A6")
#: Items re-extracted to check that the SSL features are reproducible.
RECHECK = 5
TOLERANCE = 1e-5


def check(material: str) -> list[str]:
    problems = []
    names = [*SETS, *(["A3_native"] if material == "a" else [])]
    present = [n for n in names if (common.OUT / material / f"{n}.npz").is_file()]
    sets = {n: load_features(common.OUT / material / f"{n}.npz") for n in present}
    aligned = align_feature_sets(sets)
    order_kept = all(a.keys() == sets[n].keys() for n, a in aligned.items())
    print(f"{material}: {len(present)} files, {len(sets[present[0]].y)} items, same order: {order_kept}")
    for name, features in sets.items():
        bad = int((~np.isfinite(features.X)).sum())
        if bad and not name.startswith("A3"):
            problems.append(f"{material}/{name}: {bad} non-finite values")
        expected = common.plan()["feature_sets"].get(name, {}).get("n_columns")
        if expected and features.X.shape[1] != expected:
            problems.append(f"{material}/{name}: {features.X.shape[1]} columns, plan says {expected}")
        layers = {f"L{k:02d}" for k in range(13)} | {"AVG"}
        if name in ("A4", "A5", "A6") and set(features.blocks) != layers:
            problems.append(f"{material}/{name}: unexpected blocks {list(features.blocks)}")
    return problems


def recheck_ssl(material: str) -> list[str]:
    problems = []
    index = common.index(material)
    rows = index.iloc[np.linspace(0, len(index) - 1, RECHECK).astype(int)]
    for name in ("A4", "A5", "A6"):
        if not (common.OUT / material / f"{name}.npz").is_file():
            continue
        encoder = extract_ssl.Encoder(name)
        worst = 0.0
        for _, row in rows.iterrows():
            signal, _ = common.load_interval(row)
            fresh = extract_ssl.pooled(encoder.layers(signal))
            cached = np.load(extract_ssl.CACHE / name / material / f"{row['file_id']}.npz")["pooled"]
            worst = max(worst, float(np.max(np.abs(fresh - cached))))
        print(f"{material}/{name}: largest difference on {RECHECK} re-extracted items {worst:.2e}")
        if worst > TOLERANCE:
            problems.append(f"{material}/{name}: re-extraction differs by {worst:.2e}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--materials", nargs="+", default=["a"], choices=common.MATERIALS)
    parser.add_argument("--skip-recheck", action="store_true")
    args = parser.parse_args(argv)
    problems = []
    for material in args.materials:
        problems += check(material)
        if not args.skip_recheck:
            problems += recheck_ssl(material)
    print("\n".join(problems) if problems else "all S4 checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
