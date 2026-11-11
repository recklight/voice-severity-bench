"""Freeze everything the results depend on, and check it before every run (S5).

    .venv/Scripts/python.exe experiments/freeze.py          # writes freeze.json once
    .venv/Scripts/python.exe experiments/freeze.py --check  # what verify_freeze() sees

freeze.json records the voice-severity-bench commit and the tree of its
src/, the installed packages, the model revisions and weight checksums, the
checksums of every input table, index and feature file, the two analysis-plan
files, the harness and analysis scripts, the hash of the job list and the
master seed. Freezing refuses to overwrite an existing freeze.json and
refuses while voice-severity-bench's src/ has uncommitted changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "features"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402

ROOT = common.ROOT
INPUTS = [
    "data/pvqd_manifest.json",
    "s0_audit/out/targets.csv",
    "s0_audit/out/ratings_long.csv",
    "s0_audit/out/recordings.csv",
    "s0_audit/out/demographics_clean.csv",
    "s0_audit/out/id_map.csv",
    "s0_audit/decisions.csv",
    "s1_segments/analysis_intervals.csv",
    "s1_segments/analysis_set.csv",
    "analysis_plan.md",
    "analysis_plan.yaml",
    "models/manifest.json",
    "features/common.py",
    "features/extract_mfcc_metadata.py",
    "features/extract_clinical.py",
    "features/extract_ssl.py",
    "experiments/harness.py",
    "experiments/freeze.py",
    "experiments/analyze.py",
]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(common.REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def snapshot() -> dict:
    """Everything freeze.json holds except the creation time."""
    import harness

    files = sorted(
        [ROOT / name for name in INPUTS]
        + list((ROOT / "features" / "index").glob("*.csv"))
        + list((ROOT / "features" / "out").glob("*/*.npz"))
    )
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze, missing: {missing}")
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {
        "voice_severity_bench": {
            "commit": _git("rev-parse", "HEAD"),
            "src_tree": _git("rev-parse", "HEAD:src"),
            "src_dirty": bool(_git("status", "--porcelain", "--", "src")),
        },
        "python": sys.version.split()[0],
        "pip_freeze": sorted(line for line in pip if not line.startswith("-e ")),
        "files": {p.relative_to(ROOT).as_posix(): common.sha256(p) for p in files},
        "jobs_sha256": harness.jobs_hash(),
        "n_jobs": len(harness.jobs()),
        "master_seed": int(common.plan()["master_seed"]),
    }


def freeze() -> Path:
    path = HERE / "freeze.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a freeze is never overwritten")
    state = snapshot()
    if state["voice_severity_bench"]["src_dirty"]:
        raise SystemExit("voice-severity-bench has uncommitted changes in src/; commit first")
    state["created"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def differences(recorded: dict, current: dict) -> list[str]:
    out = []
    for key in ("voice_severity_bench", "python", "pip_freeze", "jobs_sha256", "master_seed"):
        if recorded.get(key) != current.get(key):
            out.append(key)
    old, new = recorded.get("files", {}), current["files"]
    out += [f"file {name}" for name in sorted(set(old) | set(new)) if old.get(name) != new.get(name)]
    if current["voice_severity_bench"]["src_dirty"]:
        out.append("voice-severity-bench src/ has uncommitted changes")
    return out


def verify_freeze() -> str:
    """Stop unless the current state equals freeze.json; return the freeze file's SHA-256."""
    path = HERE / "freeze.json"
    if not path.is_file():
        raise SystemExit("no freeze.json; run experiments/freeze.py after the smoke tests")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    changed = differences(recorded, snapshot())
    if changed:
        raise SystemExit("state differs from freeze.json: " + "; ".join(changed[:10]))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        print("freeze ok", verify_freeze())
    else:
        print("written", freeze())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
