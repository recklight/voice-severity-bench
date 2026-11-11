"""Experiments E1-E7 of analysis_plan.md (S5).

    .venv/Scripts/python.exe experiments/harness.py list
    .venv/Scripts/python.exe experiments/harness.py run E1 --jobs 8
    .venv/Scripts/python.exe experiments/harness.py run all --jobs 8
    .venv/Scripts/python.exe experiments/harness.py smoke --jobs 8

``run`` refuses to start unless verify_freeze() finds everything as
freeze.json recorded it. Each job writes its tables and a meta.json under
results/<experiment>/<job>/; a job that fails leaves FAILED.json with the
traceback and the batch goes on. ``smoke`` runs the two pre-freeze checks
(a planted signal must be found, shuffled targets must give RMSE near the
target SD) and writes to results_smoke/, never to results/.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "features"))
import common  # noqa: E402

from voice_severity_bench import (  # noqa: E402
    Config,
    FeatureSet,
    compare_feature_sets,
    fit_model,
    load_features,
    pairwise_contrasts,
    regression_metrics,
    run_ablation,
)

RESULTS = HERE / "results"
SMOKE = HERE / "results_smoke"
DRY_RESULTS = HERE / "results_dry"
#: Set by the dry command: targets permuted across files, reduced repeats, no freeze check.
DRY = False
FREEZE = HERE / "freeze.json"
PLAN = common.plan()
SSL = ("A4", "A5", "A6")
MAIN_SETS = ("B0", "M0", "A1", "A2", "A3", "A4", "A5", "A6")
CONTRASTS = [tuple(pair) for pair in PLAN["confirmatory"]["contrasts"]]
TARGETS = [PLAN["targets"]["primary"], *PLAN["targets"]["secondary"]]
M0_GROUP_COLUMNS = [f"group_{name}" for name in PLAN["data"]["recording_groups"]]


# ------------------------------------------------------------------ jobs


@dataclass(frozen=True)
class Job:
    """One comparison (or ablation) run; its fields are hashed into freeze.json."""

    experiment: str
    name: str
    material: str = "a"
    target: str = PLAN["targets"]["primary"]
    subset: str = "all"
    sets: tuple[str, ...] = MAIN_SETS
    contrasts: tuple[tuple[str, str], ...] = tuple(CONTRASTS)
    groups: str = "merged"
    kind: str = "compare"
    options: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def folder(self) -> Path:
        return (DRY_RESULTS if DRY else RESULTS) / self.experiment / self.name


def jobs() -> list[Job]:
    out = [Job("E1", "primary")]
    for model in SSL:
        for target in TARGETS:
            blocks = tuple(f"{model}@{b}" for b in block_labels())
            out.append(Job("E2", f"{model}_{target}", target=target, sets=("B0", *blocks), contrasts=()))
    for target in PLAN["targets"]["secondary"]:
        out.append(Job("E3", target, target=target))
    for material in PLAN["materials"]["exploratory"]:
        out.append(Job("E4", material, material=material))
    out += [
        Job("E5", "E5a_pt_only", subset="PT", sets=("B0", "M0pt", *MAIN_SETS[2:]),
            contrasts=tuple(_rename(CONTRASTS, {"M0": "M0pt"}))),
        Job("E5", "E5b_leave_group_out", kind="logo", contrasts=()),
        Job("E5", "E5c_female", subset="sex=F"),
        Job("E5", "E5c_male", subset="sex=M"),
        Job("E5", "E5d_first_session", target=PLAN["targets"]["primary"] + PLAN["targets"]["first_session_suffix"]),
        Job("E5", "E5e_interval_flags", subset="no_interval_flags"),
        Job("E5", "E5e_recording_flags", subset="no_recording_flags"),
        Job("E5", "E5e_rater_disagreement", subset="rater_sd_below_20"),
        Job("E5", "E5f_native_rate", sets=(*MAIN_SETS, "A3_native"),
            contrasts=(("A3_native", "A3"), *_rename([c for c in CONTRASTS if c[1] == "A3"], {"A3": "A3_native"}))),
        Job("E5", "E5g_plus_m0", sets=("B0", "M0", *(f"{s}+M0" for s in MAIN_SETS[2:])),
            contrasts=tuple((f"{s}+M0", "M0") for s in MAIN_SETS[2:])),
        Job("E5", "E5i_one_file_groups", groups="files"),
        Job("E7", "mfcc_settings", kind="ablation", sets=(), contrasts=()),
    ]  # fmt: skip
    return out


def _rename(pairs, mapping):
    return [tuple(mapping.get(name, name) for name in pair) for pair in pairs]


def block_labels() -> list[str]:
    return [f"L{k:02d}" for k in range(int(PLAN["feature_sets"]["A4"]["n_hidden_states"]))] + ["AVG"]


def jobs_hash() -> str:
    payload = json.dumps([dataclasses.asdict(job) for job in jobs()], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------------------ configuration


def config(target: str, **extra: Any) -> Config:
    space = PLAN["model"]["ridge_alphas"]["logspace"]
    alphas = [float(a) for a in np.logspace(space[0], space[1], int(space[2]))]
    settings = {
        "seed": int(PLAN["master_seed"]),
        "data.target": target,
        "model.name": "ridge",
        "model.tune": True,
        "model.impute": "median",
        "model.block_selection": True,
        "model.ridge_alphas": alphas,
        "evaluation.n_splits": int(PLAN["evaluation"]["outer_splits"]),
        "evaluation.n_inner_splits": int(PLAN["model"]["inner_splits"]),
        "evaluation.n_repeats": int(PLAN["evaluation"]["n_repeats"]),
        "analysis.n_resamples": int(PLAN["confirmatory"]["bootstrap"]["n_resamples"]),
        "analysis.ci_level": float(PLAN["confirmatory"]["bootstrap"]["ci_level"]),
        "analysis.correction": PLAN["confirmatory"]["correction"],
    }
    if DRY:
        settings.update({"evaluation.n_repeats": 2, "analysis.n_resamples": 500})
    settings.update(extra)
    return Config().override(settings)


# ------------------------------------------------------------------ feature files


def _file_ids(material: str) -> pd.DataFrame:
    """Index rows of a material (path, group, interval, file_id, recording_group)."""
    return common.index(material)


def _base(material: str, name: str) -> FeatureSet:
    return load_features(common.OUT / material / f"{name}.npz")


def _hstack(first: FeatureSet, second: FeatureSet, label: str) -> FeatureSet:
    """Columns of ``first`` then ``second`` (same rows); blocks kept, ``second`` as one block."""
    assert first.keys() == second.keys()
    width = first.X.shape[1]
    blocks = dict(first.blocks or {"main": (0, width)})
    blocks[label] = (width, width + second.X.shape[1])
    return dataclasses.replace(
        first,
        X=np.hstack([first.X, second.X]),
        names=(*first.feature_names, *(f"{label}:{n}" for n in second.feature_names)),
        blocks=blocks,
        kind=f"{first.kind}+{second.kind}",
    )


def feature_set(material: str, name: str) -> FeatureSet:
    """A named feature file of the plan, including the derived ones (B0, M0pt, A4@L05, A1+M0)."""
    if name == "B0":
        # One constant column: after standardization ridge predicts the training mean.
        m0 = _base(material, "M0")
        return dataclasses.replace(m0, X=np.ones((m0.X.shape[0], 1)), names=("const",),
                                   kind="mean", blocks=None)  # fmt: skip
    if name == "M0pt":
        m0 = _base(material, "M0")
        keep = [i for i, n in enumerate(m0.feature_names) if n not in M0_GROUP_COLUMNS]
        return dataclasses.replace(m0, X=m0.X[:, keep],
                                   names=tuple(m0.feature_names[i] for i in keep))  # fmt: skip
    if "@" in name:
        model, block = name.split("@")
        return _base(material, model).select_blocks([block])
    if name.endswith("+M0"):
        return _hstack(_base(material, name[: -len("+M0")]), _base(material, "M0"), "M0")
    return _base(material, name)


def block_candidates(sets: dict[str, FeatureSet]) -> dict[str, list]:
    """For A4+M0 etc.: every layer together with the M0 block (plan E5g)."""
    out = {}
    for name, features in sets.items():
        if name.endswith("+M0") and features.blocks and len(features.blocks) > 2:
            out[name] = [(block, "M0") for block in features.blocks if block != "M0"]
    return out


def subset_mask(job: Job, ids: pd.DataFrame) -> np.ndarray:
    file_id = ids["file_id"]
    if job.subset == "all":
        return np.ones(len(ids), dtype=bool)
    if job.subset == "PT":
        return (ids["recording_group"] == "PT").to_numpy()
    if job.subset.startswith("sex="):
        sex = pd.read_csv(common.ROOT / "s0_audit" / "out" / "demographics_clean.csv")
        sex = sex.dropna(subset=["file_id"]).set_index("file_id")["sex"]
        return (file_id.map(sex) == job.subset.split("=")[1]).to_numpy()
    if job.subset == "no_interval_flags":
        flags = pd.read_csv(common.ROOT / "s1_segments" / "analysis_set.csv").set_index("file_id")
        flagged = flags["a_flags"].fillna("").astype(str) != ""
        return (~file_id.map(flagged).astype(bool)).to_numpy()
    if job.subset == "no_recording_flags":
        spec = PLAN["experiments"]["E5"]["E5e"]["recording_flags"]
        rec = common.recordings()
        bad = (rec["clip_fraction"] > spec["clip_fraction_above"]) | (rec["channels"] > 1)
        return (~file_id.map(bad).astype(bool)).to_numpy()
    if job.subset == "rater_sd_below_20":
        limit = PLAN["experiments"]["E5"]["E5e"]["rater_sd_at_least"]
        sd = common.targets()[f"{PLAN['targets']['primary']}__sd"]
        return (file_id.map(sd) < limit).to_numpy()
    raise ValueError(f"unknown subset {job.subset!r}")


def load_job_sets(job: Job) -> dict[str, FeatureSet]:
    ids = _file_ids(job.material)
    mask = subset_mask(job, ids)
    y = common.targets(DRY).loc[ids["file_id"], job.target].to_numpy(dtype=float)
    if job.groups == "merged":
        groups = ids["group"].to_numpy().astype(str)
    else:
        groups = ids["file_id"].to_numpy().astype(str)
    sets = {}
    for name in job.sets:
        features = feature_set(job.material, name)
        if features.keys() != [
            (p, None if np.isnan(a) else a, None if np.isnan(b) else b)
            for p, a, b in zip(ids["path"], ids["start_s"], ids["end_s"], strict=True)
        ]:
            raise RuntimeError(f"{name} ({job.material}) is not in index order")
        features = dataclasses.replace(features, y=y, groups=groups)
        sets[name] = features.take(np.flatnonzero(mask))
    return sets


# ------------------------------------------------------------------ runners


def run_compare(job: Job, n_jobs: int, cfg: Config | None = None, folder: Path | None = None) -> dict:
    sets = load_job_sets(job)
    cfg = cfg or config(job.target)
    result = compare_feature_sets(sets, cfg, block_candidates=block_candidates(sets), n_jobs=n_jobs)
    folder = folder or job.folder
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in (("folds", result.folds), ("items", result.items),
                        ("predictions", result.predictions), ("summary", result.summary)):  # fmt: skip
        table.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
    if job.contrasts:
        contrasts = pairwise_contrasts(
            result,
            [list(pair) for pair in job.contrasts],
            correction=PLAN["confirmatory"]["correction"],
            n_resamples=cfg.analysis.n_resamples,
            ci_level=cfg.analysis.ci_level,
            equivalence_margin=float(PLAN["targets"]["smallest_meaningful_difference"]),
            equivalence_ci_level=float(PLAN["confirmatory"]["equivalence_ci_level"]),
            seed=int(PLAN["confirmatory"]["bootstrap"]["seed"]),
        )
        contrasts.to_csv(folder / "contrasts.csv", index=False, lineterminator="\n")
    return {"n_items": len(result.items), "best": result.as_dict()}


def run_logo(job: Job, n_jobs: int) -> dict:
    """E5b: train on four recording groups, test on the fifth, for every set."""
    from joblib import Parallel, delayed

    sets = load_job_sets(job)
    ids = _file_ids(job.material)
    groups_rec = ids["recording_group"].to_numpy()
    cfg = config(job.target)
    tasks = [(name, held) for name in sets for held in PLAN["data"]["recording_groups"]]

    def one(name, held):
        features = sets[name]
        test = groups_rec == held
        model = fit_model(features.X[~test], features.y[~test], cfg,
                          groups=features.groups[~test], blocks=features.blocks)  # fmt: skip
        predicted = model.predict(features.X[test])
        return name, held, np.flatnonzero(test), predicted

    outputs = Parallel(n_jobs=n_jobs)(delayed(one)(n, h) for n, h in tasks)
    rows, preds = [], []
    for name, held, items, predicted in outputs:
        truth = sets[name].y[items]
        scores = regression_metrics(truth, predicted)
        rows.append({"feature_set": name, "held_out": held, "n_test": int(items.size),
                     "bias": float(np.mean(predicted - truth)), **scores})  # fmt: skip
        preds.append(pd.DataFrame({"feature_set": name, "held_out": held, "item": items,
                                   "y_true": truth, "y_pred": predicted}))  # fmt: skip
    job.folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(job.folder / "summary.csv", index=False, lineterminator="\n")
    pd.concat(preds).to_csv(job.folder / "predictions.csv", index=False, lineterminator="\n")
    return {"n_runs": len(rows)}


def run_ablation_job(job: Job, n_jobs: int) -> dict:
    """E7: the 0.1.0 MFCC grid on the /a/ intervals (package default ridge grid)."""
    spec = PLAN["experiments"]["E7"]
    cfg = Config().override(
        {
            "seed": int(PLAN["master_seed"]),
            "data.target": PLAN["targets"]["primary"],
            "audio.sample_rate": int(PLAN["audio"]["sample_rate"]),
            "audio.rms_dbfs": float(PLAN["audio"]["rms_dbfs"]),
            "ablation.n_ceps": spec["ablation"]["n_ceps"],
            "ablation.delta_order": spec["ablation"]["delta_order"],
            "ablation.models": spec["ablation"]["models"],
            "ablation.n_repeats": int(spec["n_repeats"]),
            "model.tune": bool(spec["model_tune"]),
            "evaluation.n_splits": int(PLAN["evaluation"]["outer_splits"]),
            "evaluation.n_inner_splits": int(PLAN["model"]["inner_splits"]),
        }
    )
    dataset = common.INDEX / "a.csv"
    if DRY:
        # Same recordings and folds, permuted targets, one repeat.
        index = pd.read_csv(dataset)
        index[PLAN["targets"]["primary"]] = (
            common.targets(True).loc[index["file_id"], PLAN["targets"]["primary"]].to_numpy()
        )
        dataset = HERE / "cache" / "e7_dry_index.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(dataset, index=False, lineterminator="\n")
        cfg = cfg.override({"ablation.n_repeats": 1})
    result = run_ablation(dataset, cfg, cache_dir=HERE / "cache" / "e7")
    job.folder.mkdir(parents=True, exist_ok=True)
    result.folds.to_csv(job.folder / "folds.csv", index=False, lineterminator="\n")
    result.summary.to_csv(job.folder / "summary.csv", index=False, lineterminator="\n")
    result.comparisons.to_csv(job.folder / "comparison.csv", index=False, lineterminator="\n")
    return result.as_dict()


RUNNERS = {"compare": run_compare, "logo": run_logo, "ablation": run_ablation_job}


def execute(job: Job, n_jobs: int) -> None:
    from freeze import verify_freeze

    freeze = "dry run" if DRY else verify_freeze()
    job.folder.mkdir(parents=True, exist_ok=True)
    (job.folder / "FAILED.json").unlink(missing_ok=True)
    start = time.perf_counter()
    try:
        details = RUNNERS[job.kind](job, n_jobs)
    except Exception:
        common.write_json(job.folder / "FAILED.json", {"job": dataclasses.asdict(job),
                                                       "traceback": traceback.format_exc()})  # fmt: skip
        print(f"FAILED {job.experiment}/{job.name}")
        return
    common.write_json(
        job.folder / "meta.json",
        {
            "job": dataclasses.asdict(job),
            "seconds": round(time.perf_counter() - start, 1),
            "freeze_sha256": freeze,
            "details": details,
        },
    )
    print(f"done {job.experiment}/{job.name} in {time.perf_counter() - start:.0f} s")


# ------------------------------------------------------------------ smoke tests


def smoke(n_jobs: int) -> int:
    """Pre-freeze checks on the real items, groups and pipeline; reduced repeats."""
    quick = {"evaluation.n_repeats": 2, "analysis.n_resamples": 1000}
    ids = _file_ids("a")
    rng = np.random.default_rng(20261104)
    y = rng.normal(30.0, 25.0, len(ids))
    base = feature_set("a", "M0")
    groups = ids["group"].to_numpy().astype(str)

    def synthetic(X, blocks=None, kind="smoke"):
        names = tuple(f"x{i}" for i in range(X.shape[1]))
        return dataclasses.replace(base, X=X, y=y, groups=groups, names=names, blocks=blocks, kind=kind)

    noise = rng.normal(size=(len(ids), 21))
    planted = noise.copy()
    planted[:, 0] = y / 25.0 + rng.normal(0, 0.5, len(ids))
    layers = rng.normal(size=(len(ids), 14 * 8))
    layers[:, 5 * 8] = y / 25.0 + rng.normal(0, 0.5, len(ids))
    blocks = {f"L{k:02d}": (8 * k, 8 * k + 8) for k in range(14)}
    sets = {"noise": synthetic(noise), "planted": synthetic(planted),
            "layers": synthetic(layers, blocks)}  # fmt: skip
    cfg = config("smoke_target", **quick)
    result = compare_feature_sets(sets, cfg, n_jobs=n_jobs)
    contrasts = pairwise_contrasts(result, [["planted", "noise"], ["layers", "noise"]],
                                   n_resamples=1000, seed=1)  # fmt: skip
    chosen = result.folds[result.folds["feature_set"] == "layers"]["params"].map(
        lambda text: json.loads(text)["select__block"]
    )
    SMOKE.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(SMOKE / "planted_summary.csv", index=False, lineterminator="\n")
    contrasts.to_csv(SMOKE / "planted_contrasts.csv", index=False, lineterminator="\n")
    found = bool((contrasts["ci_high"] < 0).all()) and (chosen == "L05").mean() >= 0.9
    print(result.summary[["feature_set", "rmse", "rmse_low", "rmse_high"]].to_string(index=False))
    print(contrasts[["first", "second", "difference", "ci_low", "ci_high"]].to_string(index=False))
    print(f"planted block chosen in {(chosen == 'L05').mean():.0%} of folds")

    # Shuffled targets: every set should do no better than the mean.
    job = Job("smoke", "shuffled")
    sets = load_job_sets(job)
    permuted = rng.permutation(len(ids))
    y_shuffled = sets["B0"].y[permuted]
    sets = {n: dataclasses.replace(f, y=y_shuffled) for n, f in sets.items()}
    result = compare_feature_sets(sets, config(job.target, **quick),
                                  block_candidates=block_candidates(sets), n_jobs=n_jobs)  # fmt: skip
    result.summary.to_csv(SMOKE / "shuffled_summary.csv", index=False, lineterminator="\n")
    sd = float(np.std(y_shuffled, ddof=1))
    ratio = result.summary.set_index("feature_set")["rmse"] / sd
    print(f"shuffled targets, RMSE / SD (SD = {sd:.2f}):")
    print(ratio.round(3).to_string())
    no_leak = bool((ratio > 0.97).all())
    verdict = {"planted_found": found, "no_leak": no_leak, "rmse_over_sd": ratio.round(4).to_dict()}
    common.write_json(SMOKE / "verdict.json", verdict)
    print(verdict)
    return 0 if found and no_leak else 1


# ------------------------------------------------------------------ command line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    runner = sub.add_parser("run")
    runner.add_argument("which", nargs="+", help="Experiment (E1..E7), experiment/job, or 'all'.")
    runner.add_argument("--jobs", type=int, default=8)
    smoker = sub.add_parser("smoke")
    smoker.add_argument("--jobs", type=int, default=8)
    dry = sub.add_parser("dry", help="Run jobs on permuted targets into results_dry/.")
    dry.add_argument("which", nargs="+")
    dry.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "list":
        for job in jobs():
            print(f"{job.experiment}/{job.name}  [{job.kind}, {job.material}, {job.target}, {job.subset}]")
        print(f"{len(jobs())} jobs, hash {jobs_hash()}")
        return 0
    if args.command == "smoke":
        return smoke(args.jobs)
    if args.command == "dry":
        global DRY
        DRY = True
    chosen = [
        job
        for job in jobs()
        if "all" in args.which or job.experiment in args.which or f"{job.experiment}/{job.name}" in args.which
    ]
    for job in chosen:
        execute(job, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
