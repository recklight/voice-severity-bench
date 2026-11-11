"""Tables, figures and summary.md from the experiment results (S6, written before the freeze).

    .venv/Scripts/python.exe experiments/analyze.py
    .venv/Scripts/python.exe experiments/analyze.py --results results_dry --out analysis_dry --dry

``--dry`` permutes every target across files with a fixed seed before
anything is computed, so the code can be exercised on the dry-run results
without showing a single real result before the freeze.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "features"))
import common  # noqa: E402

from voice_severity_bench import (  # noqa: E402
    bootstrap_metric,
    icc_oneway,
    leave_one_rater_out,
    paired_bootstrap,
    plot_comparison,
    rating_noise_floor,
    regression_metrics,
)

PLAN = common.plan()
PRIMARY = PLAN["targets"]["primary"]
TARGETS = [PRIMARY, *PLAN["targets"]["secondary"]]
MMD = float(PLAN["targets"]["smallest_meaningful_difference"])
SEED = int(PLAN["confirmatory"]["bootstrap"]["seed"])
N_RESAMPLES = int(PLAN["confirmatory"]["bootstrap"]["n_resamples"])


# ------------------------------------------------------------------ inputs


class Inputs:
    def __init__(self, results: Path, dry: bool):
        self.results = results
        self.dry = dry
        self.targets = common.targets(dry).copy()
        ratings = pd.read_csv(common.ROOT / "s0_audit" / "out" / "ratings_long.csv")
        if dry:
            # The same permutation as the dry harness run, for the ratings too.
            carrier = {source: file for file, source in common.dry_permutation().items()}
            ratings["file_id"] = ratings["file_id"].map(carrier)
        self.ratings = ratings
        self.demographics = (
            pd.read_csv(common.ROOT / "s0_audit" / "out" / "demographics_clean.csv")
            .dropna(subset=["file_id"])
            .set_index("file_id")
        )

    def job(self, experiment: str, name: str) -> Path:
        return self.results / experiment / name

    def has(self, experiment: str, name: str, table: str = "summary.csv") -> bool:
        return (self.job(experiment, name) / table).is_file()

    def table(self, experiment: str, name: str, table: str) -> pd.DataFrame:
        return pd.read_csv(self.job(experiment, name) / table)

    def file_ids(self, material: str) -> dict[str, str]:
        index = pd.read_csv(common.INDEX / f"{material}.csv")
        resolved = [Path(p).resolve().as_posix() for p in index["path"]]
        return dict(zip(resolved, index["file_id"], strict=True))

    def items(self, experiment: str, name: str, material: str = "a") -> pd.DataFrame:
        items = self.table(experiment, name, "items.csv")
        items["file_id"] = items["path"].map(self.file_ids(material))
        return items

    def prediction_matrix(self, experiment: str, name: str, feature_set: str) -> np.ndarray:
        predictions = self.table(experiment, name, "predictions.csv")
        part = predictions[predictions["feature_set"] == feature_set]
        return part.pivot(index="item", columns="repeat", values="y_pred").sort_index().to_numpy()

    def slot_means(self, target: str, file_ids: list[str]) -> np.ndarray:
        sub = self.ratings[self.ratings["target"] == target]
        means = sub.groupby(["file_id", "slot"])["value"].mean().unstack("slot")
        return means.reindex(file_ids).to_numpy(dtype=float)


def write(frame: pd.DataFrame, out: Path, name: str) -> None:
    frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")


# ------------------------------------------------------------------ verdicts


def verdicts(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Plan section 9: first/second better, substantial, equivalent or inconclusive."""
    table = contrasts.copy()
    side = (table["ci_high"] < 0) | (table["ci_low"] > 0)
    agree = np.sign(table["difference"]) == np.sign(table["fold_mean_difference"])
    established = (table["p_corrected_t_adjusted"] < PLAN["confirmatory"]["alpha"]) & side & agree
    table["verdict"] = np.where(
        established,
        np.where(table["difference"] < 0, "first better", "second better"),
        np.where(table.get("equivalent", False), "equivalent", "inconclusive"),
    )
    table["substantial"] = established & (table["difference"].abs() >= MMD)
    return table


# ------------------------------------------------------------------ tables


def t1_sample(inp: Inputs, out: Path) -> pd.DataFrame:
    analysis_set = pd.read_csv(common.ROOT / "s1_segments" / "analysis_set.csv").set_index("file_id")
    rec = common.recordings()
    frame = pd.DataFrame(index=analysis_set.index)
    frame["recording_group"] = frame.index.str.extract(r"^([A-Z]+)")[0].to_numpy()
    frame["primary"] = analysis_set["primary_set"].astype(bool)
    frame["severity"] = inp.targets.loc[frame.index, PRIMARY]
    frame["age"] = inp.demographics["age"].reindex(frame.index)
    frame["female"] = (inp.demographics["sex"].reindex(frame.index) == "F").astype(float)
    frame["rate_25k"] = (rec.loc[frame.index, "sample_rate"] == 25000).astype(float)
    primary = frame[frame["primary"]]
    rows = []
    for group, part in [*primary.groupby("recording_group"), ("all", primary)]:
        rows.append(
            {
                "recording_group": group,
                "n": len(part),
                "age_median": part["age"].median(),
                "female_share": part["female"].mean(),
                "rate_25k_share": part["rate_25k"].mean(),
                "severity_mean": part["severity"].mean(),
                "severity_median": part["severity"].median(),
                "severity_sd": part["severity"].std(ddof=1),
            }
        )
    table = pd.DataFrame(rows)
    write(table, out, "t1_sample")
    # E5j: the files without a valid /a/.
    missing = frame[~frame["primary"]]
    test = stats.mannwhitneyu(missing["severity"], primary["severity"], alternative="two-sided")
    e5j = pd.DataFrame(
        [
            {
                "n_missing": len(missing),
                "missing_median": missing["severity"].median(),
                "primary_median": primary["severity"].median(),
                "mann_whitney_u": test.statistic,
                "p": test.pvalue,
                "missing_groups": ", ".join(
                    f"{g} {n}" for g, n in missing["recording_group"].value_counts().items()
                ),
            }
        ]
    )
    write(e5j, out, "e5j_missing_a")
    return table


def ceiling_numbers(inp: Inputs, target: str, file_ids: list[str]) -> dict:
    means = inp.slot_means(target, file_ids)
    icc = icc_oneway(means)
    pairs = leave_one_rater_out(means)
    human = regression_metrics(pairs["reference"], pairs["rating"])
    return {
        "target": target,
        "n_files": len(file_ids),
        "icc1": icc.icc1,
        "icck": icc.icck,
        "icck_low": icc.icck_low,
        "icck_high": icc.icck_high,
        "noise_floor_rmse": rating_noise_floor(means),
        "r_max": float(np.sqrt(max(icc.icck, 0.0))),
        "target_sd": float(np.nanstd(np.nanmean(means, axis=1), ddof=1)),
        "one_rater_rmse": human["rmse"],
        "one_rater_r": human["pearson_r"],
        "one_rater_ccc": human["ccc"],
    }


def t2_reliability(inp: Inputs, out: Path) -> pd.DataFrame:
    primary_ids = inp.items("E1", "primary")["file_id"].tolist()
    rows = [ceiling_numbers(inp, target, primary_ids) for target in TARGETS]
    table = pd.DataFrame(rows)
    write(table, out, "t2_reliability")
    if not inp.dry:
        # The package must give S0's numbers on all 296 files.
        s0 = pd.read_csv(common.ROOT / "s0_audit" / "out" / "reliability.csv").set_index("target")
        all_ids = inp.targets.index.tolist()
        check = []
        for target in s0.index:
            mine = icc_oneway(inp.slot_means(target, all_ids)).icck
            check.append({"target": target, "package_icck": mine,
                          "s0_icck": s0.loc[target, "inter_icck_slotmeans"],
                          "abs_difference": abs(mine - s0.loc[target, "inter_icck_slotmeans"])})  # fmt: skip
        write(pd.DataFrame(check), out, "t2_check_against_s0")
    return table


def e6_model_vs_rater(inp: Inputs, out: Path) -> pd.DataFrame:
    items = inp.items("E1", "primary")
    means = inp.slot_means(PRIMARY, items["file_id"].tolist())
    pairs = leave_one_rater_out(means)
    groups = items["group"].to_numpy()[pairs["item"]]
    summary = inp.table("E1", "primary", "summary.csv")
    floor = rating_noise_floor(means)
    rows = []
    for name in summary["feature_set"]:
        matrix = inp.prediction_matrix("E1", "primary", name)[pairs["item"]]
        result = paired_bootstrap(pairs["reference"], matrix, pairs["rating"], groups,
                                  n_resamples=N_RESAMPLES, seed=SEED)  # fmt: skip
        model_rmse = float(summary.set_index("feature_set").loc[name, "rmse"])
        rows.append({"feature_set": name, "model_minus_rater": result.estimate,
                     "ci_low": result.low, "ci_high": result.high,
                     "model_rmse": model_rmse, "model_minus_floor": model_rmse - floor})  # fmt: skip
    table = pd.DataFrame(rows)
    write(table, out, "e6_model_vs_one_rater")
    return table


def t3_t4_primary(inp: Inputs, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = inp.table("E1", "primary", "summary.csv")
    write(summary, out, "t3_primary")
    contrasts = verdicts(inp.table("E1", "primary", "contrasts.csv"))
    write(contrasts, out, "t4_contrasts")
    return summary, contrasts


def t5_secondary(inp: Inputs, out: Path) -> None:
    rows, contrast_rows = [], []
    for target in PLAN["targets"]["secondary"]:
        if not inp.has("E3", target):
            continue
        summary = inp.table("E3", target, "summary.csv")
        sd = float(np.std(inp.items("E3", target)["y_true"], ddof=1))
        summary.insert(0, "target", target)
        summary["rmse_over_sd"] = summary["rmse"] / sd
        rows.append(summary)
        contrasts = verdicts(inp.table("E3", target, "contrasts.csv"))
        contrasts.insert(0, "target", target)
        contrast_rows.append(contrasts)
    if rows:
        write(pd.concat(rows), out, "t5_secondary")
        write(pd.concat(contrast_rows), out, "t5_secondary_contrasts")


def t6_materials(inp: Inputs, out: Path) -> None:
    rows, contrast_rows = [], []
    for material in PLAN["materials"]["exploratory"]:
        if not inp.has("E4", material):
            continue
        summary = inp.table("E4", material, "summary.csv")
        summary.insert(0, "material", material)
        rows.append(summary)
        contrasts = verdicts(inp.table("E4", material, "contrasts.csv"))
        contrasts.insert(0, "material", material)
        contrast_rows.append(contrasts)
    if rows:
        write(pd.concat(rows), out, "t6_materials")
        write(pd.concat(contrast_rows), out, "t6_materials_contrasts")


SENSITIVITY_RENAME = {"M0pt": "M0", "A3_native": "A3"}


def t7_sensitivity(inp: Inputs, out: Path, primary: pd.DataFrame) -> pd.DataFrame:
    reference = primary.set_index(["first", "second"])
    rows = []
    for folder in sorted((inp.results / "E5").glob("*")):
        if not (folder / "contrasts.csv").is_file():
            continue
        contrasts = verdicts(pd.read_csv(folder / "contrasts.csv"))
        for row in contrasts.itertuples():
            key = (SENSITIVITY_RENAME.get(row.first, row.first), SENSITIVITY_RENAME.get(row.second, row.second))
            base = reference.loc[key] if key in reference.index else None
            rows.append(
                {
                    "job": folder.name,
                    "first": row.first,
                    "second": row.second,
                    "difference": row.difference,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "p_adjusted": row.p_corrected_t_adjusted,
                    "verdict": row.verdict,
                    "primary_difference": None if base is None else base["difference"],
                    "same_direction": None if base is None else bool(np.sign(base["difference"]) == np.sign(row.difference)),
                    "same_verdict": None if base is None else bool(base["verdict"] == row.verdict),
                }
            )  # fmt: skip
    table = pd.DataFrame(rows)
    write(table, out, "t7_sensitivity")
    return table


def e5b_leave_group_out(inp: Inputs, out: Path) -> None:
    if inp.has("E5", "E5b_leave_group_out"):
        write(inp.table("E5", "E5b_leave_group_out", "summary.csv"), out, "e5b_leave_group_out")


def e5c_by_sex(inp: Inputs, out: Path) -> None:
    items = inp.items("E1", "primary")
    sex = inp.demographics["sex"].reindex(items["file_id"]).to_numpy()
    summary = inp.table("E1", "primary", "summary.csv")
    rows = []
    for name in summary["feature_set"]:
        matrix = inp.prediction_matrix("E1", "primary", name)
        for value in ("F", "M"):
            mask = sex == value
            result = bootstrap_metric(items["y_true"].to_numpy()[mask], matrix[mask],
                                      items["group"].to_numpy()[mask],
                                      n_resamples=N_RESAMPLES, seed=SEED)  # fmt: skip
            rows.append({"feature_set": name, "sex": value, "n": int(mask.sum()),
                         "rmse": result.estimate, "ci_low": result.low, "ci_high": result.high})  # fmt: skip
    write(pd.DataFrame(rows), out, "e5c_rmse_by_sex")


def e5h_residuals(inp: Inputs, out: Path) -> None:
    items = inp.items("E1", "primary")
    recording_group = items["file_id"].str.extract(r"^([A-Z]+)")[0].to_numpy()
    summary = inp.table("E1", "primary", "summary.csv")
    rows = []
    for name in summary["feature_set"]:
        residual = inp.prediction_matrix("E1", "primary", name).mean(axis=1) - items["y_true"].to_numpy()
        samples = [residual[recording_group == g] for g in np.unique(recording_group)]
        test = stats.kruskal(*samples)
        total = np.sum((residual - residual.mean()) ** 2)
        between = sum(s.size * (s.mean() - residual.mean()) ** 2 for s in samples)
        rows.append({"feature_set": name, "kruskal_h": test.statistic, "p": test.pvalue,
                     "eta_squared": between / total,
                     **{f"mean_residual_{g}": s.mean() for g, s in zip(np.unique(recording_group), samples)}})  # fmt: skip
    write(pd.DataFrame(rows), out, "e5h_residuals_by_group")


def e2_layers(inp: Inputs, out: Path) -> pd.DataFrame:
    rows = []
    for model in ("A4", "A5", "A6"):
        for target in TARGETS:
            if not inp.has("E2", f"{model}_{target}"):
                continue
            summary = inp.table("E2", f"{model}_{target}", "summary.csv")
            sd = float(np.std(inp.items("E2", f"{model}_{target}")["y_true"], ddof=1))
            for row in summary.itertuples():
                if "@" not in row.feature_set:
                    continue
                rows.append({"model": model, "target": target, "block": row.feature_set.split("@")[1],
                             "rmse": row.rmse, "rmse_low": row.rmse_low, "rmse_high": row.rmse_high,
                             "rmse_over_sd": row.rmse / sd, "r2": row.r2})  # fmt: skip
    table = pd.DataFrame(rows)
    if not table.empty:
        write(table, out, "e2_layer_curves")
    return table


def e7_mfcc_settings(inp: Inputs, out: Path) -> None:
    if inp.has("E7", "mfcc_settings"):
        write(inp.table("E7", "mfcc_settings", "summary.csv"), out, "e7_mfcc_settings")
        write(inp.table("E7", "mfcc_settings", "comparison.csv"), out, "e7_mfcc_comparison")


# ------------------------------------------------------------------ figures


def save(figure: Figure, out: Path, name: str) -> None:
    figure.savefig(out / f"{name}.png", dpi=200)
    figure.savefig(out / f"{name}.pdf")


def f1_confound(inp: Inputs, out: Path) -> None:
    items = inp.items("E1", "primary")
    group = items["file_id"].str.extract(r"^([A-Z]+)")[0]
    figure = Figure(figsize=(6.4, 3.6), layout="constrained")
    axes = figure.subplots()
    names = sorted(group.unique())
    data = [items["y_true"][group == g] for g in names]
    axes.boxplot(data, showfliers=False)
    rng = np.random.default_rng(0)
    for k, values in enumerate(data, start=1):
        axes.scatter(k + rng.uniform(-0.15, 0.15, values.size), values, s=6, alpha=0.5, color="#2a78d6")
    rates = common.recordings()["sample_rate"]
    labels = [f"{g}\n{int(rates[items['file_id'][group == g]].mode()[0]) / 1000:g} kHz" for g in names]
    axes.set_xticks(range(1, len(names) + 1), labels)
    axes.set(ylabel="CAPE-V severity (mean rating)", title="Severity by recording group")
    save(figure, out, "f1_recording_group_confound")


def f2_primary(inp: Inputs, out: Path, ceiling: pd.DataFrame) -> None:
    summary = inp.table("E1", "primary", "summary.csv")
    row = ceiling.set_index("target").loc[PRIMARY]
    figure = plot_comparison(summary, "rmse", references={"rating noise floor": row["noise_floor_rmse"],
                                                          "one rater": row["one_rater_rmse"]}, dpi=200)  # fmt: skip
    save(figure, out, "f2_primary_rmse")


def f3_layers(layers: pd.DataFrame, out: Path) -> None:
    if layers.empty:
        return
    figure = Figure(figsize=(7.0, 3.6), layout="constrained")
    axes = figure.subplots()
    for model, part in layers[layers["target"] == PRIMARY].groupby("model"):
        part = part[part["block"] != "AVG"]
        x = part["block"].str[1:].astype(int)
        axes.errorbar(x, part["rmse"], yerr=[part["rmse"] - part["rmse_low"], part["rmse_high"] - part["rmse"]],
                      marker="o", capsize=2, label=model)  # fmt: skip
    axes.set(xlabel="hidden state", ylabel="RMSE (CAPE-V severity)", title="Layer by layer")
    axes.legend(frameon=False)
    save(figure, out, "f3_layer_curves")
    models = sorted(layers["model"].unique())
    figure = Figure(figsize=(4.0 * len(models), 4.2), layout="constrained")
    for axes, model in zip(np.atleast_1d(figure.subplots(1, len(models))), models, strict=True):
        grid = layers[layers["model"] == model].pivot(index="target", columns="block", values="rmse_over_sd")
        grid = grid.reindex(index=[t for t in TARGETS if t in grid.index])
        image = axes.imshow(grid.to_numpy(), aspect="auto", cmap="viridis_r")
        axes.set_xticks(range(grid.shape[1]), grid.columns, rotation=90, fontsize=7)
        axes.set_yticks(range(grid.shape[0]), grid.index, fontsize=7)
        axes.set_title(model)
        figure.colorbar(image, ax=axes, label="RMSE / SD")
    save(figure, out, "f3_layer_by_target")


def f4_agreement(inp: Inputs, out: Path, summary: pd.DataFrame) -> None:
    best = summary.sort_values("rank")["feature_set"].iloc[0]
    items = inp.items("E1", "primary")
    predicted = inp.prediction_matrix("E1", "primary", best).mean(axis=1)
    truth = items["y_true"].to_numpy()
    figure = Figure(figsize=(9.0, 4.0), layout="constrained")
    left, right = figure.subplots(1, 2)
    left.scatter(truth, predicted, s=8, alpha=0.6)
    left.plot([0, 100], [0, 100], color="grey", linewidth=1)
    left.set(xlabel="mean rating", ylabel=f"predicted ({best}, mean of repeats)", title="Observed and predicted")
    mean, diff = (truth + predicted) / 2, predicted - truth
    bias, spread = diff.mean(), 1.96 * diff.std(ddof=1)
    right.scatter(mean, diff, s=8, alpha=0.6)
    for value in (bias, bias - spread, bias + spread):
        right.axhline(value, color="grey", linestyle="--", linewidth=1)
    right.set(xlabel="mean of rating and prediction", ylabel="prediction - rating", title="Bland-Altman")
    save(figure, out, "f4_agreement")


def f5_forest(primary: pd.DataFrame, sensitivity: pd.DataFrame, out: Path) -> None:
    labels = [f"{a} - {b}" for a, b in zip(primary["first"], primary["second"], strict=True)]
    figure = Figure(figsize=(7.0, 0.45 * len(labels) + 1.2), layout="constrained")
    axes = figure.subplots()
    rows = np.arange(len(labels))[::-1]
    axes.errorbar(primary["difference"], rows,
                  xerr=[primary["difference"] - primary["ci_low"], primary["ci_high"] - primary["difference"]],
                  fmt="o", color="black", capsize=3, label="primary (E1)")  # fmt: skip
    if not sensitivity.empty:
        position = dict(zip(zip(primary["first"], primary["second"], strict=True), rows, strict=True))
        for row in sensitivity.itertuples():
            key = (SENSITIVITY_RENAME.get(row.first, row.first), SENSITIVITY_RENAME.get(row.second, row.second))
            if key in position:
                axes.plot(row.difference, position[key] + 0.2, "|", color="#eb6834", markersize=8)
        axes.plot([], [], "|", color="#eb6834", label="sensitivity analyses (E5)")
    for value in (-MMD, 0.0, MMD):
        axes.axvline(value, color="grey", linestyle=":" if value else "-", linewidth=1)
    axes.set_yticks(rows, labels)
    axes.set(xlabel="RMSE difference (first - second)", title="Confirmatory contrasts")
    axes.legend(frameon=False, fontsize="small")
    save(figure, out, "f5_contrasts_forest")


def f6_segmentation(out: Path) -> None:
    source = common.ROOT / "s1_segments" / "acceptance_figures"
    figures = sorted(source.glob("*.png"))
    if figures:
        shutil.copy(figures[0], out / f"f6_segmentation_{figures[0].stem}.png")


# ------------------------------------------------------------------ summary


def summary_md(out: Path, primary: pd.DataFrame, contrasts: pd.DataFrame, ceiling: pd.DataFrame,
               e6: pd.DataFrame, dry: bool) -> None:  # fmt: skip
    row = ceiling.set_index("target").loc[PRIMARY]
    lines = ["# Summary of the numbers cited in the paper", ""]
    if dry:
        lines += ["**Dry run on permuted targets: none of these numbers is a result.**", ""]
    lines += [
        f"- Files in the primary set: {int(row['n_files'])}",
        f"- ICC(1,k) of the mean severity rating: {row['icck']:.3f} [{row['icck_low']:.3f}, {row['icck_high']:.3f}]",
        f"- Rating noise floor (RMSE): {row['noise_floor_rmse']:.2f}; one rater: {row['one_rater_rmse']:.2f}; r_max {row['r_max']:.3f}",
        "",
        "## RMSE per feature set (E1)",
        "",
        "| set | RMSE | 95% CI | R² | r | CCC |",
        "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in primary.sort_values("rank").itertuples():
        lines.append(f"| {r.feature_set} | {r.rmse:.2f} | {r.rmse_low:.2f}–{r.rmse_high:.2f} | {r.r2:.3f} | {r.pearson_r:.3f} | {r.ccc:.3f} |")
    lines += ["", "## Confirmatory contrasts", "", "| contrast | Δ | 95% CI | Holm p | verdict |", "|---|---|---|---|---|"]
    for r in contrasts.itertuples():
        extra = " (substantial)" if r.substantial else ""
        lines.append(f"| {r.first} − {r.second} | {r.difference:+.2f} | {r.ci_low:+.2f}–{r.ci_high:+.2f} | {r.p_corrected_t_adjusted:.3g} | {r.verdict}{extra} |")
    lines += ["", "## Model minus one rater (E6)", "", "| set | Δ RMSE | 95% CI |", "|---|---|---|"]
    for r in e6.itertuples():
        lines.append(f"| {r.feature_set} | {r.model_minus_rater:+.2f} | {r.ci_low:+.2f}–{r.ci_high:+.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE / "analysis")
    parser.add_argument("--dry", action="store_true", help="Permute every target first.")
    args = parser.parse_args(argv)
    results = args.results if args.results.is_absolute() else HERE / args.results
    out = args.out if args.out.is_absolute() else HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    inp = Inputs(results, args.dry)
    t1_sample(inp, out)
    ceiling = t2_reliability(inp, out)
    primary, contrasts = t3_t4_primary(inp, out)
    e6 = e6_model_vs_rater(inp, out)
    t5_secondary(inp, out)
    t6_materials(inp, out)
    sensitivity = t7_sensitivity(inp, out, contrasts)
    e5b_leave_group_out(inp, out)
    e5c_by_sex(inp, out)
    e5h_residuals(inp, out)
    layers = e2_layers(inp, out)
    e7_mfcc_settings(inp, out)
    f1_confound(inp, out)
    f2_primary(inp, out, ceiling)
    f3_layers(layers, out)
    f4_agreement(inp, out, primary)
    f5_forest(contrasts, sensitivity, out)
    f6_segmentation(out)
    summary_md(out, primary, contrasts, ceiling, e6, args.dry)
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
