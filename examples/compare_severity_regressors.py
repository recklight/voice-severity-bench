"""Predicting a continuous severity rating: five regressors, agreement and correlation.

    python examples/compare_severity_regressors.py --out examples/output/compare_severity_regressors

1. Write a synthetic corpus and compute pooled MFCC features once.
2. Cross-validate linear regression, ridge, support vector regression,
   random forest and a multilayer perceptron on the same grouped folds.
3. Report RMSE, MAE, R2 and Lin's concordance (CCC) per model, and draw the
   Bland-Altman agreement plot of the best one.
4. Correlate every feature with the rating (Pearson with Fisher intervals,
   Holm-adjusted).

How the five regressors rank on synthetic vowels checks the code; it is no
reason to prefer one of them on real recordings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from voice_severity_bench import (
    Config,
    bland_altman_table,
    configure_logging,
    cross_validate,
    feature_target_correlations,
    plot_bland_altman,
    prepare_features,
    save_figure,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_severity_regressors")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"
MODELS = ("linear", "ridge", "svr", "random_forest", "mlp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_severity_regressors"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use configs/quick_demo.yaml: 8 speakers x 3 recordings of 0.5 s, 4 folds.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    out: Path = args.out

    index = write_synthetic_dataset(out / "data", config.synthetic, rng=config.seed)
    features = prepare_features(index, config)
    rows, results = [], {}
    for name in MODELS:
        candidate = config.override({"model.name": name, "model.n_estimators": 100})
        result = cross_validate(features, candidate)
        results[name] = result
        summary = result.metrics["summary"]
        rows.append(
            {"model": name}
            | {f"{k}_mean": summary[k]["mean"] for k in ("rmse", "mae", "r2", "ccc")}
            | {f"{k}_std": summary[k]["std"] for k in ("rmse", "ccc")}
        )
    table = pd.DataFrame(rows).sort_values("rmse_mean", kind="mergesort")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "model_comparison.csv", index=False, lineterminator="\n")
    for row in table.itertuples():
        LOGGER.info("%-13s RMSE %.2f +/- %.2f  CCC %.3f", row.model, row.rmse_mean,
                    row.rmse_std, row.ccc_mean)  # fmt: skip

    best = str(table.iloc[0]["model"])
    predictions = results[best].predictions
    predictions.to_csv(out / f"predictions_{best}.csv", index=False, lineterminator="\n")
    save_figure(plot_bland_altman(predictions, config.data.target), out / f"bland_altman_{best}")
    bland_altman_table(predictions, config.data.target).to_csv(
        out / f"bland_altman_{best}.csv", index=False, lineterminator="\n"
    )
    agreement = results[best].metrics["pooled"]["bland_altman"]
    LOGGER.info("%s: bias %.2f, limits of agreement %.2f .. %.2f", best, agreement["bias"],
                agreement["lower_loa"], agreement["upper_loa"])  # fmt: skip

    names = list(features.feature_names)
    correlations = feature_target_correlations(features.X, features.y, names)
    correlations.to_csv(out / "correlations.csv", index=False, lineterminator="\n")
    top = correlations.iloc[0]
    LOGGER.info("strongest feature %s: r = %.3f [%.3f, %.3f]", top["feature"], top["pearson_r"],
                top["ci_low"], top["ci_high"])  # fmt: skip
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
