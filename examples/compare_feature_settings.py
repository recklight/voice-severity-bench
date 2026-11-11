"""Which MFCC setting predicts severity best? A repeated-cross-validation ablation.

    python examples/compare_feature_settings.py --out examples/output/compare_feature_settings

1. Write a synthetic corpus with severity ratings and speaker groups.
2. Expand the ``ablation`` grid: number of cepstra x dynamic order
   (13 / 26 / 39 dims) x estimator.
3. Score every setting on the same repeated, speaker-grouped folds.
4. Rank the settings and test each one against the baseline (the first value
   of every list) with the corrected resampled t test and the Wilcoxon
   signed-rank test, Holm-adjusted.

On synthetic vowels the ranking only shows that the grid, the shared folds and
the paired tests work; it says nothing about which MFCC setting suits real voices.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from voice_severity_bench import (
    Config,
    ablation_table,
    configure_logging,
    plot_ablation,
    run_ablation,
    save_figure,
    write_json,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_feature_settings")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_feature_settings"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use configs/quick_demo.yaml: 8 speakers x 3 recordings of 0.5 s, delta_order 0 "
            "and 2 with ridge only, 4 folds x 2 repeats."
        ),
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    out: Path = args.out

    index = write_synthetic_dataset(out / "data", config.synthetic, rng=config.seed)
    result = run_ablation(index, config, cache_dir=out / "frame_cache")
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("ablation_folds", result.folds),
        ("ablation_summary", result.summary),
        ("ablation_comparison", result.comparisons),
    ):
        table.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    write_json(out / "ablation.json", result.as_dict())
    save_figure(plot_ablation(result.summary, result.metric), out / "ablation_metric_by_dimension")
    ablation_table(result.summary, result.metric).to_csv(
        out / "ablation_metric_by_dimension.csv", index=False, lineterminator="\n"
    )

    metric = result.metric
    LOGGER.info("ranking by %s (baseline %s)", metric, result.baseline)
    for row in result.summary.head(5).itertuples():
        mean, std = getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_std")
        LOGGER.info("#%d %-40s %3d dims  %.3f +/- %.3f", row.rank, row.config_id,
                    row.n_features, mean, std)  # fmt: skip
    significant = result.comparisons[result.comparisons["p_corrected_t_adjusted"] < 0.05]
    LOGGER.info(
        "%d of %d settings differ from the baseline after Holm adjustment (corrected t test)",
        len(significant),
        len(result.comparisons),
    )
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
