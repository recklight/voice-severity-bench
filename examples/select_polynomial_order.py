"""How many polynomial terms? In-sample fit against cross-validated error.

    python examples/select_polynomial_order.py --out examples/output/select_polynomial_order

1. Ten noisy points of a cubic, degrees 1 to 8: R2 climbs towards 1, while the
   leave-one-out and k-fold errors, measured on held-out points, are lowest at
   degree 3 and much larger above it.
2. Sixty points of the same cubic: with that many points every criterion
   recovers degree 3.
3. A degree with no residual degrees of freedom is marked not estimable, and
   a k-fold training set too small for a degree gives NaN (degrees 7 and 8 of
   the ten-point sweep with the default five folds).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from voice_severity_bench import (
    Config,
    configure_logging,
    make_polynomial_data,
    plot_order_sweep,
    save_figure,
    sweep_polynomial_order,
)

LOGGER = logging.getLogger("select_polynomial_order")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/select_polynomial_order"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use configs/quick_demo.yaml: 4 folds instead of 5 for the k-fold RMSE.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    analysis = config.analysis
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    for label, n_points in (("ten_points", 10), ("sixty_points", 60)):
        data = make_polynomial_data(n_points, noise_sd=0.5, rng=config.seed)
        sweep = sweep_polynomial_order(
            data["x"],
            data["y"],
            range(1, analysis.max_degree + 1),
            basis=analysis.polynomial_basis,
            n_splits=config.evaluation.n_splits,
            criterion=analysis.order_criterion,
            rng=config.seed,
        )
        sweep.table.to_csv(out / f"order_sweep_{label}.csv", index=False, lineterminator="\n")
        save_figure(plot_order_sweep(sweep), out / f"order_sweep_{label}")
        LOGGER.info("%d points: %s recommends degree %s", n_points, sweep.criterion,
                    sweep.recommended_degree)  # fmt: skip
        for row in sweep.table.itertuples():
            LOGGER.info(
                "  degree %d  R2 %.4f  adj %.4f  F p %.3g  LOOCV %.3f  k-fold %.3f%s",
                row.degree, row.r2, row.adj_r2, row.f_pvalue, row.loocv_rmse, row.kfold_rmse,
                "" if row.estimable else "  (not estimable)",
            )  # fmt: skip
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
