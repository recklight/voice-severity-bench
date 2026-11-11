"""Gradient descent against the closed form: learning rate and predictor scaling.

    python examples/compare_gradient_descent_with_ols.py \\
        --out examples/output/compare_gradient_descent_with_ols

1. Simulate a population series indexed by calendar year (x around 2000).
2. Run batch gradient descent with standardized predictors at three learning
   rates, and once on the raw years at 2e-7. With x near 2000 any rate above
   about 5e-7 diverges, and 2e-7 does not converge within 20 000 iterations.
3. Compare each run with the least-squares solution and record its cost history.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from voice_severity_bench import (
    Config,
    configure_logging,
    cost_history_table,
    fit_linear_gd,
    fit_ols,
    plot_cost_history,
    save_figure,
)

LOGGER = logging.getLogger("compare_gradient_descent_with_ols")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_gradient_descent_with_ols"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Same switch as the other examples, so all of them run with one command line; "
            "nothing here depends on configs/quick_demo.yaml."
        ),
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    rng = np.random.default_rng(config.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    max_iter = 20_000  # the raw-year run uses all of them; still well under a second

    years = np.arange(1998.0, 2018.0, 2.0)
    population = 21.8 + 0.08 * (years - 1998.0) + rng.normal(0, 0.05, years.size)
    reference = fit_ols(years, population, ["year"])
    LOGGER.info("least squares: intercept %.4f, slope %.6f", *reference.coef)

    runs = {
        "standardized, rate 0.01": {"learning_rate": 0.01},
        "standardized, rate 0.1": {"learning_rate": 0.1},
        "standardized, rate 1.0": {"learning_rate": 1.0},
        "raw years, rate 2e-7": {"learning_rate": 2e-7, "standardize": False},
    }
    rows, histories = [], {}
    for name, options in runs.items():
        result = fit_linear_gd(years, population, max_iter=max_iter, **options)
        error = float(np.max(np.abs([result.intercept - reference.coef[0],
                                     result.coef[0] - reference.coef[1]])))  # fmt: skip
        histories[name] = result.cost_history
        rows.append(
            {
                "run": name,
                "iterations": result.n_iter,
                "converged": result.converged,
                "diverged": result.diverged,
                "intercept": result.intercept,
                "slope": float(result.coef[0]),
                "max_abs_error": error,
            }
        )
        LOGGER.info("%-24s %6d iterations  converged=%s  max error %.2e", name, result.n_iter,
                    result.converged, error)  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "gradient_descent_runs.csv", index=False, lineterminator="\n")
    save_figure(plot_cost_history(histories), out / "cost_history")
    cost_history_table(histories).to_csv(out / "cost_history.csv", index=False, lineterminator="\n")
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
