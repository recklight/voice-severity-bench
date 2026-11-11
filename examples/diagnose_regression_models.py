"""Least-squares diagnostics: response surface, collinearity, stepwise selection and residuals.

    python examples/diagnose_regression_models.py --out examples/output/diagnose_regression_models

1. Fit the full second-order response surface to synthetic data, print the
   coefficient table with confidence intervals and draw the fitted surface.
2. On the published cement hardening data, show the near-exact collinearity
   (variance inflation factors) and how the stepwise start changes the
   selected model.
3. Check the residuals of the selected model: normality, constant variance,
   autocorrelation and influential observations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from voice_severity_bench import (
    Config,
    cement_hardening_heat,
    configure_logging,
    fit_ols,
    make_quadratic_surface_data,
    plot_regression_diagnostics,
    plot_response_surface,
    quadratic_surface,
    regression_diagnostics_table,
    residual_diagnostics,
    response_surface_table,
    save_figure,
    stepwise_select,
    vif,
    write_json,
)

LOGGER = logging.getLogger("diagnose_regression_models")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/diagnose_regression_models"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Same switch as the other examples; configs/quick_demo.yaml leaves the analysis "
            "settings used here at their defaults."
        ),
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    analysis = config.analysis
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1. Second-order response surface on centered predictors.
    data = make_quadratic_surface_data(30, noise_sd=1.0, rng=config.seed)
    predictors = data[["x1", "x2"]].to_numpy()
    design, terms = quadratic_surface(predictors, ["x1", "x2"])
    surface = fit_ols(design, data["y"], terms, ci_level=analysis.ci_level)
    surface.coefficient_table().to_csv(out / "surface_coefficients.csv", index=False,
                                       lineterminator="\n")  # fmt: skip
    for line in surface.summary().splitlines():
        LOGGER.info("%s", line)
    means = predictors.mean(axis=0)

    def predict(points):
        return surface.predict(quadratic_surface(points, ["x1", "x2"], means=means)[0])

    grid = response_surface_table(predict, (-3.0, 3.0), (-3.0, 3.0), ("x1", "x2"))
    grid.to_csv(out / "response_surface.csv", index=False, lineterminator="\n")
    save_figure(plot_response_surface(grid, data, ("x1", "x2"), "y"), out / "response_surface")

    # 2. Collinear cement data: VIF and the effect of the starting model.
    cement = cement_hardening_heat()
    columns = ["x1", "x2", "x3", "x4"]
    inflation = vif(cement[columns].to_numpy(), columns)
    inflation.to_csv(out / "cement_vif.csv", index=False, lineterminator="\n")
    LOGGER.info("cement VIF: %s", ", ".join(f"{t} {v:.1f}" for t, v in zip(
        inflation["term"], inflation["vif"], strict=True)))  # fmt: skip
    selections = {}
    for start in ("empty", "full"):
        result = stepwise_select(
            cement[columns],
            cement["heat"],
            columns,
            start=start,
            p_enter=analysis.p_enter,
            p_remove=analysis.p_remove,
        )
        result.steps.to_csv(out / f"cement_stepwise_{start}.csv", index=False, lineterminator="\n")
        selections[start] = list(result.selected)
        LOGGER.info("stepwise from the %s model selects %s (adjusted R2 %.4f)", start,
                    result.selected, result.final.adj_r2)  # fmt: skip

    # 3. Residual diagnostics of the model selected from the full start.
    final = fit_ols(cement[selections["full"]], cement["heat"], selections["full"])
    report = residual_diagnostics(final)
    write_json(out / "cement_diagnostics.json", {"model": final.as_dict(), "diagnostics": report,
                                                  "selected": selections})  # fmt: skip
    save_figure(plot_regression_diagnostics(final), out / "cement_residuals")
    regression_diagnostics_table(final).to_csv(
        out / "cement_residuals.csv", index=False, lineterminator="\n"
    )
    LOGGER.info("Shapiro-Wilk p %.3f, Breusch-Pagan p %.3f, Durbin-Watson %.2f, max Cook %.2f",
                report["shapiro_p"], report["breusch_pagan_p"], report["durbin_watson"],
                report["max_cooks_distance"])  # fmt: skip
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
