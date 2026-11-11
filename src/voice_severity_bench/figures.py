"""Figures and the tables behind them.

Every ``plot_<thing>`` has a ``<thing>_table`` returning the plotted numbers,
and the commands write both (``<name>.png`` beside ``<name>.csv``), so every
figure can be checked or redrawn elsewhere. The figures are built with the
object-oriented :class:`matplotlib.figure.Figure` API, so neither ``pyplot``
nor a display is needed. Figure text is English to avoid missing CJK glyphs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve

from .association import bland_altman
from .evaluation import LOWER_IS_BETTER
from .ols import OLSResult
from .polynomial_order import OrderSweepResult
from .regression_diagnostics import influence_table

#: Colors of successive series.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227")
#: Markers of successive series, a second encoding next to color.
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v")


def new_figure(width: float = 6.4, height: float = 4.8, dpi: int = 150) -> Figure:
    """A constrained-layout figure that needs no display.

    Examples
    --------
    >>> new_figure(4, 3, 100).get_size_inches().tolist()
    [4.0, 3.0]
    """
    return Figure(figsize=(width, height), dpi=dpi, layout="constrained")


def save_figure(figure: Figure, path_stem: str | Path, figure_format: str = "png") -> Path:
    """Write ``<path_stem>.<figure_format>`` (dots inside the stem are kept).

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     save_figure(new_figure(2, 2, 50), Path(folder) / "fold.1", "svg").name
    'fold.1.svg'
    """
    stem = Path(path_stem)
    destination = stem.with_name(f"{stem.name}.{figure_format}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format=figure_format)
    return destination


def _style(index: int) -> tuple[str, str]:
    return SERIES_COLORS[index % len(SERIES_COLORS)], SERIES_MARKERS[index % len(SERIES_MARKERS)]


# --------------------------------------------------------------------------- regression results


def residuals_table(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    """Observed, predicted and residual value of every out-of-fold prediction."""
    return pd.DataFrame(
        {
            "path": predictions["path"],
            "fold": predictions["fold"],
            "observed": predictions[target].astype(float),
            "predicted": predictions["predicted"].astype(float),
            "residual": predictions["predicted"].astype(float) - predictions[target].astype(float),
        }
    )


def plot_residuals(predictions: pd.DataFrame, target: str, *, dpi: int = 150) -> Figure:
    """Predicted against observed ratings, and residuals against predictions."""
    table = residuals_table(predictions, target)
    figure = new_figure(9.0, 4.0, dpi)
    left, right = figure.subplots(1, 2)
    low = float(min(table["observed"].min(), table["predicted"].min()))
    high = float(max(table["observed"].max(), table["predicted"].max()))
    for index, (fold, part) in enumerate(table.groupby("fold")):
        color, marker = _style(index)
        left.scatter(
            part["observed"],
            part["predicted"],
            s=18,
            color=color,
            marker=marker,
            label=f"fold {fold}",
        )
        right.scatter(part["predicted"], part["residual"], s=18, color=color, marker=marker)
    left.plot([low, high], [low, high], color="0.4", linewidth=1.0, linestyle="--")
    left.set(
        xlabel=f"observed {target}", ylabel=f"predicted {target}", title="Out-of-fold predictions"
    )
    left.legend(fontsize="small", frameon=False)
    right.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    right.set(xlabel=f"predicted {target}", ylabel="predicted - observed", title="Residuals")
    return figure


def bland_altman_table(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    """Means and differences of predicted and observed ratings with bias and limits."""
    result = bland_altman(predictions["predicted"].astype(float), predictions[target].astype(float))
    return pd.DataFrame(
        {
            "path": predictions["path"],
            "mean": result.means,
            "difference": result.differences,
            "bias": result.bias,
            "lower_loa": result.lower_loa,
            "upper_loa": result.upper_loa,
        }
    )


def plot_bland_altman(predictions: pd.DataFrame, target: str, *, dpi: int = 150) -> Figure:
    """Bland-Altman plot of predicted against observed ratings."""
    table = bland_altman_table(predictions, target)
    figure = new_figure(6.0, 4.2, dpi)
    axes = figure.subplots()
    axes.scatter(table["mean"], table["difference"], s=18, color=SERIES_COLORS[0])
    for column, style, name in (
        ("bias", "-", "bias"),
        ("lower_loa", "--", "lower limit"),
        ("upper_loa", "--", "upper limit"),
    ):
        level = float(table[column].iloc[0])
        axes.axhline(level, color=SERIES_COLORS[1], linestyle=style, linewidth=1.0)
        axes.annotate(
            f"{name} {level:.2f}",
            (1.0, level),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize="small",
        )
    axes.set(
        xlabel=f"mean of predicted and observed {target}",
        ylabel="predicted - observed",
        title="Agreement (Bland-Altman)",
    )
    return figure


# --------------------------------------------------------------------------- classification results


def confusion_matrix_table(classes: Sequence[str], matrix: ArrayLike) -> pd.DataFrame:
    """Counts with true classes as rows and predicted classes as columns."""
    counts = np.asarray(matrix, dtype=np.int64)
    table = pd.DataFrame(counts, columns=[f"predicted_{name}" for name in classes])
    table.insert(0, "true", list(classes))
    return table


def plot_confusion_matrix(classes: Sequence[str], matrix: ArrayLike, *, dpi: int = 150) -> Figure:
    """Pooled out-of-fold confusion matrix with counts and row percentages."""
    counts = np.asarray(matrix, dtype=np.float64)
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    figure = new_figure(4.8, 4.2, dpi)
    axes = figure.subplots()
    image = axes.imshow(counts / totals, cmap="Blues", vmin=0.0, vmax=1.0)
    for (row, column), value in np.ndenumerate(counts):
        share = value / totals[row, 0]
        axes.text(
            column,
            row,
            f"{int(value)}\n{share:.0%}",
            ha="center",
            va="center",
            color="white" if share > 0.6 else "black",
            fontsize="small",
        )
    axes.set_xticks(range(len(classes)), list(classes))
    axes.set_yticks(range(len(classes)), list(classes))
    axes.set(xlabel="predicted class", ylabel="true class", title="Confusion matrix")
    figure.colorbar(image, ax=axes, label="share of true class")
    return figure


def roc_curve_table(predictions: pd.DataFrame, target: str, positive_class: str) -> pd.DataFrame:
    """False and true positive rates of the pooled out-of-fold scores."""
    truth = predictions[target].astype(str).to_numpy() == positive_class
    false_positive, true_positive, thresholds = roc_curve(
        truth, predictions[f"score_{positive_class}"].to_numpy()
    )
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive,
            "true_positive_rate": true_positive,
            "threshold": np.minimum(thresholds, 1.0),
        }
    )


def plot_roc_curve(
    predictions: pd.DataFrame, target: str, positive_class: str, *, dpi: int = 150
) -> Figure:
    """ROC curve of the positive class from pooled out-of-fold scores."""
    table = roc_curve_table(predictions, target, positive_class)
    figure = new_figure(4.6, 4.4, dpi)
    axes = figure.subplots()
    axes.plot(table["false_positive_rate"], table["true_positive_rate"], color=SERIES_COLORS[0])
    axes.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1.0)
    axes.set(
        xlabel="1 - specificity",
        ylabel="sensitivity",
        title=f"ROC curve ({positive_class})",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    return figure


# --------------------------------------------------------------------------- ablation


def ablation_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and standard deviation of the ranking metric against the feature dimension."""
    model = summary["model"] if "model" in summary.columns else "model"
    return (
        pd.DataFrame(
            {
                "config_id": summary["config_id"],
                "model": model,
                "n_features": summary["n_features"].astype(int),
                "mean": summary[f"{metric}_mean"],
                "std": summary[f"{metric}_std"],
                "rank": summary["rank"],
            }
        )
        .sort_values(["model", "n_features"], kind="mergesort")
        .reset_index(drop=True)
    )


def plot_ablation(summary: pd.DataFrame, metric: str, *, dpi: int = 150) -> Figure:
    """Ranking metric (mean +/- std over folds) against the pooled feature dimension."""
    table = ablation_table(summary, metric)
    figure = new_figure(6.8, 4.4, dpi)
    axes = figure.subplots()
    for index, (model, part) in enumerate(table.groupby("model", sort=False)):
        color, marker = _style(index)
        axes.errorbar(
            part["n_features"],
            part["mean"],
            yerr=part["std"].fillna(0.0),
            fmt=marker,
            color=color,
            capsize=3,
            linestyle="none",
            label=str(model),
        )
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    axes.set(
        xlabel="pooled feature dimension",
        ylabel=f"{metric} ({direction})",
        title="Feature settings compared by repeated cross-validation",
    )
    axes.legend(frameon=False, fontsize="small")
    return figure


def comparison_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pooled metric and its bootstrap interval for every feature file, in summary order.

    Examples
    --------
    >>> summary = pd.DataFrame({"feature_set": ["ssl", "mfcc"], "rmse": [9.5, 11.0],
    ...                         "rmse_low": [8.7, 10.1], "rmse_high": [10.4, 12.0], "rank": [1, 2]})
    >>> comparison_table(summary, "rmse").values.tolist()
    [['ssl', 9.5, 8.7, 10.4, 1], ['mfcc', 11.0, 10.1, 12.0, 2]]
    """
    return pd.DataFrame(
        {
            "feature_set": summary["feature_set"].astype(str),
            "estimate": summary[metric].astype(float),
            "low": summary[f"{metric}_low"].astype(float),
            "high": summary[f"{metric}_high"].astype(float),
            "rank": summary["rank"].astype(int),
        }
    ).reset_index(drop=True)


def plot_comparison(
    summary: pd.DataFrame,
    metric: str,
    *,
    references: Mapping[str, float] | None = None,
    dpi: int = 150,
) -> Figure:
    """Pooled metric with its bootstrap interval per feature file, the first on top.

    ``references`` draws labelled vertical lines, e.g. the rating noise floor
    and the error of a single rater, so the distances to them can be read off.
    """
    table = comparison_table(summary, metric)
    figure = new_figure(6.8, 1.2 + 0.45 * len(table), dpi)
    axes = figure.subplots()
    rows = np.arange(len(table))[::-1]
    color, marker = _style(0)
    axes.errorbar(
        table["estimate"],
        rows,
        xerr=[table["estimate"] - table["low"], table["high"] - table["estimate"]],
        fmt=marker,
        color=color,
        capsize=3,
        linestyle="none",
    )
    for index, (label, value) in enumerate((references or {}).items(), start=1):
        line_color, _ = _style(index)
        axes.axvline(value, color=line_color, linestyle="--", linewidth=1.0, label=str(label))
    axes.set_yticks(rows, table["feature_set"].tolist())
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    axes.set(xlabel=f"{metric} ({direction})", title="Feature files compared on the same folds")
    if references:
        axes.legend(frameon=False, fontsize="small")
    return figure


# --------------------------------------------------------------------------- statistical analysis


def correlations_table(table: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """The ``top`` features with the largest absolute Pearson correlation."""
    ranked = table.dropna(subset=["pearson_r"])
    return ranked.head(top)[["feature", "pearson_r", "ci_low", "ci_high", "pearson_p_adjusted"]]


def plot_correlations(table: pd.DataFrame, top: int = 20, *, dpi: int = 150) -> Figure:
    """Horizontal bars of the strongest feature-target correlations with their intervals."""
    data = correlations_table(table, top).iloc[::-1]
    figure = new_figure(6.4, max(2.5, 0.28 * len(data) + 1.2), dpi)
    axes = figure.subplots()
    positions = np.arange(len(data))
    colors = [SERIES_COLORS[0] if r >= 0 else SERIES_COLORS[1] for r in data["pearson_r"]]
    axes.barh(positions, data["pearson_r"], color=colors)
    axes.errorbar(
        data["pearson_r"],
        positions,
        xerr=[data["pearson_r"] - data["ci_low"], data["ci_high"] - data["pearson_r"]],
        fmt="none",
        ecolor="0.3",
        capsize=2,
    )
    axes.set_yticks(positions, data["feature"].tolist(), fontsize="small")
    axes.axvline(0.0, color="0.4", linewidth=0.8)
    axes.set(
        xlabel="Pearson r with the target (with confidence interval)",
        xlim=(-1, 1),
        title="Feature-target correlations",
    )
    return figure


def regression_diagnostics_table(result: OLSResult) -> pd.DataFrame:
    """Per-observation fitted values, residuals, leverage and Cook's distance."""
    return influence_table(result)


def plot_regression_diagnostics(result: OLSResult, *, dpi: int = 150) -> Figure:
    """Four residual panels: residuals vs fitted, normal Q-Q, leverage, Cook's distance."""
    table = regression_diagnostics_table(result)
    figure = new_figure(8.4, 6.6, dpi)
    (a, b), (c, d) = figure.subplots(2, 2)
    a.scatter(table["fitted"], table["residual"], s=16, color=SERIES_COLORS[0])
    a.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    a.set(xlabel="fitted value", ylabel="residual", title="Residuals vs fitted")
    b.scatter(table["normal_quantile"], table["studentized"], s=16, color=SERIES_COLORS[0])
    finite = table["normal_quantile"].dropna()
    if not finite.empty:
        span = [float(finite.min()), float(finite.max())]
        b.plot(span, span, color="0.4", linestyle="--", linewidth=1.0)
    b.set(xlabel="normal quantile", ylabel="studentized residual", title="Normal Q-Q")
    c.scatter(table["leverage"], table["studentized"], s=16, color=SERIES_COLORS[0])
    c.axvline(
        2.0 * result.n_params / result.n_obs,
        color=SERIES_COLORS[1],
        linestyle="--",
        linewidth=1.0,
        label="2p/n",
    )
    c.legend(frameon=False, fontsize="small")
    c.set(xlabel="leverage", ylabel="studentized residual", title="Leverage")
    d.vlines(np.arange(result.n_obs), 0.0, table["cooks_distance"], color=SERIES_COLORS[0])
    d.axhline(
        4.0 / result.n_obs, color=SERIES_COLORS[1], linestyle="--", linewidth=1.0, label="4/n"
    )
    d.legend(frameon=False, fontsize="small")
    d.set(xlabel="observation", ylabel="Cook's distance", title="Influence")
    return figure


def order_sweep_table(result: OrderSweepResult) -> pd.DataFrame:
    """The per-degree statistics of a polynomial order sweep."""
    return result.table.copy()


def plot_order_sweep(result: OrderSweepResult, *, dpi: int = 150) -> Figure:
    """In-sample fit and out-of-sample error against the polynomial degree."""
    table = order_sweep_table(result)
    figure = new_figure(9.0, 4.0, dpi)
    left, right = figure.subplots(1, 2)
    for index, column in enumerate(("r2", "adj_r2")):
        color, marker = _style(index)
        left.plot(table["degree"], table[column], marker=marker, color=color, label=column)
    left.set(xlabel="degree", ylabel="share of variance", title="In-sample fit", ylim=(None, 1.02))
    left.legend(frameon=False, fontsize="small")
    for index, column in enumerate(("loocv_rmse", "kfold_rmse")):
        color, marker = _style(index + 2)
        right.plot(table["degree"], table[column], marker=marker, color=color, label=column)
    if result.recommended_degree is not None:
        right.axvline(
            result.recommended_degree,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
            label=f"chosen by {result.criterion}",
        )
    right.set_yscale("log")
    right.set(xlabel="degree", ylabel="RMSE", title="Out-of-sample error")
    right.legend(frameon=False, fontsize="small")
    return figure


def response_surface_table(
    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    first_range: tuple[float, float],
    second_range: tuple[float, float],
    names: tuple[str, str],
    *,
    resolution: int = 25,
) -> pd.DataFrame:
    """A regular grid over two predictors with the fitted response at every node."""
    first = np.linspace(*first_range, resolution)
    second = np.linspace(*second_range, resolution)
    grid_first, grid_second = np.meshgrid(first, second)
    points = np.column_stack([grid_first.ravel(), grid_second.ravel()])
    return pd.DataFrame({names[0]: points[:, 0], names[1]: points[:, 1], "fitted": predict(points)})


def plot_response_surface(
    grid: pd.DataFrame,
    observed: pd.DataFrame,
    names: tuple[str, str],
    response: str,
    *,
    dpi: int = 150,
) -> Figure:
    """Fitted second-order surface with the observations."""
    resolution = round(float(np.sqrt(len(grid))))
    shape = (resolution, resolution)
    figure = new_figure(6.4, 5.2, dpi)
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(
        grid[names[0]].to_numpy().reshape(shape),
        grid[names[1]].to_numpy().reshape(shape),
        grid["fitted"].to_numpy().reshape(shape),
        cmap="viridis",
        alpha=0.6,
        linewidth=0,
    )
    axes.scatter(
        observed[names[0]],
        observed[names[1]],
        observed[response],
        color=SERIES_COLORS[1],
        s=16,
        depthshade=False,
    )
    axes.set(
        xlabel=names[0], ylabel=names[1], zlabel=response, title="Second-order response surface"
    )
    axes.view_init(elev=30, azim=140)
    return figure


def cost_history_table(histories: Mapping[str, ArrayLike]) -> pd.DataFrame:
    """Long table of gradient-descent cost per run and iteration."""
    frames = [
        pd.DataFrame(
            {
                "run": name,
                "iteration": np.arange(len(np.asarray(values))),
                "cost": np.asarray(values, dtype=np.float64),
            }
        )
        for name, values in histories.items()
    ]
    return pd.concat(frames, ignore_index=True)


def plot_cost_history(histories: Mapping[str, ArrayLike], *, dpi: int = 150) -> Figure:
    """Cost against iteration for several gradient-descent runs (log scales)."""
    table = cost_history_table(histories)
    figure = new_figure(6.4, 4.2, dpi)
    axes = figure.subplots()
    for index, (name, part) in enumerate(table.groupby("run", sort=False)):
        color, _ = _style(index)
        axes.plot(part["iteration"] + 1, part["cost"], color=color, label=str(name))
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set(xlabel="iteration", ylabel="cost J", title="Gradient descent convergence")
    axes.legend(frameon=False, fontsize="small")
    return figure
