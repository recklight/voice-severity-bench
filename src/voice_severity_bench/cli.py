"""Command-line interface.

Commands
--------
synthesize
    Write a synthetic dataset of sustained vowels with severity ratings.
index
    Match a folder of recordings to a ratings table and write ``dataset.csv``.
prepare
    Compute one pooled cepstral feature vector per recording (``features.npz``).
evaluate
    Cross-validate the configured model; write metrics, predictions and figures.
train
    Fit the configured model on every item and save it.
predict
    Predict new recordings with a saved model and the settings stored in it.
extract
    Export frame-level features of a folder tree (npz, HTK or text).
ablate
    Compare feature settings and models by repeated cross-validation on shared folds.
compare
    Compare feature files from different front ends on shared folds, with bootstrap intervals.
correlate
    Correlate every feature with a continuous rating (Pearson with CI, Spearman).
regress
    Ordinary least squares with coefficient inference and residual diagnostics.
stepwise
    Stepwise predictor selection by partial F tests, with variance inflation factors.
order-sweep
    Compare polynomial degrees in sample and by cross-validation.

Commands that use settings take ``--config`` (a YAML file) and any number of
``--set section.key=value`` overrides; ``--seed`` and ``--dataset`` are
shortcuts for ``--set seed=...`` and ``--set data.dataset=...``. Every command
except ``index`` writes its results to ``--out`` (by default a new
``<command>_<UTC time>`` folder under ``output.root``; ``synthesize`` uses
``data/synthetic``) together with ``config.yaml`` and ``manifest.json``.
``index`` writes only ``FOLDER/dataset.csv``. ``predict`` uses only the
settings stored in the model file.

A missing file, an invalid configuration or a missing optional dependency
ends the command with a one-line message and exit code 2; the traceback is
logged at DEBUG level.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from . import __version__
from .ablation import run_ablation
from .association import feature_target_correlations
from .comparison import compare_feature_sets, pairwise_contrasts
from .config import Config
from .corpus import FEATURE_FORMATS, extract_corpus
from .dataset import (
    build_dataset_index,
    load_analysis_table,
    load_dataset_index,
    load_features,
    save_features,
)
from .design_matrix import quadratic_surface
from .errors import ConfigError
from .evaluation import cross_validate
from .figures import (
    ablation_table,
    bland_altman_table,
    comparison_table,
    confusion_matrix_table,
    correlations_table,
    order_sweep_table,
    plot_ablation,
    plot_bland_altman,
    plot_comparison,
    plot_confusion_matrix,
    plot_correlations,
    plot_order_sweep,
    plot_regression_diagnostics,
    plot_residuals,
    plot_roc_curve,
    regression_diagnostics_table,
    residuals_table,
    roc_curve_table,
    save_figure,
)
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import load_model, model_filename, save_model
from .ols import fit_ols
from .pipeline import predict_inputs, prepare_features, train_model
from .polynomial_order import sweep_polynomial_order
from .regression_diagnostics import residual_diagnostics, vif
from .splitting import has_groups
from .stepwise import stepwise_select
from .synthetic import write_synthetic_dataset

_LOGGER = logging.getLogger(__name__)

#: Designs accepted by ``regress --design``.
_DESIGNS: tuple[str, ...] = ("linear", "quadratic")

# typer validates an Enum option and lists its members in --help by itself, both
# in the releases built on click and in the later ones without it.
_FeatureFormat = Enum("_FeatureFormat", {name: name for name in FEATURE_FORMATS}, type=str)
_Design = Enum("_Design", {name: name for name in _DESIGNS}, type=str)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Local variables of a failing frame can hold whole datasets; never dump them.
    pretty_exceptions_show_locals=False,
    help="Cepstral voice features against severity ratings: ablation, inference and agreement.",
)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML configuration file.")
]
SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override one setting: section.key=value (repeatable)."),
]
SeedOption = Annotated[
    int | None, typer.Option("--seed", help="Master seed; overrides the configuration.")
]
DatasetOption = Annotated[
    Path | None,
    typer.Option("--dataset", "-d", help="Dataset index CSV; overrides data.dataset."),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out", "-o", help="Output folder (default: a new run folder under output.root)."
    ),
]
ModelOption = Annotated[Path, typer.Option("--model", "-m", help="Model file written by 'train'.")]
FiguresOption = Annotated[
    bool, typer.Option("--figures/--no-figures", help="Also draw the figures (with their CSV).")
]
TargetOption = Annotated[
    str | None,
    typer.Option("--target", "-t", help="Response column of a CSV table (ignored for .npz)."),
]
PredictorsOption = Annotated[
    str | None,
    typer.Option(
        "--predictors", "-p", help="Comma-separated predictor columns (default: all numeric)."
    ),
]
TableArgument = Annotated[
    Path, typer.Argument(help="features.npz written by 'prepare', or a CSV table.")
]


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"voice-severity-bench {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    try:
        configure_logging(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from None


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Turn the library's expected failures into a one-line message and exit code 2."""
    try:
        yield
    except (ValueError, FileNotFoundError, ImportError) as exc:
        _LOGGER.debug("command failed", exc_info=True)
        raise _fail(str(exc)) from None


def _coerce(raw: str) -> Any:
    """Parse a ``--set`` value: true/false, none/null, then JSON, else the string itself."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _load_config(
    config_file: Path | None,
    overrides: list[str] | None,
    shortcuts: Mapping[str, Any] | None = None,
) -> Config:
    """Apply ``--config``, then every ``--set``, then the shortcuts (``None`` = not given)."""
    config = Config.from_yaml(config_file) if config_file is not None else Config()
    changes: dict[str, Any] = {}
    for item in overrides or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        changes[key.strip()] = _coerce(raw.strip())
    for key, value in (shortcuts or {}).items():
        if value is not None:
            changes[key] = value.as_posix() if isinstance(value, Path) else value
    return config.override(changes) if changes else config


def _report(label: str, value: object) -> None:
    """Print one line of the summary a command shows when it finishes."""
    typer.echo(f"{label:<11}: {value}")


def _run_folder(config: Config, command: str, out: Path | None) -> Path:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        return out
    return run_directory(config.output.root, command)


def _finish(
    folder: Path,
    command: str,
    config: Config,
    inputs: Sequence[Path],
    details: Mapping[str, Any],
) -> None:
    """Write ``config.yaml`` and ``manifest.json`` and print the closing line."""
    config.to_yaml(folder / "config.yaml")
    manifest = build_manifest(
        command=command, config=config.to_dict(), inputs=inputs, seed=config.seed, extra=details
    )
    write_manifest(folder / "manifest.json", manifest)
    typer.echo(f"written to {folder.as_posix()}")


def _write_csv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def _figure(folder: Path, name: str, figure: Any, table: pd.DataFrame, config: Config) -> None:
    """Write a figure and the table behind it under the same stem."""
    save_figure(figure, folder / name, config.output.figure_format)
    _write_csv(table, folder / f"{name}.csv")


def _mean_std(summary: Mapping[str, Mapping[str, float]], name: str) -> str:
    entry = summary[name]
    return f"{entry['mean']:.3f} +/- {entry['std']:.3f}"


def _predictor_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ConfigError("--predictors needs at least one column name")
    return names


@app.command("synthesize")
def synthesize_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Folder for the synthetic dataset.")
    ] = Path("data/synthetic"),
) -> None:
    """Write a synthetic dataset of sustained vowels with severity ratings."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        index = write_synthetic_dataset(out, config.synthetic, rng=config.seed)
        table = pd.read_csv(index)
        _report("items", len(table))
        _report("speakers", table["group"].nunique())
        _report("severity", f"{table['severity'].min():.1f} .. {table['severity'].max():.1f}")
        _report("dysphonic", int((table["label"] == "dysphonic").sum()))
        _finish(out, "synthesize", config, [], {"n_items": len(table)})


@app.command("index")
def index_command(
    folder: Annotated[Path, typer.Argument(help="Folder searched recursively for .wav files.")],
    targets: Annotated[
        Path, typer.Option("--targets", help="CSV with an identifier column and the ratings.")
    ],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    id_column: Annotated[
        str, typer.Option("--id-column", help="Identifier column matched to file names.")
    ] = "id",
) -> None:
    """Match a folder of recordings to a ratings table and write FOLDER/dataset.csv."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        destination = build_dataset_index(folder, targets, id_column=id_column)
        index = load_dataset_index(destination, target=config.data.target, task=config.data.task)
        _report("items", len(index))
        _report("target", config.data.target)
        _report("speakers", index["group"].nunique() if has_groups(index["group"]) else "(none)")
        typer.echo(f"written to {destination.as_posix()}")


@app.command("prepare")
def prepare_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
) -> None:
    """Compute one pooled cepstral feature vector per recording (features.npz)."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"data.dataset": dataset})
        features = prepare_features(config.data.dataset, config)
        folder = _run_folder(config, "prepare", out)
        save_features(folder / "features.npz", features)
        items = pd.DataFrame(
            {
                "path": features.paths,
                config.data.target: features.y,
                "group": features.groups,
            }
        )
        _write_csv(items, folder / "items.csv")
        _report("items", features.X.shape[0])
        _report("features", features.X.shape[1])
        _report("frame dims", config.frontend.n_dims)
        _finish(
            folder,
            "prepare",
            config,
            [Path(config.data.dataset)],
            {"n_items": int(features.X.shape[0]), "n_features": int(features.X.shape[1])},
        )


@app.command("evaluate")
def evaluate_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
    figures: FiguresOption = True,
) -> None:
    """Cross-validate the configured model; write metrics, predictions and figures."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        result = cross_validate(data, config)
        folder = _run_folder(config, "evaluate", out)
        write_json(folder / "metrics.json", result.metrics)
        _write_csv(result.folds, folder / "folds.csv")
        _write_csv(result.predictions, folder / "predictions.csv")
        target = config.data.target
        summary = result.metrics["summary"]
        dpi = config.output.dpi
        if config.data.task == "regression":
            if figures:
                _figure(
                    folder,
                    "residuals",
                    plot_residuals(result.predictions, target, dpi=dpi),
                    residuals_table(result.predictions, target),
                    config,
                )
                _figure(
                    folder,
                    "bland_altman",
                    plot_bland_altman(result.predictions, target, dpi=dpi),
                    bland_altman_table(result.predictions, target),
                    config,
                )
            _report("items", result.metrics["n_items"])
            _report("folds", result.metrics["n_splits"])
            _report("MAE", _mean_std(summary, "mae"))
            _report("RMSE", _mean_std(summary, "rmse"))
            _report("R2", _mean_std(summary, "r2"))
            _report("CCC", _mean_std(summary, "ccc"))
        else:
            pooled = result.metrics["pooled"]
            classes = pooled["classes"]
            if figures:
                _figure(
                    folder,
                    "confusion_matrix",
                    plot_confusion_matrix(classes, pooled["confusion_matrix"], dpi=dpi),
                    confusion_matrix_table(classes, pooled["confusion_matrix"]),
                    config,
                )
                if len(classes) == 2:
                    positive = config.evaluation.positive_class
                    _figure(
                        folder,
                        "roc_curve",
                        plot_roc_curve(result.predictions, target, positive, dpi=dpi),
                        roc_curve_table(result.predictions, target, positive),
                        config,
                    )
            _report("items", result.metrics["n_items"])
            _report("folds", result.metrics["n_splits"])
            _report("UAR", _mean_std(summary, "uar"))
            _report("accuracy", _mean_std(summary, "accuracy"))
            if "roc_auc" in summary:
                _report("ROC-AUC", _mean_std(summary, "roc_auc"))
        _finish(folder, "evaluate", config, [features], {"n_items": result.metrics["n_items"]})


@app.command("train")
def train_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Fit the configured model on every item and save it."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        estimator = train_model(data, config)
        folder = _run_folder(config, "train", out)
        classes = None
        if config.data.task == "classification":
            classes = sorted({str(label) for label in data.y})
        path = save_model(folder / model_filename(config), estimator, config, classes)
        _report("items", data.X.shape[0])
        _report("model", config.model.name)
        _report("file", path.name)
        _finish(folder, "train", config, [features], {"n_items": int(data.X.shape[0])})


@app.command("predict")
def predict_command(
    inputs: Annotated[list[Path], typer.Argument(help="WAV files to predict.")],
    model: ModelOption,
    out: OutOption = None,
) -> None:
    """Predict new recordings with a saved model and the settings stored in it."""
    with _reported_errors():
        bundle = load_model(model)
        config = Config.from_dict(bundle["config"])
        table = predict_inputs(inputs, bundle)
        folder = _run_folder(config, "predict", out)
        _write_csv(table, folder / "predictions.csv")
        _report("items", len(table))
        _report("model", config.model.name)
        _finish(folder, "predict", config, [model, *inputs], {"n_items": len(table)})


@app.command("extract")
def extract_command(
    folder: Annotated[Path, typer.Argument(help="Folder searched recursively for .wav files.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    out: OutOption = None,
    fmt: Annotated[
        _FeatureFormat, typer.Option("--format", help="Output file format.")
    ] = _FeatureFormat["npz"],
) -> None:
    """Export frame-level features of a folder tree (npz, HTK or text)."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        target = _run_folder(config, "extract", out)
        table = extract_corpus(folder, target / "features", config, fmt=fmt.value)
        _write_csv(table, target / "extraction.csv")
        counts = table["status"].value_counts()
        _report("files", len(table))
        _report("written", int(counts.get("ok", 0)))
        _report("too short", int(counts.get("too_short", 0)))
        _report("failed", int(counts.get("error", 0)))
        _report("frame dims", config.frontend.n_dims)
        _finish(target, "extract", config, [], {"n_files": len(table), "format": fmt.value})


@app.command("ablate")
def ablate_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Keep frame-level features here between runs."),
    ] = None,
    figures: FiguresOption = True,
) -> None:
    """Compare feature settings and models by repeated cross-validation on shared folds."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed, "data.dataset": dataset})
        result = run_ablation(config.data.dataset, config, cache_dir=cache_dir)
        folder = _run_folder(config, "ablate", out)
        _write_csv(result.folds, folder / "ablation_folds.csv")
        _write_csv(result.summary, folder / "ablation_summary.csv")
        _write_csv(result.comparisons, folder / "ablation_comparison.csv")
        write_json(folder / "ablation.json", result.as_dict())
        if figures:
            _figure(
                folder,
                "ablation_metric_by_dimension",
                plot_ablation(result.summary, result.metric, dpi=config.output.dpi),
                ablation_table(result.summary, result.metric),
                config,
            )
        best = result.summary.iloc[0]
        _report("settings", len(result.summary))
        _report("metric", result.metric)
        _report("baseline", result.baseline)
        _report("best", best["config_id"])
        _report(
            result.metric,
            f"{best[f'{result.metric}_mean']:.3f} +/- {best[f'{result.metric}_std']:.3f}",
        )
        _finish(folder, "ablate", config, [Path(config.data.dataset)], result.as_dict())


def _named_files(items: Sequence[str]) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ConfigError(f"feature files are given as NAME=PATH, got {item!r}")
        if name.strip() in named:
            raise ConfigError(f"feature file name {name.strip()!r} is used twice")
        named[name.strip()] = Path(path.strip())
    return named


def _contrast_pairs(items: Sequence[str] | None) -> list[tuple[str, str]] | None:
    if not items:
        return None
    pairs = []
    for item in items:
        first, sep, second = item.partition(":")
        if not sep or not first.strip() or not second.strip():
            raise ConfigError(f"--contrast expects FIRST:SECOND, got {item!r}")
        pairs.append((first.strip(), second.strip()))
    return pairs


@app.command("compare")
def compare_command(
    features: Annotated[
        list[str],
        typer.Argument(help="NAME=features.npz for every feature file; the first is the baseline."),
    ],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
    contrast: Annotated[
        list[str] | None,
        typer.Option(
            "--contrast",
            help="FIRST:SECOND pair to test (repeatable); default every file against the first.",
        ),
    ] = None,
    margin: Annotated[
        float | None,
        typer.Option("--margin", min=0.0, help="Equivalence margin in units of the metric."),
    ] = None,
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric", help="Ranking metric; rmse for regression, uar for classification."
        ),
    ] = None,
    jobs: Annotated[int, typer.Option("--jobs", min=1, help="Folds fitted in parallel.")] = 1,
    figures: FiguresOption = True,
) -> None:
    """Compare feature files from different front ends on shared folds, with bootstrap intervals."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        named = _named_files(features)
        # Loaded without the configuration: the files come from different front
        # ends, so their settings cannot all match one configuration.
        loaded = {name: load_features(path) for name, path in named.items()}
        result = compare_feature_sets(loaded, config, metric=metric, n_jobs=jobs)
        contrasts = pairwise_contrasts(
            result,
            _contrast_pairs(contrast),
            correction=config.analysis.correction,
            n_resamples=config.analysis.n_resamples,
            ci_level=config.analysis.ci_level,
            equivalence_margin=margin,
            seed=config.seed,
        )
        folder = _run_folder(config, "compare", out)
        _write_csv(result.folds, folder / "comparison_folds.csv")
        _write_csv(result.items, folder / "comparison_items.csv")
        _write_csv(result.predictions, folder / "comparison_predictions.csv")
        _write_csv(result.summary, folder / "comparison_summary.csv")
        _write_csv(contrasts, folder / "comparison_contrasts.csv")
        write_json(folder / "comparison.json", result.as_dict())
        if figures:
            _figure(
                folder,
                "comparison_metric_by_feature_file",
                plot_comparison(result.summary, result.metric, dpi=config.output.dpi),
                comparison_table(result.summary, result.metric),
                config,
            )
        best = result.summary.iloc[0]
        metric = result.metric
        _report("items", len(result.items))
        _report("feature files", len(result.names))
        _report("best", best["feature_set"])
        _report(
            metric,
            f"{best[metric]:.3f} [{best[f'{metric}_low']:.3f}, {best[f'{metric}_high']:.3f}]",
        )
        _finish(folder, "compare", config, list(named.values()), result.as_dict())


@app.command("correlate")
def correlate_command(
    table: TableArgument,
    config_file: ConfigOption = None,
    set_: SetOption = None,
    out: OutOption = None,
    target: TargetOption = None,
    top: Annotated[int, typer.Option("--top", min=1, help="Features shown in the figure.")] = 20,
    figures: FiguresOption = True,
) -> None:
    """Correlate every feature with a continuous rating (Pearson with CI, Spearman)."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        X, y = load_analysis_table(table, target=target)
        result = feature_target_correlations(
            X.to_numpy(),
            y.to_numpy(),
            list(X.columns),
            ci_level=config.analysis.ci_level,
            correction=config.analysis.correction,
        )
        folder = _run_folder(config, "correlate", out)
        _write_csv(result, folder / "correlations.csv")
        if figures:
            _figure(
                folder,
                "feature_correlations",
                plot_correlations(result, top, dpi=config.output.dpi),
                correlations_table(result, top),
                config,
            )
        significant = int((result["pearson_p_adjusted"] < 0.05).sum())
        strongest = result.iloc[0]
        _report("items", len(y))
        _report("features", len(result))
        _report("p_adj<0.05", significant)
        _report("strongest", f"{strongest['feature']} (r = {strongest['pearson_r']:.3f})")
        _finish(
            folder, "correlate", config, [table], {"n_items": len(y), "n_significant": significant}
        )


def _regression_outputs(
    folder: Path,
    result: Any,
    X: pd.DataFrame,
    config: Config,
    figures: bool,
    extra: Mapping[str, Any] | None = None,
) -> None:
    _write_csv(result.coefficient_table(), folder / "coefficients.csv")
    payload = {"model": result.as_dict(), "diagnostics": residual_diagnostics(result)}
    payload.update(extra or {})
    write_json(folder / "regression_summary.json", payload)
    if X.shape[1] >= 2:
        _write_csv(vif(X.to_numpy(), list(X.columns)), folder / "vif.csv")
    if figures and result.estimable:
        _figure(
            folder,
            "regression_diagnostics",
            plot_regression_diagnostics(result, dpi=config.output.dpi),
            regression_diagnostics_table(result),
            config,
        )


@app.command("regress")
def regress_command(
    table: TableArgument,
    config_file: ConfigOption = None,
    set_: SetOption = None,
    out: OutOption = None,
    target: TargetOption = None,
    predictors: PredictorsOption = None,
    design: Annotated[
        _Design,
        typer.Option(
            "--design", help="Linear terms only, or the full second-order response surface."
        ),
    ] = _Design["linear"],
    figures: FiguresOption = True,
) -> None:
    """Ordinary least squares with coefficient inference and residual diagnostics."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        X, y = load_analysis_table(table, target=target, predictors=_predictor_list(predictors))
        if design.value == "quadratic":
            matrix, names = quadratic_surface(X.to_numpy(), list(X.columns))
            X = pd.DataFrame(matrix, columns=list(names))
        result = fit_ols(
            X.to_numpy(), y.to_numpy(), list(X.columns), ci_level=config.analysis.ci_level
        )
        folder = _run_folder(config, "regress", out)
        _regression_outputs(folder, result, X, config, figures)
        _report("items", result.n_obs)
        _report("parameters", result.n_params)
        _report("R2", f"{result.r2:.3f} (adjusted {result.adj_r2:.3f})")
        _report("F test", f"F = {result.f_stat:.3g}, p = {result.f_pvalue:.3g}")
        _report("condition", f"{result.condition_number:.3g}")
        _finish(
            folder, "regress", config, [table], {"n_items": result.n_obs, "design": design.value}
        )


@app.command("stepwise")
def stepwise_command(
    table: TableArgument,
    config_file: ConfigOption = None,
    set_: SetOption = None,
    out: OutOption = None,
    target: TargetOption = None,
    predictors: PredictorsOption = None,
    figures: FiguresOption = True,
) -> None:
    """Stepwise predictor selection by partial F tests, with variance inflation factors."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        X, y = load_analysis_table(table, target=target, predictors=_predictor_list(predictors))
        analysis = config.analysis
        result = stepwise_select(
            X.to_numpy(),
            y.to_numpy(),
            list(X.columns),
            start=analysis.stepwise_start,
            p_enter=analysis.p_enter,
            p_remove=analysis.p_remove,
            ci_level=analysis.ci_level,
        )
        folder = _run_folder(config, "stepwise", out)
        _write_csv(result.steps, folder / "stepwise_steps.csv")
        selected = X[list(result.selected)]
        _regression_outputs(
            folder, result.final, selected, config, figures, {"selection": result.as_dict()}
        )
        _report("items", result.final.n_obs)
        _report("candidates", X.shape[1])
        _report("selected", ", ".join(result.selected) or "(none)")
        _report("R2", f"{result.final.r2:.3f} (adjusted {result.final.adj_r2:.3f})")
        _finish(folder, "stepwise", config, [table], result.as_dict())


@app.command("order-sweep")
def order_sweep_command(
    table: TableArgument,
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
    x: Annotated[str, typer.Option("--x", help="Predictor column (one).")] = "x",
    target: TargetOption = None,
    figures: FiguresOption = True,
) -> None:
    """Compare polynomial degrees in sample and by cross-validation."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        X, y = load_analysis_table(table, target=target, predictors=[x])
        analysis = config.analysis
        result = sweep_polynomial_order(
            X[x].to_numpy(),
            y.to_numpy(),
            range(1, analysis.max_degree + 1),
            basis=analysis.polynomial_basis,
            n_splits=config.evaluation.n_splits,
            criterion=analysis.order_criterion,
            rng=config.seed,
        )
        folder = _run_folder(config, "order-sweep", out)
        _write_csv(order_sweep_table(result), folder / "order_sweep.csv")
        if figures:
            save_figure(
                plot_order_sweep(result, dpi=config.output.dpi),
                folder / "order_sweep",
                config.output.figure_format,
            )
        _report("items", len(y))
        _report("degrees", f"1 .. {analysis.max_degree}")
        _report("criterion", result.criterion)
        _report("chosen", result.recommended_degree)
        _finish(folder, "order-sweep", config, [table], result.as_dict())


def main() -> None:
    """Entry point of the ``voice-severity-bench`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
