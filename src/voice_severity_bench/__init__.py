"""Cepstral voice features against continuous severity ratings.

MFCCs from a configurable front end are pooled per recording and scored
against a severity rating by cross-validated regression or classification.
The package also runs an ablation over feature settings and models,
least-squares inference with regression diagnostics, and correlation and
agreement analysis between features, predictions and clinical ratings.

* :class:`Config` and its sections -- every setting, validated, stored as YAML;
* :func:`cepstral_features` / :func:`append_dynamics` / :func:`frame_features` /
  :func:`feature_layout` -- the MFCC front end with 13, 26 or 39 dimensions;
* :func:`pool_frames` / :func:`pooled_feature_names` -- utterance statistics;
* :func:`write_synthetic_dataset` / :func:`make_vowel` -- synthetic vowels with a severity rating;
* :func:`load_dataset_index` / :class:`FeatureSet` / :func:`save_features` /
  :func:`load_features` -- dataset index and feature files;
* :func:`extract_corpus` / :func:`read_audio` -- batch export of frame-level features;
* :func:`prepare_features` / :func:`train_model` / :func:`predict_inputs` -- the pipeline;
* :func:`cross_validate` / :class:`CrossValidationResult` -- grouped, stratified k-fold evaluation;
* :func:`build_model` / :func:`fit_model` / :func:`save_model` / :func:`load_model` -- estimators;
* :func:`expand_grid` / :func:`run_ablation` / :func:`compare_to_baseline` -- the ablation study;
* :func:`feature_set_from_table` / :func:`compare_feature_sets` / :func:`pairwise_contrasts` --
  feature files from other front ends compared on shared folds;
* :func:`bootstrap_metric` / :func:`paired_bootstrap` -- speaker-level bootstrap intervals;
* :func:`permute_group_labels` / :func:`permutation_test` -- speaker-level permutation null
  of a classifier, with the inner tuning repeated;
* :func:`icc_oneway` / :func:`rating_noise_floor` / :func:`leave_one_rater_out` -- rater
  agreement and the ceiling it sets for a model;
* :func:`fit_ols` / :class:`OLSResult` -- least squares with inference;
* :func:`vif` / :func:`residual_diagnostics` and the individual tests -- regression diagnostics;
* :func:`stepwise_select` / :func:`sweep_polynomial_order` / :func:`fit_linear_gd` --
  model building;
* :func:`pearson_ci` / :func:`concordance_ccc` / :func:`bland_altman` /
  :func:`feature_target_correlations` -- association and agreement;
* :func:`save_figure` with the ``plot_<thing>`` / ``<thing>_table`` pairs -- figures and
  the tables behind them;
* :class:`TorchMLPRegressor` / :class:`TorchMLPClassifier` -- PyTorch perceptrons
  (needs the dl extra).
"""

from __future__ import annotations

from .ablation import (
    AblationPoint,
    AblationResult,
    compare_to_baseline,
    corrected_resampled_ttest,
    expand_grid,
    run_ablation,
    wilcoxon_signed_rank,
)
from .association import (
    BlandAltmanResult,
    CorrelationResult,
    adjust_pvalues,
    bland_altman,
    concordance_ccc,
    feature_target_correlations,
    pearson_ci,
    spearman,
)
from .bootstrap import (
    BootstrapInterval,
    bootstrap_metric,
    group_codes,
    group_weights,
    paired_bootstrap,
    weighted_metric,
)
from .cepstral_frontend import (
    append_dynamics,
    cepstral_features,
    compact_preset,
    feature_layout,
    frame_features,
    htk_c0_preset,
    htk_dct,
    sinusoidal_lifter,
)
from .comparison import (
    ComparisonResult,
    align_feature_sets,
    compare_feature_sets,
    pairwise_contrasts,
    repeat_averaged_predictions,
)
from .config import (
    FEATURE_SECTIONS,
    FIGURE_FORMATS,
    AblationConfig,
    AnalysisConfig,
    AudioConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    FrontendConfig,
    ModelConfig,
    OutputConfig,
    PoolingConfig,
    SyntheticConfig,
)
from .corpus import (
    extract_corpus,
    htk_parameter_kind,
    iter_audio_files,
    load_frame_features,
    read_audio,
    read_text_matrix,
)
from .dataset import (
    DATASET_INDEX_NAME,
    FeatureSet,
    build_dataset_index,
    feature_set_from_table,
    load_analysis_table,
    load_dataset_index,
    load_features,
    save_features,
)
from .design_matrix import PolynomialBasis, polynomial_features, quadratic_surface
from .errors import (
    ConfigError,
    DatasetError,
    DesignMatrixError,
    MissingDependencyError,
    ModelFileError,
    require_module,
)
from .evaluation import (
    CrossValidationResult,
    classification_metrics,
    cross_validate,
    regression_metrics,
)
from .figures import (
    ablation_table,
    bland_altman_table,
    comparison_table,
    cost_history_table,
    plot_ablation,
    plot_bland_altman,
    plot_comparison,
    plot_correlations,
    plot_cost_history,
    plot_order_sweep,
    plot_regression_diagnostics,
    plot_residuals,
    plot_response_surface,
    regression_diagnostics_table,
    response_surface_table,
    save_figure,
)
from .gradient_descent import GradientDescentResult, fit_linear_gd
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import MODEL_NAMES, BlockSelector, build_model, fit_model, load_model, save_model
from .neural import TorchMLPClassifier, TorchMLPRegressor
from .ols import OLSResult, fit_ols
from .permutation import PermutationResult, permutation_test, permute_group_labels
from .pipeline import predict_inputs, prepare_features, train_model, utterance_features
from .polynomial_order import OrderSweepResult, sweep_polynomial_order
from .pooling import pool_frames, pooled_feature_names
from .reference_data import cement_hardening_heat, shrout_fleiss_ratings
from .regression_diagnostics import (
    HypothesisTest,
    breusch_pagan,
    condition_number,
    cooks_distance,
    durbin_watson,
    jarque_bera,
    leverage,
    residual_diagnostics,
    shapiro_wilk,
    vif,
)
from .reliability import ICCResult, icc_oneway, leave_one_rater_out, rating_noise_floor
from .splitting import repeated_split_indices, split_indices
from .stepwise import StepwiseResult, stepwise_select
from .synthetic import (
    make_collinear_data,
    make_polynomial_data,
    make_quadratic_surface_data,
    make_vowel,
    write_synthetic_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "DATASET_INDEX_NAME",
    "FEATURE_SECTIONS",
    "FIGURE_FORMATS",
    "MODEL_NAMES",
    "AblationConfig",
    "AblationPoint",
    "AblationResult",
    "AnalysisConfig",
    "AudioConfig",
    "BlandAltmanResult",
    "BlockSelector",
    "BootstrapInterval",
    "ComparisonResult",
    "Config",
    "ConfigError",
    "CorrelationResult",
    "CrossValidationResult",
    "DataConfig",
    "DatasetError",
    "DesignMatrixError",
    "EvaluationConfig",
    "FeatureSet",
    "FrontendConfig",
    "GradientDescentResult",
    "HypothesisTest",
    "ICCResult",
    "MissingDependencyError",
    "ModelConfig",
    "ModelFileError",
    "OLSResult",
    "OrderSweepResult",
    "OutputConfig",
    "PermutationResult",
    "PolynomialBasis",
    "PoolingConfig",
    "StepwiseResult",
    "SyntheticConfig",
    "TorchMLPClassifier",
    "TorchMLPRegressor",
    "__version__",
    "ablation_table",
    "adjust_pvalues",
    "align_feature_sets",
    "append_dynamics",
    "bland_altman",
    "bland_altman_table",
    "bootstrap_metric",
    "breusch_pagan",
    "build_dataset_index",
    "build_manifest",
    "build_model",
    "cement_hardening_heat",
    "cepstral_features",
    "classification_metrics",
    "compact_preset",
    "compare_feature_sets",
    "compare_to_baseline",
    "comparison_table",
    "concordance_ccc",
    "condition_number",
    "configure_logging",
    "cooks_distance",
    "corrected_resampled_ttest",
    "cost_history_table",
    "cross_validate",
    "durbin_watson",
    "expand_grid",
    "extract_corpus",
    "feature_layout",
    "feature_set_from_table",
    "feature_target_correlations",
    "fit_linear_gd",
    "fit_model",
    "fit_ols",
    "frame_features",
    "group_codes",
    "group_weights",
    "htk_c0_preset",
    "htk_dct",
    "htk_parameter_kind",
    "icc_oneway",
    "iter_audio_files",
    "jarque_bera",
    "leave_one_rater_out",
    "leverage",
    "load_analysis_table",
    "load_dataset_index",
    "load_features",
    "load_frame_features",
    "load_model",
    "make_collinear_data",
    "make_polynomial_data",
    "make_quadratic_surface_data",
    "make_vowel",
    "paired_bootstrap",
    "pairwise_contrasts",
    "pearson_ci",
    "permutation_test",
    "permute_group_labels",
    "plot_ablation",
    "plot_bland_altman",
    "plot_comparison",
    "plot_correlations",
    "plot_cost_history",
    "plot_order_sweep",
    "plot_regression_diagnostics",
    "plot_residuals",
    "plot_response_surface",
    "polynomial_features",
    "pool_frames",
    "pooled_feature_names",
    "predict_inputs",
    "prepare_features",
    "quadratic_surface",
    "rating_noise_floor",
    "read_audio",
    "read_text_matrix",
    "regression_diagnostics_table",
    "regression_metrics",
    "repeat_averaged_predictions",
    "repeated_split_indices",
    "require_module",
    "residual_diagnostics",
    "response_surface_table",
    "run_ablation",
    "run_directory",
    "save_features",
    "save_figure",
    "save_model",
    "shapiro_wilk",
    "shrout_fleiss_ratings",
    "sinusoidal_lifter",
    "spearman",
    "split_indices",
    "stepwise_select",
    "sweep_polynomial_order",
    "train_model",
    "utterance_features",
    "vif",
    "weighted_metric",
    "wilcoxon_signed_rank",
    "write_json",
    "write_manifest",
    "write_synthetic_dataset",
]
