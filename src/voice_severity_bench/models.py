"""Estimator registry, nested hyperparameter tuning and model files.

Every model is a scikit-learn :class:`~sklearn.pipeline.Pipeline`

``select`` (blocked feature files only) -> ``impute`` (``model.impute``) ->
``scale`` (StandardScaler) -> ``reduce`` (none, PCA or LDA) -> ``model``

so imputation, standardization and dimensionality reduction are learned on
the training fold only, never on the data they are evaluated on. Regression estimators are
ordinary least squares, ridge (Hoerl & Kennard, 1970), support vector
regression (Drucker et al., 1997), random forests (Breiman, 2001) and a
multilayer perceptron; classification estimators are logistic regression,
a support vector classifier, random forests and a perceptron with a logistic
hidden layer (all from scikit-learn, Pedregosa et al., 2011), plus a PyTorch
perceptron for either task (``dl`` extra). Support vector regression and the
regression perceptron predict a standardized target, so ``svr_epsilon`` is in
standard deviations of the rating. Linear discriminant analysis (Fisher, 1936)
keeps at most ``n_classes - 1`` directions; asking for more is reported as a
configuration error before anything is fitted.

With ``model.tune`` each training fold runs its own inner cross-validation
over a small grid (:data:`SEARCH_SPACES`) and refits the best setting, so the
outer folds measure the whole selection procedure (Varma & Simon, 2006). A
feature file divided into blocks, such as the layers of a speech
representation network, can have the block chosen in the same inner loop
(``model.block_selection``): picking the best layer on the test folds would
make every layer-wise model look better than it is.

For ridge regression without a reducer the inner search does not refit the
pipeline once per penalty. Each inner training fold is imputed, scaled and
decomposed once, :math:`X = U S V^T`, and every penalty follows from the
same factors, :math:`\\hat\\beta_\\alpha = V \\operatorname{diag}(s / (s^2 + \\alpha)) U^T y`
(Hastie, Tibshirani & Friedman, 2009, section 3.4.1). The choice is the one
GridSearchCV would make, at a fraction of the cost for wide feature blocks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.validation import check_is_fitted

from .errors import ConfigError, ModelFileError, require_module
from .neural import TorchMLPClassifier, TorchMLPRegressor
from .splitting import split_indices

if TYPE_CHECKING:
    from .config import Config, ModelConfig

_LOGGER = logging.getLogger(__name__)

#: Hidden activations accepted by ``model.activation``.
ACTIVATIONS: tuple[str, ...] = ("logistic", "relu", "tanh")
#: Dimensionality reductions accepted by ``model.reducer``.
REDUCERS: tuple[str, ...] = ("none", "pca", "lda")
#: Missing-value treatments accepted by ``model.impute``.
IMPUTERS: tuple[str, ...] = ("none", "median")
#: Class weightings accepted by ``model.class_weight``.
CLASS_WEIGHTS: tuple[str, ...] = ("balanced",)
#: Classifiers that take a class weighting. The perceptrons have no such
#: parameter, so unbalanced classes there are handled by thresholds or resampling.
_WEIGHTED_CLASSIFIERS: tuple[str, ...] = ("logistic", "svc", "random_forest")
#: Value of the ``format`` key of every model file written here.
MODEL_FORMAT: str = "voice_severity_bench/model-v1"
#: Share of variance kept by PCA when ``model.n_components`` is null.
_PCA_VARIANCE: float = 0.95


def _linear(config: ModelConfig, seed: int) -> BaseEstimator:
    return LinearRegression()


def _ridge(config: ModelConfig, seed: int) -> BaseEstimator:
    return Ridge(alpha=config.ridge_alpha)


def _svr(config: ModelConfig, seed: int) -> BaseEstimator:
    return TransformedTargetRegressor(
        regressor=SVR(C=config.svr_c, epsilon=config.svr_epsilon), transformer=StandardScaler()
    )


def _forest_regressor(config: ModelConfig, seed: int) -> BaseEstimator:
    return RandomForestRegressor(
        n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=seed, n_jobs=1
    )


def _mlp_regressor(config: ModelConfig, seed: int) -> BaseEstimator:
    return TransformedTargetRegressor(
        regressor=MLPRegressor(
            hidden_layer_sizes=tuple(config.hidden_units),
            activation=config.activation,
            alpha=config.mlp_alpha,
            learning_rate_init=config.learning_rate,
            max_iter=config.max_iter,
            early_stopping=config.early_stopping,
            n_iter_no_change=config.patience,
            random_state=seed,
        ),
        transformer=StandardScaler(),
    )


def _torch_regressor(config: ModelConfig, seed: int) -> BaseEstimator:
    return TorchMLPRegressor(
        hidden_units=tuple(config.hidden_units),
        activation=config.activation,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.mlp_alpha,
        random_state=seed,
    )


def _logistic(config: ModelConfig, seed: int) -> BaseEstimator:
    return LogisticRegression(
        C=config.logistic_c, max_iter=config.max_iter, class_weight=config.class_weight
    )


def _svc(config: ModelConfig, seed: int) -> BaseEstimator:
    return SVC(
        C=config.svc_c, probability=True, random_state=seed, class_weight=config.class_weight
    )


def _forest_classifier(config: ModelConfig, seed: int) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=seed,
        n_jobs=1,
        class_weight=config.class_weight,
    )


class CodedMLPClassifier(MLPClassifier):
    """Multilayer perceptron classifier that trains on integer class codes.

    Behaves like :class:`sklearn.neural_network.MLPClassifier` (same
    parameters, ``classes_`` holds the original names) but encodes the labels
    before fitting, because the early-stopping score of some scikit-learn
    releases fails on string labels.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> CodedMLPClassifier:
        """Fit on the codes of the sorted class names."""
        names, codes = np.unique(np.asarray(y).astype(str), return_inverse=True)
        super().fit(X, codes)
        self.class_names_ = names
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Class names of the most probable classes."""
        return self.class_names_[np.asarray(super().predict(X), dtype=np.intp)]

    @property
    def classes_(self) -> np.ndarray:
        """Class names, in the column order of :meth:`predict_proba`."""
        return self.class_names_

    @classes_.setter
    def classes_(self, value: np.ndarray) -> None:
        # The parent assigns its integer codes here during fitting; they stay internal.
        self._codes = value


def _mlp_classifier(config: ModelConfig, seed: int) -> BaseEstimator:
    return CodedMLPClassifier(
        hidden_layer_sizes=tuple(config.hidden_units),
        activation=config.activation,
        alpha=config.mlp_alpha,
        learning_rate_init=config.learning_rate,
        max_iter=config.max_iter,
        early_stopping=config.early_stopping,
        n_iter_no_change=config.patience,
        random_state=seed,
    )


def _torch_classifier(config: ModelConfig, seed: int) -> BaseEstimator:
    return TorchMLPClassifier(
        hidden_units=tuple(config.hidden_units),
        activation=config.activation,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.mlp_alpha,
        random_state=seed,
    )


_Factory = Callable[["ModelConfig", int], BaseEstimator]

#: Column ranges ``name -> (start, stop)`` of a blocked feature matrix.
Blocks = Mapping[str, tuple[int, int]]


class BlockSelector(TransformerMixin, BaseEstimator):
    """Keep the columns of one block, or of several blocks side by side.

    Parameters
    ----------
    blocks : mapping of str to (int, int), optional
        Half-open column range of every block.
    block : str or tuple of str, optional
        The block (or blocks, concatenated in this order) to keep; ``None``
        keeps every column.

    Examples
    --------
    >>> X = np.arange(12.0).reshape(2, 6)
    >>> blocks = {"layer0": (0, 2), "layer1": (2, 4), "meta": (4, 6)}
    >>> BlockSelector(blocks, "layer1").fit_transform(X).tolist()
    [[2.0, 3.0], [8.0, 9.0]]
    >>> BlockSelector(blocks, ("layer0", "meta")).fit_transform(X).shape
    (2, 4)
    """

    def __init__(self, blocks: Blocks | None = None, block: str | tuple[str, ...] | None = None):
        self.blocks = blocks
        self.block = block

    def _columns(self, n_features: int) -> np.ndarray:
        if self.block is None:
            return np.arange(n_features)
        names = (self.block,) if isinstance(self.block, str) else tuple(self.block)
        blocks = dict(self.blocks or {})
        unknown = [name for name in names if name not in blocks]
        if unknown:
            raise ConfigError(f"unknown feature block(s) {unknown}; available: {list(blocks)}")
        columns = np.concatenate([np.arange(*blocks[name]) for name in names])
        if columns.size == 0 or columns.max() >= n_features:
            raise ConfigError(f"block(s) {list(names)} do not fit a matrix of {n_features} columns")
        return columns

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BlockSelector:
        """Resolve the kept columns for matrices of this width."""
        width = np.asarray(X).shape[1]
        self.columns_ = self._columns(width)
        self.n_features_in_ = width
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """The kept columns of ``X``."""
        check_is_fitted(self, "columns_")
        matrix = np.asarray(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} columns; the selector was fitted on {self.n_features_in_}"
            )
        return matrix[:, self.columns_]


_REGRESSORS: dict[str, _Factory] = {
    "linear": _linear,
    "ridge": _ridge,
    "svr": _svr,
    "random_forest": _forest_regressor,
    "mlp": _mlp_regressor,
    "torch_mlp": _torch_regressor,
}
_CLASSIFIERS: dict[str, _Factory] = {
    "logistic": _logistic,
    "svc": _svc,
    "random_forest": _forest_classifier,
    "mlp": _mlp_classifier,
    "torch_mlp": _torch_classifier,
}
#: Estimators accepted for ``data.task = "regression"``.
REGRESSION_MODELS: tuple[str, ...] = tuple(_REGRESSORS)
#: Estimators accepted for ``data.task = "classification"``.
CLASSIFICATION_MODELS: tuple[str, ...] = tuple(_CLASSIFIERS)
#: Model names accepted by :func:`build_model` and ``model.name``.
MODEL_NAMES: tuple[str, ...] = tuple(dict.fromkeys((*_REGRESSORS, *_CLASSIFIERS)))

#: Hyperparameter grids searched by the inner loop when ``model.tune`` is true.
SEARCH_SPACES: dict[tuple[str, str], dict[str, list[Any]]] = {
    ("regression", "ridge"): {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ("regression", "svr"): {
        "model__regressor__C": [0.1, 1.0, 10.0],
        "model__regressor__epsilon": [0.05, 0.1, 0.5],
    },
    ("regression", "random_forest"): {
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 3],
    },
    ("regression", "mlp"): {"model__regressor__alpha": [1e-4, 1e-2, 1.0]},
    ("classification", "logistic"): {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ("classification", "svc"): {"model__C": [0.1, 1.0, 10.0]},
    ("classification", "random_forest"): {
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 3],
    },
    ("classification", "mlp"): {"model__alpha": [1e-4, 1e-2, 1.0]},
}
#: Inner-loop score of the hyperparameter search (higher is better).
_SEARCH_SCORING: dict[str, str] = {
    "regression": "neg_root_mean_squared_error",
    "classification": "balanced_accuracy",
}


def models_for_task(task: str) -> tuple[str, ...]:
    """Model names that fit a task.

    Examples
    --------
    >>> models_for_task("classification")
    ('logistic', 'svc', 'random_forest', 'mlp', 'torch_mlp')
    """
    if task == "regression":
        return REGRESSION_MODELS
    if task == "classification":
        return CLASSIFICATION_MODELS
    raise ConfigError(
        f"data.task={task!r} is not supported; choose 'regression' or 'classification'"
    )


def _reducer(config: ModelConfig) -> BaseEstimator | str:
    if config.reducer == "pca":
        components = _PCA_VARIANCE if config.n_components is None else config.n_components
        return PCA(n_components=components, svd_solver="full")
    if config.reducer == "lda":
        return LinearDiscriminantAnalysis(n_components=config.n_components)
    return "passthrough"


def _imputer(config: ModelConfig) -> SimpleImputer | None:
    if config.impute == "median":
        # keep_empty_features: a column without a single value in a training
        # fold keeps its place, so the indicator columns stay aligned.
        return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    return None


def build_model(config: Config, *, blocks: Blocks | None = None) -> BaseEstimator:
    """Untrained pipeline for ``config.model`` and ``config.data.task``.

    Parameters
    ----------
    config : Config
        Run configuration; ``config.seed`` seeds every randomized estimator.
    blocks : mapping of str to (int, int), optional
        Column ranges of a blocked feature matrix; adds a ``select`` step
        (:class:`BlockSelector`) that keeps every column until a block is set.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps ``select`` (with ``blocks``), ``impute`` (with
        ``model.impute: median``), ``scale``, ``reduce`` and ``model``. The
        PyTorch perceptron standardizes internally, so its pipeline has no
        ``scale`` and ``reduce`` steps.

    Raises
    ------
    ConfigError
        If the model does not fit the task, or ``model.class_weight`` is set
        for an estimator that cannot use it.

    Examples
    --------
    >>> from voice_severity_bench.config import Config
    >>> pipeline = build_model(Config().override({"model.name": "svr", "model.reducer": "pca"}))
    >>> [name for name, _ in pipeline.steps], type(pipeline[-1].regressor).__name__
    (['scale', 'reduce', 'model'], 'SVR')
    >>> pipeline = build_model(Config().override({"model.impute": "median"}), blocks={"a": (0, 3)})
    >>> [name for name, _ in pipeline.steps]
    ['select', 'impute', 'scale', 'reduce', 'model']
    """
    task = config.data.task
    registry = _REGRESSORS if task == "regression" else _CLASSIFIERS
    name = config.model.name
    if name not in registry:
        raise ConfigError(
            f"model.name={name!r} does not fit data.task={task!r}; choose one of {list(registry)}"
        )
    if config.model.class_weight is not None and (
        task != "classification" or name not in _WEIGHTED_CLASSIFIERS
    ):
        raise ConfigError(
            f"model.class_weight works for the classifiers {list(_WEIGHTED_CLASSIFIERS)}, "
            f"not for {name!r} with data.task={task!r}"
        )
    estimator = registry[name](config.model, config.seed)
    steps: list[tuple[str, Any]] = []
    if blocks is not None:
        steps.append(("select", BlockSelector(dict(blocks))))
    imputer = _imputer(config.model)
    if imputer is not None:
        steps.append(("impute", imputer))
    if name != "torch_mlp":
        steps += [("scale", StandardScaler()), ("reduce", _reducer(config.model))]
    steps.append(("model", estimator))
    return Pipeline(steps)


def search_space(
    config: Config,
    *,
    blocks: Blocks | None = None,
    block_candidates: Sequence[str | tuple[str, ...]] | None = None,
) -> dict[str, list[Any]]:
    """Hyperparameter grid of the configured model (empty when nothing is tuned).

    Parameters
    ----------
    config : Config
        Run configuration; a non-empty ``model.ridge_alphas`` replaces the ridge grid.
    blocks : mapping of str to (int, int), optional
        Column ranges; with ``model.block_selection`` the block joins the grid.
    block_candidates : sequence, optional
        Blocks (names, or tuples of names kept together) to choose from;
        every single block when omitted.

    Examples
    --------
    >>> from voice_severity_bench.config import Config
    >>> search_space(Config())
    {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    >>> config = Config().override(
    ...     {"model.tune": True, "model.block_selection": True, "model.ridge_alphas": [1.0, 10.0]}
    ... )
    >>> search_space(config, blocks={"layer0": (0, 4), "layer1": (4, 8)})
    {'model__alpha': [1.0, 10.0], 'select__block': ['layer0', 'layer1']}
    """
    grid = {
        key: list(values)
        for key, values in SEARCH_SPACES.get((config.data.task, config.model.name), {}).items()
    }
    if config.model.ridge_alphas and config.model.name == "ridge":
        grid["model__alpha"] = [float(value) for value in config.model.ridge_alphas]
    if config.model.block_selection and blocks:
        candidates = list(blocks) if block_candidates is None else list(block_candidates)
        grid["select__block"] = [
            candidate if isinstance(candidate, str) else tuple(candidate)
            for candidate in candidates
        ]
    return grid


def _check_reducer(config: Config, y: np.ndarray, n_features: int) -> None:
    if config.model.reducer != "lda":
        return
    n_classes = np.unique(y.astype(str)).size
    limit = min(n_classes - 1, n_features)
    requested = config.model.n_components
    if requested is not None and requested > limit:
        raise ConfigError(
            f"model.n_components={requested} exceeds what LDA can extract from {n_classes} "
            f"classes and {n_features} features (at most {limit} discriminant directions)"
        )


def fit_model(
    X: ArrayLike,
    y: ArrayLike,
    config: Config,
    *,
    groups: ArrayLike | None = None,
    blocks: Blocks | None = None,
    block_candidates: Sequence[str | tuple[str, ...]] | None = None,
) -> BaseEstimator:
    """Fit the configured pipeline, tuning it by inner cross-validation if asked.

    Parameters
    ----------
    X : array_like, shape (n_items, n_features)
        Training features.
    y : array_like, shape (n_items,)
        Training targets.
    config : Config
        Run configuration (``model``, ``data.task``, ``evaluation.n_inner_splits``, ``seed``).
    groups : array_like, shape (n_items,), optional
        Speakers; the inner folds never split one.
    blocks, block_candidates : optional
        Column blocks of ``X`` and the choices of ``model.block_selection``
        (see :func:`search_space`).

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline (the refitted best one when tuning). The chosen
        values are its own parameters, e.g. ``pipeline.get_params()["select__block"]``.

    Examples
    --------
    >>> from voice_severity_bench.config import Config
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 3))
    >>> y = X @ np.array([1.0, 0.0, -2.0]) + rng.normal(0, 0.1, 40)
    >>> tuned = fit_model(X, y, Config().override({"model.tune": True}))
    >>> float(tuned[-1].alpha) in (0.01, 0.1, 1.0)
    True
    """
    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y)
    if config.data.task == "regression":
        targets = targets.astype(np.float64)
    else:
        targets = targets.astype(str)
    _check_reducer(config, targets, features.shape[1])
    pipeline = build_model(config, blocks=blocks)
    grid = (
        search_space(config, blocks=blocks, block_candidates=block_candidates)
        if config.model.tune
        else {}
    )
    if not grid:
        if config.model.tune:
            _LOGGER.info(
                "model %r has no hyperparameter grid; fitting it as configured", config.model.name
            )
        return pipeline.fit(features, targets)
    folds = split_indices(
        targets,
        groups,
        task=config.data.task,
        n_splits=config.evaluation.n_inner_splits,
        random_state=config.seed,
    )
    if _ridge_path_applies(config, grid):
        best = _ridge_path_search(pipeline, grid, features, targets, folds)
        _LOGGER.debug("inner search chose %s", best)
        return pipeline.set_params(**best).fit(features, targets)
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=_SEARCH_SCORING[config.data.task],
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(features, targets)
    _LOGGER.debug("inner search chose %s", search.best_params_)
    return search.best_estimator_


def _ridge_path_applies(config: Config, grid: Mapping[str, Sequence[Any]]) -> bool:
    return (
        config.data.task == "regression"
        and config.model.name == "ridge"
        and config.model.reducer == "none"
        and "model__alpha" in grid
        and set(grid) <= {"model__alpha", "select__block"}
    )


def _ridge_path_search(
    pipeline: Pipeline,
    grid: Mapping[str, Sequence[Any]],
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Best ``model__alpha`` (and ``select__block``) by mean inner-fold RMSE, as GridSearchCV.

    Ties go to the first setting in GridSearchCV's order, where the block
    varies fastest.
    """
    alphas = np.asarray(grid["model__alpha"], dtype=np.float64)
    blocks = list(grid.get("select__block", [None]))
    scores = np.zeros((alphas.size, len(blocks)))
    for column, block in enumerate(blocks):
        steps = clone(pipeline[:-1])
        if block is not None:
            steps.set_params(select__block=block)
        for train, test in folds:
            fitted = clone(steps)
            X_train = fitted.fit_transform(X[train], y[train])
            X_test = fitted.transform(X[test])
            x_mean, y_mean = X_train.mean(axis=0), y[train].mean()
            U, singular, Vt = np.linalg.svd(X_train - x_mean, full_matrices=False)
            projected = U.T @ (y[train] - y_mean)
            test_scores = (X_test - x_mean) @ Vt.T
            shrink = singular[None, :] / (singular[None, :] ** 2 + alphas[:, None])
            predicted = (shrink * projected[None, :]) @ test_scores.T + y_mean
            errors = np.sqrt(np.mean((predicted - y[test][None, :]) ** 2, axis=1))
            scores[:, column] += errors / len(folds)
    row, column = np.unravel_index(int(np.argmin(scores)), scores.shape)
    best: dict[str, Any] = {"model__alpha": float(alphas[row])}
    if "select__block" in grid:
        best["select__block"] = blocks[column]
    return best


def model_filename(config: Config) -> str:
    """``model.pt`` for the PyTorch perceptron, ``model.joblib`` for everything else."""
    return "model.pt" if config.model.name == "torch_mlp" else "model.joblib"


def save_model(
    path: str | Path,
    estimator: Any,
    config: Config,
    classes: Sequence[str] | None = None,
) -> Path:
    """Write a trained pipeline with the configuration that produced it.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file (``model.joblib``, or ``model.pt`` for ``torch_mlp``).
    estimator : sklearn.pipeline.Pipeline
        Trained pipeline.
    config : Config
        Complete configuration; ``predict`` extracts features with it.
    classes : sequence of str, optional
        Class names (classification only).

    Returns
    -------
    pathlib.Path
        The written file.
    """
    from . import __version__  # deferred: the package __init__ imports this module

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "package_version": __version__,
        "config": config.to_dict(),
    }
    if classes is not None:
        bundle["classes"] = [str(name) for name in classes]
    if config.model.name == "torch_mlp":
        torch = require_module("torch", extra="dl", feature="saving the PyTorch perceptron")
        bundle["state_dict"] = estimator[-1].network_state()
        torch.save(bundle, destination)
    else:
        bundle["estimator"] = estimator
        joblib.dump(bundle, destination)
    _LOGGER.info("model written to %s", destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model file written by :func:`save_model`.

    Only open model files from sources you trust: joblib and PyTorch files
    written with pickle can execute code when loaded.

    Parameters
    ----------
    path : str or pathlib.Path
        ``model.joblib`` or ``model.pt``.

    Returns
    -------
    dict
        Keys ``format``, ``package_version``, ``config``, ``classes`` (when
        present) and ``estimator`` (a ready-to-use pipeline, rebuilt from the
        ``state_dict`` for PyTorch models).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ModelFileError
        If the file was not written by this package or is incomplete.
    """
    from .config import Config

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"model file not found: {source}. Run 'voice-severity-bench train' first"
        )
    if source.suffix == ".pt":
        torch = require_module("torch", extra="dl", feature="loading a PyTorch model")
        bundle = torch.load(source, map_location="cpu", weights_only=True)
    else:
        try:
            bundle = joblib.load(source)
        except Exception as exc:  # joblib raises many unrelated types for foreign files
            raise ModelFileError(f"{source} is not a readable model file: {exc}") from exc
    if not isinstance(bundle, Mapping) or bundle.get("format") != MODEL_FORMAT:
        raise ModelFileError(
            f"{source} was not written by voice-severity-bench (expected format {MODEL_FORMAT!r})"
        )
    missing = [key for key in ("config", "package_version") if key not in bundle]
    if missing:
        raise ModelFileError(f"{source} lacks {missing}")
    loaded = dict(bundle)
    config = Config.from_dict(loaded["config"])
    if "state_dict" in loaded:
        pipeline = build_model(config)
        pipeline[-1].load_network_state(loaded.pop("state_dict"), loaded.get("classes"))
        loaded["estimator"] = pipeline
    elif "estimator" not in loaded:
        raise ModelFileError(f"{source} holds no trained estimator")
    if config.data.task == "classification" and "classes" not in loaded:
        raise ModelFileError(f"{source} is a classification model without its class names")
    return loaded
