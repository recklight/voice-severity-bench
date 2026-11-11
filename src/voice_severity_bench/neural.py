"""Multilayer perceptrons in PyTorch, wrapped as scikit-learn estimators (``dl`` extra).

The networks have the same structure as the scikit-learn perceptrons in
:mod:`voice_severity_bench.models`: fully connected hidden layers with a
logistic, ReLU or tanh activation, a linear output for regression and a
softmax output trained with cross-entropy for classification. They train with
Adam (Kingma & Ba, 2015) on mini-batches, on the CPU. The layers keep
PyTorch's default initialization, uniform on +-1/sqrt(fan_in): the "commonly
used heuristic" of Glorot & Bengio (2010, eq. 1), not the normalized
initialization they propose.

Inputs (and, for regression, targets) are standardized with statistics of the
training data that are stored as buffers of the network, so a saved
``state_dict`` is the complete model. PyTorch is imported only inside the
methods: the module (and the whole package) imports without it, and a missing
installation raises :class:`~voice_severity_bench.errors.MissingDependencyError`
naming the ``dl`` extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .errors import require_module

#: Activation names and the matching ``torch.nn`` layer classes.
_ACTIVATION_LAYERS: dict[str, str] = {"logistic": "Sigmoid", "relu": "ReLU", "tanh": "Tanh"}


def _torch() -> Any:
    return require_module("torch", extra="dl", feature="the PyTorch perceptron (torch_mlp)")


def build_network(
    torch: Any,
    n_features: int,
    n_outputs: int,
    hidden_units: Sequence[int],
    activation: str,
) -> Any:
    """A fully connected network with standardization buffers.

    Parameters
    ----------
    torch : module
        The imported ``torch`` package.
    n_features, n_outputs : int
        Input and output widths.
    hidden_units : sequence of int
        Hidden layer sizes.
    activation : {"logistic", "relu", "tanh"}
        Hidden activation.

    Returns
    -------
    torch.nn.Module
        Network whose buffers ``x_mean``, ``x_scale``, ``y_mean`` and
        ``y_scale`` hold the standardization of inputs and outputs.
    """

    class _Perceptron(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            width = n_features
            for units in hidden_units:
                layers.append(torch.nn.Linear(width, int(units)))
                layers.append(getattr(torch.nn, _ACTIVATION_LAYERS[activation])())
                width = int(units)
            layers.append(torch.nn.Linear(width, n_outputs))
            self.layers = torch.nn.Sequential(*layers)
            self.register_buffer("x_mean", torch.zeros(n_features, dtype=torch.float32))
            self.register_buffer("x_scale", torch.ones(n_features, dtype=torch.float32))
            self.register_buffer("y_mean", torch.zeros(1, dtype=torch.float32))
            self.register_buffer("y_scale", torch.ones(1, dtype=torch.float32))

        def forward(self, inputs: Any) -> Any:
            return self.layers((inputs - self.x_mean) / self.x_scale)

    return _Perceptron()


class _TorchPerceptron(BaseEstimator):
    """Shared training loop of the regressor and the classifier."""

    _task: str = "regression"

    def __init__(
        self,
        hidden_units: Sequence[int] = (100,),
        activation: str = "logistic",
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        random_state: int = 0,
    ) -> None:
        self.hidden_units = hidden_units
        self.activation = activation
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _prepare_targets(self, torch: Any, y: NDArray[Any]) -> tuple[Any, int]:
        """Target tensor and output width: float column or class codes."""
        if self._task == "classification":
            self.classes_, codes = np.unique(np.asarray(y).astype(str), return_inverse=True)
            return torch.as_tensor(codes, dtype=torch.long), int(self.classes_.size)
        values = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        return torch.as_tensor(values, dtype=torch.float32), 1

    def _loss(self, torch: Any) -> Any:
        """Mean squared error for regression, cross-entropy on logits for classification."""
        if self._task == "classification":
            return torch.nn.CrossEntropyLoss()
        return torch.nn.MSELoss()

    def fit(self, X: ArrayLike, y: ArrayLike) -> _TorchPerceptron:
        """Train the network on standardized inputs with Adam."""
        torch = _torch()
        torch.manual_seed(int(self.random_state))
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {features.shape}")
        targets, n_outputs = self._prepare_targets(torch, np.asarray(y))
        self.n_features_in_ = features.shape[1]
        network = build_network(
            torch, self.n_features_in_, n_outputs, self.hidden_units, self.activation
        )
        scale = features.std(axis=0)
        network.x_mean.copy_(torch.as_tensor(features.mean(axis=0), dtype=torch.float32))
        network.x_scale.copy_(torch.as_tensor(np.where(scale > 0, scale, 1.0), dtype=torch.float32))
        if self._task == "regression":
            raw = np.asarray(y, dtype=np.float64)
            spread = float(raw.std()) or 1.0
            network.y_mean.fill_(float(raw.mean()))
            network.y_scale.fill_(spread)
            targets = (targets - float(raw.mean())) / spread
        inputs = torch.as_tensor(features, dtype=torch.float32)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=float(self.learning_rate), weight_decay=self.weight_decay
        )
        loss_function = self._loss(torch)
        generator = torch.Generator().manual_seed(int(self.random_state))
        n_items = inputs.shape[0]
        network.train()
        for _ in range(int(self.n_epochs)):
            order = torch.randperm(n_items, generator=generator)
            for start in range(0, n_items, int(self.batch_size)):
                batch = order[start : start + int(self.batch_size)]
                optimizer.zero_grad()
                loss = loss_function(network(inputs[batch]), targets[batch])
                loss.backward()
                optimizer.step()
        network.eval()
        self.network_ = network
        return self

    def _raw_outputs(self, X: ArrayLike) -> NDArray[np.float64]:
        torch = _torch()
        features = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float32)
        with torch.no_grad():
            return self.network_(features).cpu().numpy().astype(np.float64)

    def network_state(self) -> dict[str, Any]:
        """The trained ``state_dict`` (tensors only), for ``model.pt``."""
        return dict(self.network_.state_dict())

    def load_network_state(self, state: Mapping[str, Any], classes: Sequence[str] | None) -> None:
        """Rebuild the network from :meth:`network_state` output."""
        torch = _torch()
        first = state["layers.0.weight"]
        last_key = max(
            (key for key in state if key.startswith("layers.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        network = build_network(
            torch,
            int(first.shape[1]),
            int(state[last_key].shape[0]),
            self.hidden_units,
            self.activation,
        )
        network.load_state_dict(dict(state))
        network.eval()
        self.network_ = network
        self.n_features_in_ = int(first.shape[1])
        if classes is not None:
            self.classes_ = np.asarray(list(classes))


class TorchMLPRegressor(RegressorMixin, _TorchPerceptron):
    """PyTorch perceptron for continuous targets (mean squared error loss).

    Parameters
    ----------
    hidden_units : sequence of int, optional
        Hidden layer sizes.
    activation : {"logistic", "relu", "tanh"}, optional
        Hidden activation.
    n_epochs : int, optional
        Passes over the training data.
    learning_rate : float, optional
        Adam step size.
    batch_size : int, optional
        Mini-batch size.
    weight_decay : float, optional
        L2 penalty of Adam.
    random_state : int, optional
        Seed of the initialization and of the mini-batch order.

    References
    ----------
    .. [1] X. Glorot and Y. Bengio, "Understanding the difficulty of training
       deep feedforward neural networks," in Proc. 13th International
       Conference on Artificial Intelligence and Statistics (AISTATS), PMLR
       vol. 9, 2010, pp. 249-256.
    .. [2] D. P. Kingma and J. Ba, "Adam: A method for stochastic
       optimization," in Proc. 3rd International Conference on Learning
       Representations (ICLR), 2015, arXiv:1412.6980.
    """

    _task = "regression"

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predicted ratings in the original units."""
        outputs = self._raw_outputs(X)[:, 0]
        mean = float(self.network_.y_mean.item())
        scale = float(self.network_.y_scale.item())
        return outputs * scale + mean


class TorchMLPClassifier(ClassifierMixin, _TorchPerceptron):
    """PyTorch perceptron for class labels (softmax output, cross-entropy loss).

    Takes the same parameters as :class:`TorchMLPRegressor`.

    References
    ----------
    .. [1] X. Glorot and Y. Bengio, "Understanding the difficulty of training
       deep feedforward neural networks," in Proc. 13th International
       Conference on Artificial Intelligence and Statistics (AISTATS), PMLR
       vol. 9, 2010, pp. 249-256.
    .. [2] D. P. Kingma and J. Ba, "Adam: A method for stochastic
       optimization," in Proc. 3rd International Conference on Learning
       Representations (ICLR), 2015, arXiv:1412.6980.
    """

    _task = "classification"

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Softmax class probabilities, columns in the order of ``classes_``."""
        logits = self._raw_outputs(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Most probable class of every row."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
