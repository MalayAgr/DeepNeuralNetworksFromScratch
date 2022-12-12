from __future__ import annotations

from typing import Any

import numpy as np
from dnn.layers.activations import ActivationType

from .base_layer import BaseLayer, LayerInputType
from .utils import add_activation


class Dense(BaseLayer):
    """Fully-connected layer.

    Inherits from
    ----------
    BaseLayer

    Attributes
    ----------
    units: int
        Number of logistic units in the layer.

    activation: Instance of Activation
        Activation that should be applied on the layer.

    initializer: str
        Initializer for the weights of the layer.

    use_bias: bool
        Indicates whether the layer uses a bias or not.

    weights: np.ndarray
        Weights of the layer.

    biases: np.ndarray
        Biases of the layer. The attribute is only present if
        use_bias is True.

    linear: np.ndarray
        Numpy array holding the linear combination weights * input + biases.

    activations: np.ndarray
        Numpy array holding the output of the layer after the activation
        has been applied, i.e. activation(linear).

    Input shape
    ----------
    (x, batch_size), where x is the number of input units.

    Output shape
    ----------
    (units, batch_size).

    Example
    ----------
    >>> import numpy as np
    >>> from dnn import Input
    >>> from dnn.layers import Dense

    >>> ip = Input(shape=(5, None)) # Create input
    >>> ip.ip = np.random.rand(5, 64)

    >>> layer = Dense(ip=ip, units=10, activation="relu")
    >>> layer.build() # Initialize parameters
    >>> layer.forward_step().shape # Forward step
    (10, 64)
    """

    reset = ("linear", "activations")
    str_attrs = ("units", "activation")

    __slots__ = ("activations", "linear", "weights", "biases")

    def __init__(
        self,
        ip: LayerInputType,
        units: int,
        activation: ActivationType = None,
        initializer: str = "he",
        use_bias: bool = True,
        name: str = None,
    ) -> None:
        """
        Arguments
        ----------
        ip: Input to the layer.

        units: Number of logistic units in the layer.

        activation: Activation for the layer. When None, a "linear" activation is used
        i.e. no activation is applied.

        initializer: Initializer for the weights of the layer. Can be one of
        "he", "xavier", "xavier_uniform", "zeros" or "ones". Defaults to "he".

        use_bias: Indicates whether the layer should use a bias. Defaults to True.

        name: Name for the layer. It should be unique for a model.
        When None, a name is automatically generated.
        """
        self.units = units
        self.activation = add_activation(activation)
        self.initializer = initializer

        self.weights = None
        params = ["weights"]

        self.use_bias = use_bias

        if use_bias is True:
            self.biases = None
            params.append("biases")

        self.linear = None
        self.activations = None

        super().__init__(ip=ip, params=params, name=name)

    def fans(self) -> tuple[int, int]:
        fan_in = self.input_shape()[0]
        return fan_in, self.units

    def build(self) -> Any:
        ip_units, units = self.fans()

        shape = (units, ip_units)

        self.weights = self._add_param(shape=shape, initializer=self.initializer)

        if self.use_bias:
            shape = (units, 1)
            self.biases = self._add_param(shape=shape, initializer="zeros")

    def count_params(self) -> int:
        ip_units, units = self.fans()

        total = ip_units * units

        if self.use_bias:
            total += units

        return total

    def output(self) -> np.ndarray | None:
        return self.activations

    def output_shape(self) -> tuple[int, ...]:
        if self.activations is not None:
            return self.activations.shape

        return self.units, None

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.linear = np.matmul(self.weights, self.input(), dtype=np.float32)

        if self.use_bias:
            self.linear += self.biases

        self.activations = self.activation.forward_step(ip=self.linear)

        return self.activations

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        return self.activation.backprop(grad, ip=self.linear)

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        ip = self.input()

        dW = np.matmul(grad, ip.T, dtype=np.float32)

        self.gradients["weights"] = dW

        if self.use_bias:
            self.gradients["biases"] = np.sum(grad, keepdims=True, axis=1)

    def backprop_inputs(self, grad, *args, **kwargs) -> np.ndarray:
        return np.matmul(self.weights.T, grad, dtype=np.float32)
