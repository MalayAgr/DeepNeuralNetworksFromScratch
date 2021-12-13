from __future__ import annotations

from typing import Any

import numpy as np

from .base_layer import BaseLayer, LayerInputType


class BatchNorm(BaseLayer):
    """Batch normalization layer.

    Inherits from
    ----------
    BaseLayer

    Attributes
    ----------
    axis: int
        Axis along which batch normalization should be performed.

    momentum: float
        Momentum for the moving averages of the mean and standard deviation.

    epsilon: float
        Epsilon to avoid divide by zero.

    gamma: np.ndarray
        Standard deviation scaling parameter.

    beta: np.ndarray
        Meaning scaling parameter.

    std: np.ndarray
        Standard deviation of the input.

    norm: np.ndarray
        Norm of the input, i.e. (input - mean) / std.

    scaled_norm: np.ndarray
        Norm after it is scaled by gamma and beta, i.e. gamma * norm + beta.

    mean_mva: np.ndarray
        Moving average of mean.

    std_mva: np.ndarray
        Moving average of standard deviation.

    Methods
    ----------
    x_dim() -> int
        Returns the dimension of the input along the axis attribute.

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> import numpy as np
    >>> from dnn import Input
    >>> from dnn.layers import BatchNorm

    >>> ip = Input(shape=(3, 6, 6, None)) # Create input
    >>> ip.ip = np.random.rand(3, 6, 6, 64)

    >>> layer = BatchNorm(ip=ip)
    >>> layer.forward_step().shape # Forward step
    (3, 6, 6, 64)
    """

    reset = ("std", "norm", "scaled_norm")

    __slots__ = ("gamma", "beta", "norm", "scaled_norm", "mean_mva", "std_mva")

    def __init__(
        self,
        ip: LayerInputType,
        axis: int = 0,
        momentum: float = 0.5,
        epsilon: float = 1e-7,
        name: str = None,
    ) -> None:
        """
        Arguments
        ----------
        ip: Input to the layer.

        axis: Axis along which normalization should be carried out.
        Defaults to 0.

        momentum: Momentum for the moving averages. Defaults to 0.5.

        epsilon: Small value to prevent divide by zero. Defaults to 1e-7.

        name: Name of the layer. Should be unique in a model.
        When None, a name is automatically generated.

        Raises
        ----------
        ValueError: When axis is out of bounds or it is negative but not -1.
        """
        self.gamma = None
        self.beta = None

        params = ["gamma", "beta"]

        super().__init__(ip=ip, params=params, name=name)

        ndims = len(self.input_shape())

        if axis >= ndims:
            msg = (
                "axis is out of bounds for the layer. "
                f"Should be -1 or between 0 and {ndims - 1} but got {axis} instead."
            )
            raise ValueError(msg)

        if axis < 0 and axis != -1:
            raise ValueError("-1 is the only negative value allowed for axis.")

        self.axis = axis

        self._ndims = ndims

        # Resolve axis = -1 to refer to the last dimension
        self._axis = ndims - 1 if axis == -1 else axis

        # Get all axes which are not equal to axis
        self._axes = tuple(ax for ax in range(ndims) if ax != self._axis)

        self.momentum = momentum
        self.epsilon = epsilon

        self.std = None
        self.norm = None
        self.scaled_norm = None

        self.mean_mva = None
        self.std_mva = None

    def fans(self) -> tuple[int, int]:
        if not isinstance(self.ip_layer, BaseLayer):
            raise TypeError("fans() can only be used when the input is another layer.")

        _, ip_fan_out = self.ip_layer.fans()

        return ip_fan_out, ip_fan_out

    def x_dim(self) -> int:
        """Method to obtain the dimension of the input along the axis attribute."""
        return self.input_shape()[self.axis]

    def build(self) -> Any:
        x_dim = self.x_dim()

        # The position of x_dim depends on the value of axis
        # Eg - If axis = 0 and the input is 4D, shape should be (x_dim, 1, 1, 1)
        # But if axis = -1, the shape should be (1, 1, 1, x_dim)
        shape = [1] * self._ndims
        shape[self._axis] = x_dim
        shape = tuple(shape)

        self.gamma = self._add_param(shape=shape, initializer="ones")

        self.beta = self._add_param(shape=shape, initializer="zeros")

        self.mean_mva = self._add_param(shape=shape, initializer="zeros")

        self.std_mva = self._add_param(shape=shape, initializer="ones")

    def count_params(self) -> int:
        return 2 * self.x_dim()

    def output(self) -> np.ndarray | None:
        return self.scaled_norm

    def output_shape(self) -> tuple[int, ...]:
        return self.input_shape()

    def _update_mva(self, mean: np.ndarray, std: np.ndarray) -> None:
        mom, one_minus_mom = self.momentum, 1 - self.momentum

        self.mean_mva *= mom
        self.mean_mva += one_minus_mom * mean

        self.std_mva *= mom
        self.std_mva += one_minus_mom * std

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        ip = self.input()

        # If in training mode, the mean and std are calculated for the current batch
        # and the moving averages updated
        if self.training:
            mean = ip.mean(axis=self._axes, keepdims=True)
            std = np.sqrt(ip.var(axis=self._axes, keepdims=True) + self.epsilon)

            self._update_mva(mean, std)

            self.std = std
        # Otherwise, the moving averages act as the mean and std
        else:
            mean, std = self.mean_mva, self.std_mva

        self.norm = ip - mean
        self.norm /= std

        self.scaled_norm = self.gamma * self.norm
        self.scaled_norm += self.beta

        return self.scaled_norm

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        self.gradients = {
            "gamma": np.sum(grad * self.norm, axis=self._axes, keepdims=True),
            "beta": np.sum(grad, axis=self._axes, keepdims=True),
        }

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        grad *= self.gamma

        # Calculate share of the mean in the gradient
        mean_share = grad.sum(axis=self._axes, keepdims=True)

        # Calculate share of the variance in the gradient
        var_share = self.norm * np.sum(grad * self.norm, axis=self._axes, keepdims=True)

        # Since mean and std are calculated across all dimensions except axis,
        # The gradient should be scaled by the product of all dimensions except the axis
        scale = grad.size / self.x_dim()

        grad = scale * grad
        grad -= mean_share
        grad -= var_share
        grad /= self.std
        grad /= scale

        return grad
