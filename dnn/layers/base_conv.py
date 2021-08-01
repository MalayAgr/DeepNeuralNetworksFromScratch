from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .activations import Activation
from .base_layer import BaseLayer, LayerInput

from .utils import (
    accumulate_dX_conv,
    add_activation,
    compute_conv_output_dim,
    compute_conv_padding,
)


class Conv(BaseLayer):
    reset = (
        "convolutions",
        "activations",
        "_vec_ip",
        "_vec_kernel",
    )

    str_attrs = ("filters", "kernel_size", "stride", "padding", "activation")

    __slots__ = (
        "convolutions",
        "activations",
        "_vec_ip",
        "_vec_kernel",
        "kernels",
        "biases",
    )

    def __init__(
        self,
        ip: LayerInput,
        filters: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        activation: Optional[Union[Activation, str]] = None,
        padding: str = "valid",
        initializer: str = "he",
        use_bias: bool = True,
        name: str = None,
    ) -> None:
        self.filters = filters

        self.kernel_size = kernel_size
        self.kernel_H, self.kernel_W = kernel_size

        self.stride = stride
        self.stride_H, self.stride_W = stride

        self.padding = padding
        self.p_H, self.p_W = compute_conv_padding(kernel_size, mode=padding)

        self.initializer = initializer

        self.activation = add_activation(activation)

        self.kernels = None
        params = ["kernels"]

        self.use_bias = use_bias
        if use_bias:
            self.biases = None
            params.append("biases")

        super().__init__(ip=ip, params=params, name=name)

        self.ip_C = self.input_shape()[0]

        self.convolutions = None
        self.activations = None

        self._vec_ip = None
        self._vec_kernel = None

    def fans(self) -> Tuple[int, int]:
        receptive_field_size = np.prod(self.kernel_size)
        fan_in = self.ip_C * receptive_field_size
        return fan_in, receptive_field_size * self.filters

    def build(self) -> Any:
        shape = (self.ip_C, *self.kernel_size, self.filters)

        self.kernels = self._add_param(shape=shape, initializer=self.initializer)

        if self.use_bias:
            shape = (self.filters, 1, 1, 1)
            self.biases = self._add_param(shape=shape, initializer="zeros")

    def count_params(self) -> int:
        total = np.prod(self.kernels.shape)

        if self.use_bias:
            return total + self.biases.shape[0]

        return total

    def output(self) -> Optional[np.ndarray]:
        return self.activations

    def output_area(self) -> Tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        oH = compute_conv_output_dim(ipH, self.kernel_H, self.p_H, self.stride_H)
        oW = compute_conv_output_dim(ipW, self.kernel_W, self.p_W, self.stride_W)

        return oH, oW

    def output_shape(self) -> Tuple:
        if self.activations is not None:
            return self.activations.shape

        oH, oW = self.output_area()

        return self.filters, oH, oW, None

    def padded_shape(self) -> Tuple[int, int]:
        ipH, ipW = self.input_shape()[1:-1]
        return ipH + 2 * self.p_H, ipW + 2 * self.p_W

    @abstractmethod
    def conv_func(
        self,
    ) -> Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Method to compute the convolutional output of the layer."""
        pass

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.convolutions, self._vec_ip, self._vec_kernel = self.conv_func()

        if self.use_bias:
            self.convolutions += self.biases

        self.activations = self.activation.forward_step(ip=self.convolutions)

        return self.activations

    @abstractmethod
    def _reshape_dZ(self, dZ: np.ndarray) -> np.ndarray:
        """Method to reshape the gradient of loss wrt convolutional output."""
        pass

    @abstractmethod
    def _compute_dW(self, dZ: np.ndarray) -> np.ndarray:
        """Method to compute the gradient of the loss wrt kernel."""
        pass

    @abstractmethod
    def _compute_dB(self, dZ: np.ndarray) -> np.ndarray:
        """Method to compute the gradient of the loss wrt biases."""
        pass

    @abstractmethod
    def _compute_dVec_Ip(self, dZ: np.ndarray) -> np.ndarray:
        """Method to compute the derivative of loss wrt to the vectorized input."""
        pass

    def _target_dX_shape(self) -> Tuple:
        """Method to obtain the shape of the derivative of loss wrt to the input of the layer."""
        post_pad_H, post_pad_W = self.padded_shape()
        m = self.input().shape[-1]
        return m, self.ip_C, post_pad_H, post_pad_W

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        dA = self.activation._backprop_step(grad, ip=self.convolutions)

        return self._reshape_dZ(dA)

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        dW = self._compute_dW(grad)

        self.gradients["kernels"] = dW

        if self.use_bias:
            self.gradients["biases"] = self._compute_dB(grad)

    def backprop_inputs(self, grad, *args, **kwargs) -> np.ndarray:
        dX_shape = self._target_dX_shape()
        dIp = self._compute_dVec_Ip(grad)

        return accumulate_dX_conv(
            dX_shape=dX_shape,
            output_size=self.output_area(),
            dIp=dIp,
            stride=self.stride,
            kernel_size=self.kernel_size,
            reshape=(-1, self.ip_C, self.kernel_H, self.kernel_W),
            padding=(self.p_H, self.p_W),
        )
