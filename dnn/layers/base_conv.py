from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

from .activations import ActivationType
from .base_layer import BaseLayer, LayerInput
from .utils import add_activation
from .utils import conv_utils as cutils


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
        activation: ActivationType = None,
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

        self._vec_ip: Optional[np.ndarray] = None
        self._vec_kernel: Optional[np.ndarray] = None

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

        self.built = True

    def count_params(self) -> int:
        total = self.ip_C * self.kernel_H * self.kernel_W * self.filters

        if self.use_bias:
            total += self.filters

        return total

    def output(self) -> Optional[np.ndarray]:
        return self.activations

    def pad_area(self) -> Tuple[int, int]:
        return cutils.compute_conv_padding(self.kernel_size, mode=self.padding)

    def output_area(self) -> Tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        pH, pW = self.pad_area()
        oH = cutils.compute_conv_output_dim(ipH, self.kernel_H, pH, self.stride_H)
        oW = cutils.compute_conv_output_dim(ipW, self.kernel_W, pW, self.stride_W)

        return oH, oW

    def output_shape(self) -> Tuple:
        if self.activations is not None:
            return self.activations.shape

        oH, oW = self.output_area()

        return self.filters, oH, oW, None

    def padded_shape(self) -> Tuple[int, int]:
        ipH, ipW = self.input_shape()[1:-1]
        pH, pW = self.pad_area()
        return ipH + 2 * pH, ipW + 2 * pW

    def prepare_input_and_kernel_for_conv(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.input(), self.kernels

    @abstractmethod
    def conv_func(self) -> np.ndarray:
        """Method to compute the convolutional output of the layer."""

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self._vec_ip, self._vec_kernel = self.prepare_input_and_kernel_for_conv()
        self.convolutions = self.conv_func()

        if self.use_bias:
            self.convolutions += self.biases

        self.activations = self.activation.forward_step(ip=self.convolutions)

        return self.activations

    def reshape_backprop_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Method to reshape the gradient of loss wrt convolutional output."""
        return grad

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        grad = self.activation.backprop(grad, ip=self.convolutions)
        return self.reshape_backprop_gradient(grad)

    @abstractmethod
    def compute_kernel_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Method to compute the gradient of the loss wrt kernel."""

    @abstractmethod
    def compute_bias_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Method to compute the gradient of the loss wrt biases."""

    @abstractmethod
    def compute_vec_ip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Method to compute the derivative of loss wrt to the vectorized input."""

    def get_input_gradient_shape(self) -> Tuple:
        """Method to obtain the shape of the derivative of loss wrt to the input of the layer."""
        post_pad_H, post_pad_W = self.padded_shape()
        m = self.input().shape[-1]
        return m, self.ip_C, post_pad_H, post_pad_W

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        self.gradients["kernels"] = self.compute_kernel_gradient(grad)

        if self.use_bias:
            self.gradients["biases"] = self.compute_bias_gradient(grad)

    def backprop_inputs(self, grad, *args, **kwargs) -> np.ndarray:
        ip_gradient_shape = self.get_input_gradient_shape()
        vec_ip_grad = self.compute_vec_ip_gradient(grad)

        return cutils.accumulate_dX_conv(
            dX_shape=ip_gradient_shape,
            output_size=self.output_area(),
            dIp=vec_ip_grad,
            stride=self.stride,
            kernel_size=self.kernel_size,
            reshape=(-1, self.ip_C, self.kernel_H, self.kernel_W),
            padding=self.pad_area(),
        )
