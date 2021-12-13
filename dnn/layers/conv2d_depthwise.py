from __future__ import annotations

from typing import Any

import numpy as np

from .activations import ActivationType
from .base_layer import LayerInputType
from .conv2d import Conv2D
from .utils import conv_utils as cutils


class DepthwiseConv2D(Conv2D):
    str_attrs = Conv2D.str_attrs[1:]

    def __init__(
        self,
        ip: LayerInputType,
        *args,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        activation: ActivationType = None,
        multiplier: int = 1,
        padding: str = "valid",
        initializer: str = "he",
        use_bias: bool = True,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(
            ip=ip,
            filters=multiplier,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            padding=padding,
            initializer=initializer,
            use_bias=use_bias,
            name=name,
        )

        self.multiplier = multiplier

    def fans(self) -> tuple[int, int]:
        fan_in, fan_out = super().fans()

        return fan_in, fan_out * self.ip_C

    def build(self) -> Any:
        super().build()

        if self.use_bias:
            remaining = (self.multiplier * self.ip_C) - self.biases.shape[0]
            extra = self._add_param(shape=(remaining, 1, 1, 1), initializer="zeros")
            self.biases = np.concatenate((self.biases, extra))

    def output_shape(self) -> tuple[int, ...]:
        shape = super().output_shape()

        if self.activations is not None:
            return shape

        c, h, w, m = shape

        return c * self.ip_C, h, w, m

    def prepare_input_and_kernel_for_conv(self) -> tuple[np.ndarray, np.ndarray]:
        ip = cutils.prepare_ip(
            X=self.input(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad_area(),
            vec_reshape=(self.ip_C, self.kernel_H * self.kernel_W, -1),
        )
        ip = ip.transpose(-1, 1, 0, 2)

        shape = (self.ip_C, -1, self.multiplier)
        return ip, cutils.vectorize_kernel(self.kernels, reshape=shape)

    def conv_func(self) -> np.ndarray:
        return cutils.depthwise_convolve2d(
            X=self._vec_ip,
            weights=self._vec_kernel,
            multiplier=self.multiplier,
            ip_C=self.ip_C,
            op_area=self.output_area(),
        )

    def reshape_backprop_gradient(self, grad: np.ndarray) -> np.ndarray:
        shape = (grad.shape[-1], -1, self.ip_C, self.multiplier)

        grad = np.swapaxes(grad, -1, 0).reshape(*shape)

        grad = np.moveaxis(grad, 1, 2)

        return grad

    def compute_kernel_gradient(self, grad: np.ndarray) -> np.ndarray:
        return cutils.backprop_kernel_depthwise_conv2d(
            ip=self._vec_ip,
            grad=grad,
            kernel_size=self.kernel_size,
            multiplier=self.multiplier,
        )

    def compute_bias_gradient(self, grad: np.ndarray) -> np.ndarray:
        return cutils.backprop_bias(
            grad=grad, axis=(0, 2), reshape=self.biases.shape[1:]
        )

    def compute_vec_ip_gradient(self, grad: np.ndarray) -> np.ndarray:
        kernel = np.swapaxes(self._vec_kernel, -1, -2)
        vec_ip_grad = np.matmul(grad, kernel, dtype=np.float32)
        return np.moveaxis(vec_ip_grad, 2, 1)
