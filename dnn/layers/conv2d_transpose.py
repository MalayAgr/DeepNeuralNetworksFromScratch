from __future__ import annotations

import numpy as np

from .base_layer_conv import BaseConv
from .utils import conv_utils as cutils


class Conv2DTranspose(BaseConv):
    def kernel_shape(self) -> tuple[int, ...]:
        return (self.filters, *self.kernel_size, self.ip_C)

    def output_area(self) -> tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        pH, pW = self.pad_area()
        oH = cutils.deconvolution_output_dim(ipH, self.kernel_H, pH, self.stride_H)
        oW = cutils.deconvolution_output_dim(ipW, self.kernel_W, pW, self.stride_W)

        return oH, oW

    def prepare_input_and_kernel_for_conv(self) -> tuple[np.ndarray, np.ndarray]:
        ip = self.input()
        ip = np.swapaxes(ip, 0, -1).reshape(ip.shape[-1], -1, self.ip_C)
        return ip, cutils.vectorize_kernel(kernel=self.kernels)

    def _padded_shape(self) -> tuple[int, int]:
        oH, oW = self.output_area()
        pH, pW = self.pad_area()
        return oH + 2 * pH, oW + 2 * pW

    def conv_func(self) -> np.ndarray:
        ipH, ipW, m = self.input_shape()[1:]
        shape = (m, self.filters, *self._padded_shape())

        return cutils.transpose_convolve2d(
            X=self._vec_ip,
            weights=self._vec_kernel,
            shape=shape,
            filters=self.filters,
            ip_area=(ipH, ipW),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad_area(),
        )

    def reshape_backprop_gradient(self, grad: np.ndarray) -> np.ndarray:
        grad = cutils.prepare_ip(
            grad,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad_area(),
        )

        grad = np.moveaxis(grad, -1, 0)

        return grad

    def compute_bias_gradient(self, grad: np.ndarray) -> np.ndarray:
        shape = self.biases.shape

        grad = cutils.backprop_bias(
            grad=grad,
            axis=(0, 1),
            reshape=shape[1:],
        )

        grad = grad.reshape(self.filters, -1)
        grad = grad.sum(axis=-1)
        grad = grad.reshape(shape)

        return grad

    def compute_kernel_gradient(self, grad: np.ndarray) -> np.ndarray:
        return cutils.backprop_kernel_conv2d(
            ip=grad,
            grad=self._vec_ip,
            kernel_size=self.kernel_size,
            filters=self.ip_C,
        )

    def compute_vec_ip_gradient(self, grad: np.ndarray) -> np.ndarray:
        # Not required
        ...

    def backprop_inputs(self, grad, *args, **kwargs) -> np.ndarray:
        ipH, ipW = self.input_shape()[1:-1]
        return cutils.convolve2d(
            X=grad, weights=self._vec_kernel, filters=self.ip_C, op_area=(ipH, ipW)
        )
