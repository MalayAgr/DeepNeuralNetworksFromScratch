from __future__ import annotations

from typing import Tuple

import numpy as np

from .base_conv import BaseConv
from .utils import conv_utils as cutils


class Conv2DTranspose(BaseConv):
    def kernel_shape(self) -> Tuple[int, ...]:
        return (self.filters, *self.kernel_size, self.ip_C)

    def output_area(self) -> Tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        pH, pW = self.pad_area()
        oH = cutils.deconvolution_output_dim(ipH, self.kernel_H, pH, self.stride_H)
        oW = cutils.deconvolution_output_dim(ipW, self.kernel_W, pW, self.stride_W)

        return oH, oW

    def prepare_input_and_kernel_for_conv(self) -> Tuple[np.ndarray, np.ndarray]:
        ip = self.input()
        ip = np.swapaxes(ip, 0, -1).reshape(ip.shape[-1], -1, self.ip_C)
        return ip, cutils.vectorize_kernel(kernel=self.kernels)

    def _padded_shape(self) -> Tuple[int, int]:
        oH, oW = self.output_area()
        pH, pW = self.pad_area()
        return oH + 2 * pH, oW + 2 * pW

    def conv_func(self) -> np.ndarray:
        ipH, ipW, m = self.input_shape()[1:]
        post_pad_H, post_pad_W = self._padded_shape()
        shape = (m, self.filters, post_pad_H, post_pad_W)

        areas = np.matmul(self._vec_ip, self._vec_kernel.T, dtype=np.float32)

        return cutils.accumulate_dX_conv(
            grad_shape=shape,
            output_size=(ipH, ipW),
            vec_ip_grad=areas,
            stride=self.stride,
            kernel_size=self.kernel_size,
            padding=self.pad_area(),
            reshape=(-1, self.filters, self.kernel_H, self.kernel_W),
        )

    def compute_bias_gradient(self, grad: np.ndarray) -> np.ndarray:
        pass

    def compute_kernel_gradient(self, grad: np.ndarray) -> np.ndarray:
        pass

    def compute_vec_ip_gradient(self, grad: np.ndarray) -> np.ndarray:
        pass
