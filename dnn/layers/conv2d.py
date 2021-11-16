from __future__ import annotations

from typing import Tuple

import numpy as np

from .base_conv import Conv
from .utils import conv_utils as cutils


class Conv2D(Conv):
    def prepare_input_and_kernel_for_conv(self) -> Tuple[np.ndarray, np.ndarray]:
        ip = cutils.prepare_ip(
            X=self.input(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.pad_area(),
        )
        ip = np.moveaxis(ip, -1, 0)
        return ip, cutils.vectorize_kernel(kernel=self.kernels)

    def conv_func(
        self,
    ) -> np.ndarray:
        return cutils.convolve2d(
            X=self._vec_ip,
            weights=self._vec_kernel,
            filters=self.filters,
            op_area=self.output_area(),
        )

    def reshape_backprop_gradient(self, grad: np.ndarray) -> np.ndarray:
        return np.swapaxes(grad, 0, -1).reshape(grad.shape[-1], -1, self.filters)

    def compute_kernel_gradient(self, grad: np.ndarray) -> np.ndarray:
        return cutils.backprop_kernel_conv2d(
            ip=self._vec_ip,
            grad=grad,
            kernel_size=self.kernel_size,
            filters=self.filters,
        )

    def compute_bias_gradient(self, grad: np.ndarray) -> np.ndarray:
        return cutils.backprop_bias_conv(
            grad=grad,
            axis=(0, 1),
            reshape=self.biases.shape[1:],
        )

    def compute_vec_ip_gradient(self, grad: np.ndarray) -> np.ndarray:
        return np.matmul(grad, self._vec_kernel.T, dtype=np.float32)
