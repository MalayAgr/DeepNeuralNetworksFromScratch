from typing import Tuple, Union

import numpy as np

from .base_conv import Conv
from .utils import (
    backprop_bias_conv,
    backprop_ip_conv2d,
    backprop_kernel_conv2d,
    convolve2d,
    prepare_ip_for_conv,
    vectorize_kernel_for_conv_nr,
)


class Conv2D(Conv):
    def prepare_input_and_kernel_for_conv(self) -> Tuple[np.ndarray, np.ndarray]:
        ip = prepare_ip_for_conv(
            X=self.input(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.p_H, self.p_W),
        )
        ip = np.moveaxis(ip, -1, 0)
        return ip, vectorize_kernel_for_conv_nr(kernel=self.kernels)

    def conv_func(
        self,
    ) -> np.ndarray:
        return convolve2d(
            X=self._vec_ip,
            weights=self._vec_kernel,
            filters=self.filters,
            op_area=self.output_area(),
        )

    def _reshape_output_gradient(self, grad: np.ndarray) -> np.ndarray:
        return np.swapaxes(grad, 0, -1).reshape(grad.shape[-1], -1, self.filters)

    def _compute_kernel_gradient(self, grad: np.ndarray) -> np.ndarray:
        return backprop_kernel_conv2d(
            ip=self._vec_ip,
            grad=grad,
            kernel_size=self.kernel_size,
            filters=self.filters,
        )

    def _compute_bias_gradient(self, grad: np.ndarray) -> np.ndarray:
        return backprop_bias_conv(grad=grad, axis=(0, 1), reshape=self.biases.shape[1:])

    def _compute_vec_ip_gradient(self, grad: np.ndarray) -> np.ndarray:
        return backprop_ip_conv2d(grad=grad, kernel=self._vec_kernel)
