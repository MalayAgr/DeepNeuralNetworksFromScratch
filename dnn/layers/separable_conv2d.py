from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .activations import Activation
from .base_layer import BaseLayer, LayerInput
from .conv2d import Conv2D
from .depthwise_conv2d import DepthwiseConv2D
from .utils import compute_conv_padding


class SeparableConv2D(BaseLayer):
    def __init__(
        self,
        ip: LayerInput,
        filters: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        activation: Optional[Union[Activation, str]] = None,
        multiplier: int = 1,
        padding: str = "valid",
        depthwise_initializer: str = "he",
        pointwise_initializer: str = "he",
        use_bias: bool = True,
        name: str = None,
    ) -> None:
        self._depthwise = DepthwiseConv2D(
            ip=ip,
            kernel_size=kernel_size,
            multiplier=multiplier,
            stride=stride,
            padding=padding,
            initializer=depthwise_initializer,
            use_bias=False,
        )

        self.use_bias = use_bias

        self._pointwise = Conv2D(
            ip=self._depthwise,
            filters=filters,
            kernel_size=(1, 1),
            activation=activation,
            initializer=pointwise_initializer,
            use_bias=use_bias,
        )

        params = [f"depthwise_{key}" for key in self._depthwise.param_keys]
        params.append("pointwise_kernels")

        if self.use_bias:
            params.append("biases")

        super().__init__(ip=ip, params=params, name=name)

        self._gradients = {}

        self.filters = filters

        self.kernel_size = kernel_size
        self.kernel_H, self.kernel_W = kernel_size

        self.stride = stride
        self.stride_H, self.stride_W = stride

        self.padding = padding
        self.p_H, self.p_W = compute_conv_padding(kernel_size, mode=padding)

    def fans(self) -> Tuple[int, int]:
        return self._pointwise.fans()

    def build(self) -> Any:
        self._depthwise.build()
        self._pointwise.build()
        self.built = True

    def count_params(self) -> int:
        return self._depthwise.count_params() + self._pointwise.count_params()

    def input(self) -> np.ndarray:
        return self._depthwise.input()

    def input_shape(self) -> Tuple:
        return self._depthwise.input_shape()

    def output(self) -> Optional[np.ndarray]:
        return self._pointwise.output()

    def output_shape(self) -> Tuple:
        return self._pointwise.output_shape()

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self._depthwise.forward_step()
        result = self._pointwise.forward_step()

        return result

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        grad = self._pointwise.backprop(grad=grad)
        return self._depthwise.transform_backprop_gradient(grad, *args, **kwargs)

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        self._depthwise.backprop_parameters(grad, *args, **kwargs)

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._depthwise.backprop_inputs(grad, *args, **kwargs)

    ######################################################################
    # Some attributes are defined as properties to resolve the indirection
    # Introduced by the layer being made up of other layers internally.
    # This ensures that the layer seems "flat" to outside users.
    ######################################################################

    @property
    def depthwise_kernels(self) -> np.ndarray:
        return self._depthwise.kernels

    @depthwise_kernels.setter
    def depthwise_kernels(self, kernel: np.ndarray) -> None:
        self.depthwise_kernels = kernel

    @property
    def pointwise_kernels(self) -> np.ndarray:
        return self._pointwise.kernels

    @pointwise_kernels.setter
    def pointwise_kernels(self, kernel: np.ndarray) -> None:
        self.pointwise_kernels = kernel

    @property
    def biases(self) -> Optional[np.ndarray]:
        return self._pointwise.biases

    @biases.setter
    def biases(self, bias) -> None:
        self.biases = bias

    @property
    def gradients(self) -> Dict[str, np.ndarray]:
        grads = self._depthwise.gradients
        self._gradients = {f"depthwise_{key}": grad for key, grad in grads.items()}

        grads = self._pointwise.gradients
        self._gradients["pointwise_kernels"] = grads["kernels"]

        if self.use_bias:
            self._gradients["biases"] = grads["biases"]

        return self._gradients

    @gradients.setter
    def gradients(self, value: Dict[str, np.ndarray]) -> None:
        self._gradients = value

    @property
    def activation(self) -> Activation:
        return self._pointwise.activation

    @activation.setter
    def activation(self, value: Activation):
        self.activation = value
