from __future__ import annotations

import numpy as np

from .base_layer import LayerInputType, MultiInputBaseLayer


class Add(MultiInputBaseLayer):
    """Layer which adds two or more inputs element-wise.

    Inherits from
    ----------
    MultiInputBaseLayer

    Attributes
    ----------
    added: np.ndarray
        Output of the layer after addition.

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
    >>> from dnn.layers import Add

    >>> ip1 = Input(shape=(3, None)) # Create first input
    >>> ip1.ip = np.array([[1, 2], [3, 4], [5, 6]])

    >>> ip2 = Input(shape=(3, None)) # Create second input
    >>> ip2.ip = np.array([[7, 8], [9, 10], [11, 12]])

    >>> layer = Add(ip=[ip1, ip2])
    >>> layer.forward_step() # Forward step
    array([[ 8, 10],
           [12, 14],
           [16, 18]])
    """

    def __init__(self, ip: list[LayerInputType], name: str = None) -> None:
        """
        Arguments
        ----------
        ip: Inputs to the layer.

        name: Name of the layer. Should be unique in a model.
        When None, a name is automatically generated.

        Raises
        ---------
        ValueError: When all inputs do not have the same shape.
        """
        super().__init__(ip=ip, trainable=False, name=name)

        self._validate_same_shape()

        self.added = None

    def _validate_same_shape(self) -> None:
        """Method to validate that all inputs have the same shape.

        Raises
        ----------
        ValueError: When all inputs do not have the same shape.
        """
        shapes = self.input_shape()

        first_shape = shapes[0]
        if any(shape != first_shape for shape in shapes[1:]):
            msg = f"All inputs must have the same shape. Expected shape: {first_shape}"
            raise ValueError(msg)

    def output(self) -> np.ndarray | None:
        return self.added

    def output_shape(self) -> tuple[int, ...]:
        return self.input_shape()[0]

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        """Method to carry out one step of forward propagation.

        Raises
        ---------
        ValueError: When all inputs do not have the same shape.
        """
        # Need to recheck since the batch_size, which may have been None
        # During __init__() is also available now.
        self._validate_same_shape()

        self.added = np.add.reduce(self.input())

        return self.added

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> tuple[np.ndarray]:
        num_ips = len(self.input())
        # The gradient is copied for each input
        return tuple(grad.copy() for _ in range(num_ips))
