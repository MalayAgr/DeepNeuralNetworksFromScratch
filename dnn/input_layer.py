from __future__ import annotations
from typing import Optional, Tuple

import numpy as np


class Input:
    """
    Class to represent the input layer of a neural network.

    In a neural network, the input cannot be determined until previous
    Layers have done their operation. It also doesn't make sense to lock
    A neural network with particular architecture with a single input.
    This class allows the same architecture to be used with different
    Inputs and allows the model to conveniently handle the inherently
    lazy nature of neural networks.

    Attributes:
        ip_shape (tuple): Expected shape of the input.
        ip (Numpy-array): Property which can be set to the
            Actual input that the layer will provide to its users.
    """

    def __init__(self, shape: Tuple, *args, **kwargs) -> None:
        """
        Initializes an Input instance with the given input shape.

        Args:
            shape (tuple): Input shape to be used.
            *args, **kwargs: Provided for extensibility.
        """
        if shape[-1] is not None:
            raise AttributeError("The last dimension should be set to None")
        self._shape = shape
        self._ip: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(ip_shape={self.ip_shape})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def shape(self) -> Tuple:
        """
        The shape of the input of the layer.
        """
        if self.ip is not None:
            return self.ip.shape
        return self._shape

    @property
    def ip(self) -> Optional[np.ndarray]:
        """
        The actual input to be used in the layer.
        """
        return self._ip

    @ip.setter
    def ip(self, X: np.ndarray) -> None:
        """
        Setter for the ip property.

        Args:
            X (Numpy-array): Value of the property.

        Raises:
            AttributeError when ip_shape and shape of X do not match.
        """
        # Make sure the supplied input matches the expected shape
        if X.shape[:-1] != self.ip_shape[:-1]:
            raise AttributeError("The input does not have the expected shape")
        self._ip = X
