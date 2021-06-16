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

    def __init__(self, shape, *args, **kwargs):
        """
        Initializes an Input instance with the given input shape.

        Args:
            shape (tuple): Input shape to be used.
            *args, **kwargs: Provided for extensibility.
        """
        self.ip_shape = shape
        self._ip = None

    @property
    def ip(self):
        """
        The actual input to be used in the layer.
        """
        return self._ip

    @ip.setter
    def ip(self, X):
        """
        Setter for the ip property.

        Args:
            X (Numpy-array): Value of the property.

        Raises:
            AttributeError when ip_shape and shape of X do not match.
        """
        # Make sure the supplied input matches the expected shape
        if X.shape != self.ip_shape:
            raise AttributeError("The input does not have the expected shape")
        self._ip = X
