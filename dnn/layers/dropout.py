from __future__ import annotations

import numpy as np

from .base_layer import BaseLayer, LayerInputType


class Dropout(BaseLayer):
    """Inverted dropout layer.

    Inherits from
    ----------
    BaseLayer

    Attributes
    ----------
    keep_prob: float
        Probability of retaining an input unit.

    dropout_mask: np.ndarray
        Mask used to select input units that should be dropped.

    dropped: np.ndarray
        Output of the layer.

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
    >>> from dnn.layers import Dropout

    >>> ip = Input(shape=(5, None)) # Create input
    >>> ip.ip = np.random.rand(5, 64)

    >>> layer = Dropout(ip=ip, keep_prob=0.5)
    >>> layer.forward_step().shape # Forward step
    (5, 64)
    """

    reset = ("dropped", "dropout_mask")
    str_attrs = ("keep_prob",)

    def __init__(
        self, ip: LayerInputType, keep_prob: float = 0.5, name: str = None
    ) -> None:
        """
        Arguments
        ----------
        ip: Input to the layer.

        keep_prob: Probability of retaining an input unit. Defaults to 0.5.

        name: Name of the layer. Should be unique in a model.
        When None, a name is automatically generated.

        Raises
        ----------
        ValueError: When keep_prob is not between 0 (exclusive) and 1 (inclusive).
        """
        if not 0 < keep_prob <= 1:
            raise ValueError("keep_prob should be in the interval (0, 1].")

        self.keep_prob = keep_prob

        super().__init__(ip=ip, trainable=False, name=name)

        self.dropped = None
        self.dropout_mask = None

    def output(self) -> np.ndarray | None:
        return self.dropped

    def output_shape(self) -> tuple[int, ...]:
        return self.input_shape()

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        ip = self.input()

        self.dropout_mask = (
            np.random.rand(*ip.shape).astype(np.float32) < self.keep_prob
        )

        self.dropped = ip * self.dropout_mask
        self.dropped /= self.keep_prob

        return self.dropped

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        grad *= self.dropout_mask
        grad /= self.keep_prob

        return grad
