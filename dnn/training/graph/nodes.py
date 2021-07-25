from abc import ABC, abstractmethod
from typing import Generator, List, Tuple, Union

import numpy as np
from dnn.layers import BaseLayer


class Node(ABC):
    def __init__(self, name: str, source: bool = False) -> None:
        self.name = name
        self.backprop_grad = 0
        self.is_source = source
        self.visited = False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def parents(self) -> Union[List[str], None]:
        return None

    @property
    @abstractmethod
    def trainable_weights(self) -> List[str]:
        """Method to obtain a list of names of trainable weights of the node."""

    @abstractmethod
    def get_trainable_weight_values(self) -> Generator[np.ndarray, None, None]:
        """Method to obtain a generator of the trainable weights of the node."""

    @property
    @abstractmethod
    def gradients(self) -> Generator[np.ndarray, None, None]:
        """Method to obtain the gradient of the loss wrt the trainable weights of the node."""

    @abstractmethod
    def forward(self) -> np.ndarray:
        """
        Method to execute the forward step of the node.
        """

    @abstractmethod
    def forward_output(self) -> np.ndarray:
        """Method to obtain the output of the forward pass through the node."""

    @abstractmethod
    def backprop(self) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """
        Method to execute the backprop step of the node.

        The attribute backprop_grad has the backpropagated gradients for the node.
        """


class LayerNode(Node):
    def __init__(self, layer: BaseLayer, source: bool = False) -> None:
        self.layer = layer
        super().__init__(name=layer.name, source=source)

    @property
    def parents(self) -> Union[List[str], None]:
        ip_layers = self.layer.ip_layer

        if self.is_source:
            return None

        if isinstance(ip_layers, BaseLayer):
            return [ip_layers.name]

        return [layer.name for layer in ip_layers]

    @property
    def trainable_weights(self) -> List[str]:
        if self.layer.trainable is True:
            return self.layer.param_keys
        return []

    def get_trainable_weight_values(self) -> Generator[np.ndarray, None, None]:
        attrs = self.trainable_weights
        return (getattr(self.layer, attr) for attr in attrs)

    @property
    def gradients(self) -> Generator[np.ndarray, None, None]:
        keys = self.trainable_weights
        return (self.layer.gradients[key] for key in keys)

    def forward(self) -> np.ndarray:
        return self.layer.forward_step()

    def forward_output(self) -> np.ndarray:
        return self.layer.output()

    def backprop(self) -> Union[np.ndarray, Tuple[np.ndarray]]:
        return self.layer.backprop_step(self.backprop_grad)
