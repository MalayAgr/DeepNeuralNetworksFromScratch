from __future__ import annotations

from itertools import count

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dnn.input_layer import Input


class BaseLayer(ABC):
    reset: Tuple = None
    str_attrs = ()

    _id = count(0)

    def __init__(
        self,
        ip: Union[Input, BaseLayer],
        *args,
        trainable: bool = True,
        params: Optional[List] = None,
        name: str = None,
        **kwargs,
    ) -> None:
        if ip is not None and not isinstance(ip, (Input, BaseLayer)):
            msg = (
                f"A {self.__class__.__name__} can have only instances "
                "of Input or a subclass of BaseLayer as ip"
            )
            raise AttributeError(msg)

        self.ip_layer = ip

        self.name = self._make_name() if name is None else name

        self.trainable = trainable

        self.is_training = False

        self.param_keys = params

        self.requires_dX = True
        self.gradients = {}

    def __str__(self) -> str:
        attrs = ", ".join(f"{attr}={getattr(self, attr)}" for attr in self.str_attrs)
        return f"{self.__class__.__name__}({attrs})"

    def __repr__(self) -> str:
        return self.__str__()

    def _make_name(self) -> str:
        cls = self.__class__.__name__.lower()
        _id = next(BaseLayer._id)
        base = f"{cls}{_id}_"

        return base + "/".join(
            f"{attr[0]}@{getattr(self, attr)}"
            for attr in self.str_attrs
            if attr != "activation"
        )

    @abstractmethod
    def fans(self) -> Tuple[int, int]:
        """
        Method to obtain the number of input and output units
        """

    def _initializer_variance(self, initializer: str) -> float:
        fan_in, fan_out = self.fans()

        return {
            "he": 2 / fan_in,
            "xavier": 1 / fan_in,
            "xavier_uniform": 6 / (fan_in + fan_out),
        }[initializer]

    def _add_param(self, shape: Tuple, initializer: str) -> np.ndarray:
        """Helper method to initialize a parameter with the given shape and initializer"""
        if initializer == "zeros":
            return np.zeros(shape=shape, dtype=np.float32)
        if initializer == "ones":
            return np.ones(shape=shape, dtype=np.float32)

        variance = self._initializer_variance(initializer)

        return np.random.randn(*shape).astype(np.float32) * np.sqrt(variance)

    def build(self) -> Any:
        """
        Method to build the layer, usually by initializing the parameters
        """
        if self.param_keys is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement build"
            )

    def count_params(self) -> int:
        """
        Method to count the number of trainable parameters in the layer
        """
        if self.param_keys is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement count_params"
            )
        return 0

    def input(self) -> np.ndarray:
        if self.ip_layer is None:
            raise ValueError("No input found")

        ret_val = (
            self.ip_layer.ip
            if isinstance(self.ip_layer, Input)
            else self.ip_layer.output()
        )

        if ret_val is None:
            raise ValueError("No input found")

        return ret_val

    def input_shape(self) -> Tuple:
        if isinstance(self.ip_layer, Input):
            return self.ip_layer.ip_shape
        return self.ip_layer.output_shape()

    @abstractmethod
    def output(self) -> np.ndarray:
        """
        Method to obtain the output of the layer
        """

    @abstractmethod
    def output_shape(self) -> Tuple:
        """
        Method to determine the shape of the output of the layer
        """

    @abstractmethod
    def forward_step(self, *args, **kwargs) -> np.ndarray:
        """
        One step of forward propagation
        """

    @abstractmethod
    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        One step of backpropagation
        """

    def get_param_shapes(self, keys: List) -> Dict:
        """
        Helper method to obtain the shapes of the specified parameters of the layer.
        """
        return {key: getattr(self, key).shape for key in keys}

    def update_params(self, updates: Dict) -> None:
        """
        Helper method to perform one update step on the parameters of the layer.
        """
        for key, update in updates.items():
            new_value = getattr(self, key)
            new_value -= update
            setattr(self, key, new_value)

    def reset_attrs(self) -> None:
        for attr in self.reset:
            if isinstance(attr, tuple):
                setattr(self, attr[0], attr[-1])
                continue
            setattr(self, attr, None)


class MultiInputBaseLayer(BaseLayer):
    def __init__(
        self,
        ip: List[Union[Input, BaseLayer]],
        *args,
        trainable: bool,
        params: Optional[List],
        name: str,
        **kwargs,
    ) -> None:
        if ip is not None:
            if not isinstance(ip, List):
                raise ValueError(f"{self.__class__.__name__} expects a list of inputs.")

            if any(not isinstance(i, (BaseLayer, Input)) for i in ip):
                msg = (
                    f"{self.__class__.__name__} expects all inputs in the list of inputs "
                    "to be instances of Input or other layers."
                )
                raise ValueError(msg)

        super().__init__(
            ip=None, *args, trainable=trainable, params=params, name=name, **kwargs
        )

        self.ip_layer = ip

    def input(self) -> List[np.ndarray]:
        ret_val = []

        for ip in self.ip_layer:
            op = ip.ip if isinstance(ip, Input) else ip.output()

            if op is None:
                raise ValueError(f"No input found in {ip}.")

            ret_val.append(op)

        return ret_val

    def input_shape(self) -> List[Tuple]:
        return [ip.ip_shape if isinstance(ip, Input) else ip.output_shape() for ip in self.ip_layer]


LayerInput = Union[Input, BaseLayer]
