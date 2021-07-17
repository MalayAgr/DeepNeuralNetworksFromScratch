from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from dnn.input_layer import Input


class BaseLayer(ABC):
    reset = None
    str_attrs = tuple()

    def __init__(
        self,
        ip: Union[Input, BaseLayer],
        *args,
        trainable: bool = True,
        params: Optional[list] = None,
        **kwargs,
    ) -> None:
        if ip is not None and not isinstance(ip, (Input, BaseLayer)):
            msg = (
                f"A {self.__class__.__name__} can have only instances "
                "of Input or a subclass of BaseLayer as ip"
            )
            raise AttributeError(msg)

        self.ip_layer = ip

        self.trainable = trainable

        self.is_training = False

        if params is not None:
            self.param_map = self._add_params(params)

        self.requires_dX = True
        self.gradients = {}

        self._add_extra_attrs(kwargs)

    def __str__(self) -> str:
        attrs = ", ".join(f"{attr}={getattr(self, attr)}" for attr in self.str_attrs)
        return f"{self.__class__.__name__}({attrs})"

    def _add_params(self, params: list) -> dict:
        for param in params:
            self.__setattr__(param, None)

        return {param: param for param in params}

    def _add_extra_attrs(self, attrs: dict) -> None:
        for attr, value in attrs.items():
            self.__setattr__(attr, value)

    @abstractmethod
    def fans(self) -> tuple[int, int]:
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

    def count_params(self) -> int:
        """
        Method to count the number of trainable parameters in the layer
        """
        if not hasattr(self, "param_map"):
            return 0
        raise NotImplementedError(
            f"{self.__class__.__name__} instances need to implement count_params"
        )

    def build(self) -> Any:
        """
        Method to build the layer, usually by initializing the parameters
        """
        if hasattr(self, "param_map"):
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement build"
            )

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

    def input_shape(self) -> tuple:
        if isinstance(self.ip_layer, Input):
            return self.ip_layer.ip_shape
        return self.ip_layer.output_shape()

    @abstractmethod
    def output(self) -> np.ndarray:
        """
        Method to obtain the output of the layer
        """

    @abstractmethod
    def output_shape(self) -> tuple:
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

    def reset_attrs(self) -> None:
        for attr in self.reset:
            if isinstance(attr, tuple):
                setattr(self, attr[0], attr[-1])
                continue
            setattr(self, attr, None)
