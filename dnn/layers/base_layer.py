from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from dnn.input_layer import Input


class BaseLayer(ABC):
    reset = ()
    str_attrs = ()

    _id = count(0)

    __slots__ = ("__dict__", "param_keys", "gradients")

    def __init__(
        self,
        ip: Union[Input, BaseLayer, None],
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

        self._built = False

        self.param_keys = params

        self.requires_dX = True
        self.gradients = {}

    def __str__(self) -> str:
        attrs = ", ".join(f"{attr}={getattr(self, attr)}" for attr in self.str_attrs)
        return f"{self.__class__.__name__}({attrs})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, value: bool) -> None:
        self._built = value

    def _make_name(self) -> str:
        cls = self.__class__.__name__.lower()
        _id = next(BaseLayer._id)
        base = f"{cls}{_id}_"

        return base + "/".join(
            f"{attr[0]}@{getattr(self, attr)}"
            for attr in self.str_attrs
            if attr != "activation"
        )

    def fans(self) -> Tuple[int, int]:
        """
        Method to obtain the number of input and output units
        """
        if self.trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement fans"
            )

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
        if self.trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement build"
            )

    def count_params(self) -> int:
        """
        Method to count the number of trainable parameters in the layer
        """
        if self.trainable:
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
            raise ValueError("No input found.")

        return ret_val

    def input_shape(self) -> Tuple:
        if self.ip_layer is None:
            raise ValueError("No input found.")

        if isinstance(self.ip_layer, Input):
            return self.ip_layer.shape

        return self.ip_layer.output_shape()

    @abstractmethod
    def output(self) -> Optional[np.ndarray]:
        """
        Method to obtain the output of the layer
        """

    @abstractmethod
    def output_shape(self) -> Tuple:
        """
        Method to determine the shape of the output of the layer
        """

    def _forward_step(self, *args, **kwargs) -> np.ndarray:
        if not self.built:
            self.build()
            self.built = True
        return self.forward_step(*args, **kwargs)

    @abstractmethod
    def forward_step(self, *args, **kwargs) -> np.ndarray:
        """
        One step of forward propagation
        """

    def _backprop_step(
        self, grad: np.ndarray, *args, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray], None]:

        grad = self.transform_backprop_gradient(grad, *args, **kwargs)

        if self.trainable:
            self.backprop_parameters(grad, *args, **kwargs)

        ret_val = (
            self.backprop_inputs(grad, *args, **kwargs) if self.requires_dX else None
        )

        # Clear memory by resetting unnecessary attributes.
        self._reset_attrs()

        return ret_val

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """
        Apply transformations on the backpropagated gradient so that
        it is appropriate for use by the layer.

        By default, it returns the gradient as is. Override the method
        to apply other transformations like reshape, transpose, activations, etc.
        """
        return grad

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        """
        Method to compute the gradient of loss wrt the layer's parameters.

        Trainable layers must implement this method.

        The implementation should store the derivatives in the 'gradients'
        attribute of the layer.
        """
        if self.trainable:
            raise NotImplementedError(
                "Trainable layers must implement backprop_parameters"
            )

    @abstractmethod
    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Method to compute the derivative of loss wrt the layer's input.

        It should return a single Numpy array.
        """

    def _reset_attrs(self) -> None:
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
        params: Optional[List] = None,
        name: str,
        **kwargs,
    ) -> None:
        if ip is not None:
            self._validate_ip(ip)

        super().__init__(
            ip=None, *args, trainable=trainable, params=params, name=name, **kwargs
        )

        self.ip_layer = ip

    def _validate_ip(self, ip):
        if not isinstance(ip, List):
            raise ValueError(f"{self.__class__.__name__} expects a list of inputs.")

        if any(not isinstance(i, (BaseLayer, Input)) for i in ip):
            msg = (
                f"{self.__class__.__name__} expects all inputs in the list of inputs "
                "to be instances of Input or other layers."
            )
            raise ValueError(msg)

    def input(self) -> List[np.ndarray]:
        ret_val = []

        for ip in self.ip_layer:
            op = ip.ip if isinstance(ip, Input) else ip.output()

            if op is None:
                raise ValueError(f"No input found in {ip}.")

            ret_val.append(op)

        return ret_val

    def input_shape(self) -> List[Tuple]:
        return [
            ip.shape if isinstance(ip, Input) else ip.output_shape()
            for ip in self.ip_layer
        ]

    @abstractmethod
    def output(self) -> Union[np.ndarray, List[np.ndarray], None]:
        """Method to obtain the output(s) of the layer."""

    @abstractmethod
    def output_shape(self) -> Union[Tuple, List[Tuple]]:
        """
        Method to determine the shape of the output(s) of the layer
        """

    @abstractmethod
    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray]:
        """
        Method to compute the derivative of loss wrt the layer's inputs.

        It should return a tuple with as many Numpy arrays as the number of inputs.
        """


LayerInput = Union[Input, BaseLayer]
