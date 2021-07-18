from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np

from .base_layer import BaseLayer, LayerInput


class Activation(BaseLayer):
    name = None

    def __init__(self, *args, ip: Optional[LayerInput] = None, **kwargs) -> None:
        super().__init__(ip=ip, trainable=False)

        self.activations = None

    @classmethod
    def get_activation_classes(cls) -> dict:
        result = {}

        for sub_cls in cls.__subclasses__():
            name = sub_cls.name
            result.update(sub_cls.get_activation_classes())
            if name is not None and name not in result:
                result.update({name: sub_cls})
        return result

    @abstractmethod
    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        """
        The formula used to calculate the activations.
        Subclasses classes must implement this.
        If the activation function is called g with input z,
        this should return g(z).
        Arguments:
            ip (Numpy-array): The input z for the function.
        Returns:
            A Numpy-array with the calculated activations, g(z).
        """

    @abstractmethod
    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        The formula used to calculate the derivatives.
        Subclasses classes must implement this.
        If the activation function is called g with input z,
        this should return g'(z).
        Arguments:
            ip (Numpy-array): The input z with respect to which derivatives
                need to be calculated.
        Example:
            For sigmoid, the derivative is sigmoid(z) * (1 - sigmoid(z)).
            This should be implemented as:
                def derivative_func(self):
                    sigmoid = self.activation_func(ip)
                    return sigmoid * (1 - sigmoid)
        Returns:
            A Numpy-array with the calculated derivatives, g'(z).
        """

    def compute_activations(self, ip: Optional[np.ndarray] = None) -> np.ndarray:
        if ip is None:
            ip = self.input()

        return self.activation_func(ip).astype(np.float32)

    def compute_derivatives(self, ip: Optional[np.ndarray] = None) -> np.ndarray:
        if ip is None:
            ip = self.input()
            return self.derivative_func(ip, activations=self.activations).astype(
                np.float32
            )
        return self.derivative_func(ip).astype(np.float32)

    def fans(self) -> Tuple[int, int]:
        ip_layer_fans = self.ip_layer.fans()
        return ip_layer_fans[-1], ip_layer_fans[-1]

    def output(self) -> np.ndarray:
        return self.activations

    def output_shape(self):
        return self.input_shape()

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        ip = kwargs.pop("ip", None)

        self.activations = self.compute_activations(ip)

        return self.activations

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        ip = kwargs.pop("ip", None)

        dA = dA * self.compute_derivatives(ip)

        self.activations = None

        return dA


class Sigmoid(Activation):
    name = "sigmoid"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        z = 1.0
        z /= np.exp(-ip) + 1
        return z

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        return activations * (1 - activations)


class Softmax(Activation):
    name = "softmax"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        z = ip - np.max(ip, axis=0, keepdims=True)
        z = np.exp(z)
        z /= np.sum(z, axis=0, keepdims=True)
        return z

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        units = activations.shape[0]

        grads = np.eye(units, dtype=np.float32) - activations.T[..., None]
        grads *= activations[:, None, :].T

        return np.moveaxis(grads, 0, -1)

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        dA = super().backprop_step(dA, *args, **kwargs)

        dA = np.sum(dA, axis=1)

        return dA


class Tanh(Activation):
    name = "tanh"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.tanh(ip)

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        activations **= 2
        return 1 - activations


class ReLU(Activation):
    name = "relu"

    def activation_func(self, ip):
        return np.maximum(0, ip)

    def derivative_func(self, ip, activations=None):
        return np.where(ip > 0, 1.0, 0)


class LeakyReLU(Activation):
    name = "lrelu"
    default_alpha = 0.01

    def __init__(self, *args, ip: Optional[LayerInput] = None, **kwargs) -> None:
        alpha = kwargs.pop("alpha", None)
        if alpha is None:
            alpha = self.default_alpha
        self.alpha = alpha

        super().__init__(ip=ip, *args, **kwargs)

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.where(ip > 0, ip, self.alpha * ip)

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.where(ip > 0, 1.0, self.alpha)


class ELU(LeakyReLU):
    name = "elu"
    default_alpha = 1.0

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.where(ip > 0, ip, self.alpha * (np.exp(ip) - 1))

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.where(ip > 0, 1.0, self.alpha * np.exp(ip))
