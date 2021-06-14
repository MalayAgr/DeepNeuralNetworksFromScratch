from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    name = None

    def __init__(self, ip=None, *args, **kwargs):
        self.ip = ip
        self.activations = None
        self.derivatives = None

    @classmethod
    def _get_activation_classes(cls):
        result = {}

        for sub_cls in cls.__subclasses__():
            name = sub_cls.name
            result.update(sub_cls._get_activation_classes())
            if name is not None and name not in result:
                result.update({name: sub_cls})
        return result

    @abstractmethod
    def activation_func(self, ip):
        pass

    @abstractmethod
    def derivative_func(self, ip):
        pass

    def _get_activation_ip(self, ip, message):
        if ip is None and self.ip is None:
            raise AttributeError(message)

        if ip is None:
            return self.ip
        return ip

    def _get_derivative_ip(self, ip, message):
        if ip is None and self.activations is None:
            raise AttributeError(message)

        if ip is None:
            return self.activations
        return self.activation_func(ip)

    def calculate_activations(self, ip=None):
        z = self._get_activation_ip(
            ip,
            f"{self.__class__.__name__} requires an input to compute activations",
        )
        activations = self.activation_func(z)
        if ip is None:
            self.activations = activations
        return activations

    def calculate_derivatives(self, ip=None):
        z = self._get_derivative_ip(
            ip,
            f"{self.__class__.__name__} requires an input to compute derivatives",
        )
        derivatives = self.derivative_func(z)
        if ip is None:
            self.derivatives = derivatives
        return derivatives


class Sigmoid(Activation):
    name = "sigmoid"

    def activation_func(self, ip):
        return 1 / (1 + np.exp(-ip))

    def derivative_func(self, ip):
        return ip * (1 - ip)


class Tanh(Activation):
    name = "tanh"

    def activation_func(self, ip):
        return np.tanh(ip)

    def derivative_func(self, ip):
        return 1 - ip ** 2


class ReLU(Activation):
    name = "relu"

    def activation_func(self, ip):
        return np.maximum(0, ip)

    def derivative_func(self, ip):
        return np.where(ip > 0, 1, 0)


class LeakyReLU(Activation):
    name = "lrelu"

    def __init__(self, ip=None, *args, **kwargs):
        alpha = kwargs.get("alpha")
        if alpha is None:
            alpha = 0.01
        self.alpha = alpha

        super().__init__(ip=ip, *args, **kwargs)

    def activation_func(self, ip):
        return np.maximum(self.alpha * ip, ip)

    def derivative_func(self, ip):
        return np.where(ip > 0, 1.0, self.alpha)
