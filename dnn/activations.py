import numpy as np
from abc import abstractmethod


class Activation:
    def __init__(self, ip=None):
        self.ip = ip
        self.activations = None
        self.derivatives = None

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
    def activation_func(self, ip):
        return 1 / (1 + np.exp(-ip))

    def derivative_func(self, ip):
        return ip * (1 - ip)


class Tanh(Activation):
    def activation_func(self, ip):
        return np.tanh(ip)

    def derivative_func(self, ip):
        return 1 - ip ** 2


class ReLU(Activation):
    def activation_func(self, ip):
        return np.maximum(0, ip)

    def derivative_func(self, ip):
        return np.where(ip > 0, 1, 0)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def activation_func(self, ip):
        return np.maximum(self.alpha * ip, ip)

    def derivative_func(self, ip):
        return np.where(ip > 0, 1.0, self.alpha)
