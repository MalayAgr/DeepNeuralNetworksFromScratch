import numpy as np
from abc import abstractmethod


class Activation:
    def __init__(self, ip=None):
        self.ip = ip
        self.activations = None
        self.slope = None

    def get_compute_ip(self, ip, message):
        if ip is None and self.ip is None:
            raise AttributeError(message)

        if ip is None:
            return self.ip
        return ip

    def get_derivative_ip(self, ip, message):
        if ip is None and self.activations is None:
            raise AttributeError(message)

        if ip is None:
            return self.activations
        return self.compute(ip)

    @abstractmethod
    def compute(self, ip=None):
        return self.get_compute_ip(
            ip,
            f"{self.__class__.__name__} requires an input to compute activations",
        )

    @abstractmethod
    def derivative(self, ip=None):
        return self.get_derivative_ip(
            ip,
            f"{self.__class__.__name__} requires an input to compute derivatives",
        )


class Sigmoid(Activation):
    def compute(self, ip=None):
        z = super().compute(ip)
        activations = 1 / (1 + np.exp(-z))
        if ip is None:
            self.activations = activations
        return activations

    def derivative(self, ip=None):
        z = super().derivative(ip)
        slope = z * (1 - z)
        if ip is None:
            self.slope = slope
        return slope


class Tanh(Activation):
    def compute(self, ip=None):
        z = super().compute(ip)
        activations = np.tanh(z)
        if ip is None:
            self.activations = activations
        return activations

    def derivative(self, ip=None):
        z = super().derivative(ip)
        slope = 1 - z ** 2
        if ip is None:
            self.slope = slope
        return slope


class ReLU(Activation):
    def compute(self, ip=None):
        z = super().compute(ip)
        activations = np.maximum(0, z)
        if ip is None:
            self.activations = activations
        return activations

    def derivative(self, ip=None):
        z = super().derivative(ip)
        slope = np.where(z > 0, 1, 0)
        if ip is None:
            self.slope = slope
        return slope


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def compute(self, ip=None):
        z = super().compute(ip)
        activations = np.maximum(self.alpha * z, z)
        if ip is None:
            self.activations = activations
        return activations

    def derivative(self, ip=None):
        z = super().derivative(ip)
        slope = np.where(z > 0, 1.0, self.alpha)
        if ip is None:
            self.slope = slope
        return slope
