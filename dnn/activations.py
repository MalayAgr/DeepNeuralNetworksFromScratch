from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    name = None

    def __init__(self, ip=None, *args, **kwargs):
        self.ip = ip
        self.activations = None
        self.derivatives = None

    @classmethod
    def get_activation_classes(cls):
        result = {}

        for sub_cls in cls.__subclasses__():
            name = sub_cls.name
            result.update(sub_cls.get_activation_classes())
            if name is not None and name not in result:
                result.update({name: sub_cls})
        return result

    @abstractmethod
    def activation_func(self, ip):
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
    def derivative_func(self, ip):
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

    def _get_ip(self, ip, message):
        if ip is None and self.ip is None:
            raise AttributeError(message)

        if ip is None:
            return self.ip
        return ip

    def calculate_activations(self, ip=None):
        z = self._get_ip(
            ip,
            f"{self.__class__.__name__} requires an input to compute activations",
        )
        activations = self.activation_func(z)
        if ip is None:
            self.activations = activations
        return activations

    def calculate_derivatives(self, ip=None):
        z = self._get_ip(
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
        activations = self.activation_func(ip)
        return activations * (1 - activations)


class Tanh(Activation):
    name = "tanh"

    def activation_func(self, ip):
        return np.tanh(ip)

    def derivative_func(self, ip):
        return 1 - self.activation_func(ip) ** 2


class ReLU(Activation):
    name = "relu"

    def activation_func(self, ip):
        return np.maximum(0, ip)

    def derivative_func(self, ip):
        return np.where(ip > 0, 1.0, 0)


class LeakyReLU(Activation):
    name = "lrelu"
    default_alpha = 0.01

    def __init__(self, ip=None, *args, **kwargs):
        alpha = kwargs.pop("alpha", None)
        if alpha is None:
            alpha = self.default_alpha
        self.alpha = alpha

        super().__init__(ip=ip, *args, **kwargs)

    def activation_func(self, ip):
        return np.where(ip > 0, ip, self.alpha * ip)

    def derivative_func(self, ip):
        return np.where(ip > 0, 1.0, self.alpha)


class ELU(LeakyReLU):
    name = "elu"
    default_alpha = 1.0

    def activation_func(self, ip):
        return np.where(ip > 0, ip, self.alpha * (np.exp(ip) - 1))

    def derivative_func(self, ip):
        return np.where(ip > 0, 1.0, self.alpha * np.exp(ip))
