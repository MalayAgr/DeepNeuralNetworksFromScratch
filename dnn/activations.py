from abc import ABC, abstractmethod

import numpy as np

from dnn.base_layer import BaseLayer


class Activation(ABC, BaseLayer):
    name = None

    def __init__(self, *args, ip=None, **kwargs):
        super().__init__(ip=ip, trainable=False, activations=None, derivatives=None)

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

    def calculate_activations(self, ip=None):
        if ip is None:
            ip = self.input()

        return self.activation_func(ip)

    def calculate_derivatives(self, ip=None):
        if ip is None:
            ip = self.input()

        return self.derivative_func(ip)

    def count_params(self):
        return 0

    def output(self):
        return self.activations

    def output_shape(self):
        return self.input_shape()

    def forward_step(self, *args, **kwargs):
        ip = kwargs.pop("ip", None)

        self.activations = self.calculate_activations(ip)

        return self.activations

    def backprop_step(self, dA, *args, **kwargs):
        ip = kwargs.pop("ip", None)

        dZdA = self.calculate_derivatives(ip)

        self.dX = dA * dZdA

        return self.dX


class Sigmoid(Activation):
    name = "sigmoid"

    def activation_func(self, ip):
        return 1 / (1 + np.exp(-ip))

    def derivative_func(self, ip):
        activations = self.activation_func(ip)
        return activations * (1 - activations)


class Softmax(Activation):
    name = "softmax"

    def activation_func(self, ip):
        z = ip - np.max(ip, axis=0, keepdims=True)
        exp_z = np.exp(z)
        summation = np.sum(exp_z, axis=0, keepdims=True)
        return exp_z / summation

    def derivative_func(self, ip):
        activations = self.activation_func(ip)
        units = activations.shape[0]
        grads = (np.eye(units) - activations.T[..., None]) * activations[:, None, :].T
        return np.moveaxis(grads, 0, -1)

    def backprop_step(self, dA, *args, **kwargs):
        super().backprop_step(dA, *args, **kwargs)

        self.dX = np.sum(self.dX, axis=1)

        return self.dX


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

    def __init__(self, *args, ip=None, **kwargs):
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
