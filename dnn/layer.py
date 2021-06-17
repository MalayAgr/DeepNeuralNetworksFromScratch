import numpy as np
from .input_layer import Input
from .utils import activation_factory


class Layer:
    def __init__(self, ip, units, activation, initializer="he"):
        if not isinstance(ip, (Input, self.__class__)):
            msg = f"A {self.__class__.__name__} can have only instances of Input or itself as ip"
            raise AttributeError(msg)

        self.ip_layer = ip
        self.units = units
        self.activation = self.init_activation(activation)
        self.initializer = initializer
        self.weights, self.biases = self.init_params()

        self.linear = None
        self.activations = None
        self.dZ = None
        self.gradients = None

    def __str__(self):
        activation_cls = self.activation.__class__.__name__
        return f"{self.__class__.__name__}(units={self.units}, activation={activation_cls})"

    def __repr__(self):
        return self.__str__()

    def init_activation(self, activation):
        return activation_factory(activation)

    def get_y_dim(self):
        if isinstance(self.ip_layer, self.__class__):
            return self.ip_layer.units
        return self.ip_layer.ip_shape[0]

    def get_initializer(self, denom):
        return {
            "he": 2 / denom,
            "xavier": 1 / denom,
            "xavier_uniform": 6 / (denom + self.units),
        }[self.initializer]

    def init_params(self):
        x_dim, y_dim = self.units, self.get_y_dim()
        variance = self.get_initializer(y_dim)
        weights = np.random.randn(x_dim, y_dim) * np.sqrt(variance)
        biases = np.zeros((x_dim, 1))
        return weights, biases

    def get_ip(self):
        no_activs = (
            hasattr(self.ip_layer, "activations") and self.ip_layer.activations is None
        )
        no_ip = hasattr(self.ip_layer, "ip") and self.ip_layer.ip is None

        if no_activs or no_ip:
            raise ValueError("No input found.")

        if isinstance(self.ip_layer, self.__class__):
            return self.ip_layer.activations
        return self.ip_layer.ip

    def forward_step(self):
        linear = np.matmul(self.weights, self.get_ip()) + self.biases
        activations = self.activation.calculate_activations(linear)

        self.linear, self.activations = linear, activations

        return activations
