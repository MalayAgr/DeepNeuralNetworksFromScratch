import numpy as np

from dnn.input_layer import Input
from dnn.utils import activation_factory


class BatchNorm:
    def __init__(self, ip, epsilon=1e-7):
        if not isinstance(ip, Layer):
            raise AttributeError("ip should be an instance of Layer")

        self.ip_layer = ip
        self.epsilon = epsilon

        self.gamma, self.beta = self.init_params()

        self.std = None
        self.Z_hat = None
        self.norm = None

    def init_params(self):
        gamma = np.ones(shape=(self.ip_layer.units, 1))
        beta = np.zeros(shape=(self.ip_layer.units, 1))
        return gamma, beta

    def compute_norm(self, X):
        mean = np.mean(X, axis=1, keepdims=True)

        var = np.var(X, axis=1, keepdims=True)
        std = np.sqrt(var + self.epsilon)
        self.std = std

        Z_hat = (X - mean) / std
        self.Z_hat = Z_hat

        self.norm = self.gamma * Z_hat + self.beta
        return self.norm


class Layer:
    def __init__(self, ip, units, activation, initializer="he", batch_norm=False):
        if not isinstance(ip, (Input, self.__class__)):
            msg = f"A {self.__class__.__name__} can have only instances of Input or itself as ip"
            raise AttributeError(msg)

        self.ip_layer = ip

        self.units = units
        self.activation = self.init_activation(activation)

        self.initializer = initializer
        self.weights, self.biases = self.init_params()

        self.param_map = {"weights": "weights", "biases": "biases"}

        if batch_norm is True:
            self.batch_norm = BatchNorm(self)
            self.param_map.update(
                {"gamma": "batch_norm.gamma", "beta": "batch_norm.beta"}
            )
        else:
            self.batch_norm = False

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

        activation_ip = linear
        if self.batch_norm is not False:
            activation_ip = self.batch_norm.compute_norm(linear)

        activations = self.activation.calculate_activations(activation_ip)

        self.linear, self.activations = linear, activations

        return activations
