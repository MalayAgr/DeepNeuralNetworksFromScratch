import numpy as np

from .activations import activation_factory


class Layer:
    def __init__(self, ip, units, activation):
        self.ip = ip
        self.units = units
        self.ip_shape = self.get_ip_shape(ip)
        self.train_size = self.ip_shape[-1]
        self.activation = self.init_activation(activation)
        self.weights, self.biases = self.init_params()

    def __repr__(self):
        return f"{self.__class__.__name__}(ip_shape={self.ip_shape}, units={self.units}, activation={self.activation.__class__.__name__})"

    def get_ip_shape(self, ip):
        if isinstance(ip, self.__class__) and hasattr(ip, "activations"):
            return ip.activations.shape
        elif isinstance(ip, self.__class__):
            return ip.weights.shape[0], ip.ip_shape[-1]
        return ip.shape

    def get_ip(self):
        if isinstance(self.ip, self.__class__):
            return self.ip.activations
        return self.ip

    def init_activation(self, activation):
        return activation_factory(activation)

    def init_params(self):
        x_dim, y_dim = self.units, self.ip_shape[0]
        weights = np.random.randn(x_dim, y_dim) * 0.01
        biases = np.zeros((x_dim, 1))
        return weights, biases

    def forward_step(self):
        linear = np.matmul(self.weights, self.get_ip()) + self.biases
        activations = self.activation.calculate_activations(linear)

        self.linear, self.activations = linear, activations

        return activations

    def get_layer_dA(self, dA_params):
        if isinstance(dA_params, self.__class__):
            next_weights = dA_params.weights
            next_dZ = dA_params.dZ

            return np.matmul(next_weights.T, next_dZ)
        return dA_params

    def layer_gradients(self, dA_params):
        layer_dA = self.get_layer_dA(dA_params)

        dZ = layer_dA * self.activation.calculate_derivatives(self.linear)

        gradients = {
            "weights": np.matmul(dZ, self.get_ip().T) / self.train_size,
            "biases": np.sum(dZ, keepdims=True, axis=1) / self.train_size,
        }

        self.dZ = dZ
        self.gradients = gradients

        return gradients
