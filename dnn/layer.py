import numpy as np

from dnn.input_layer import Input
from dnn.utils import activation_factory


class BatchNorm:
    def __init__(self, ip, epsilon=1e-7, momentum=0.5):
        if not isinstance(ip, Layer):
            raise AttributeError("ip should be an instance of Layer")

        self.ip_layer = ip
        self.epsilon = epsilon

        self.gamma, self.beta = self.init_params()

        self.std = None
        self.Z_hat = None
        self.norm = None

        self.mean_ewa, self.std_ewa = self.init_ewa()
        self.momentum = momentum

    def init_params(self):
        gamma = np.ones(shape=(self.ip_layer.units, 1))
        beta = np.zeros(shape=(self.ip_layer.units, 1))
        return gamma, beta

    def init_ewa(self):
        mean = np.zeros(shape=(self.ip_layer.units, 1), dtype=np.float64)
        std = np.ones(shape=(self.ip_layer.units, 1), dtype=np.float64)

        return mean, std

    def update_ewa(self, mean, std):
        self.mean_ewa = self.momentum * self.mean_ewa + (1 - self.momentum) * mean
        self.std_ewa = self.momentum * self.std_ewa + (1 - self.momentum) * std

    def compute_norm(self, X):
        if self.ip_layer.is_training:
            mean = np.mean(X, axis=1, keepdims=True)

            var = np.var(X, axis=1, keepdims=True) + self.epsilon
            std = np.sqrt(var)

            self.update_ewa(mean, std)

            self.std = std
        else:
            mean = self.mean_ewa
            std = self.std_ewa

        Z_hat = np.divide(X - mean, std)
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

        self.batch_norm = batch_norm

        if batch_norm is True:
            self.batch_norm = BatchNorm(self)
            self.param_map.update(
                {"gamma": "batch_norm.gamma", "beta": "batch_norm.beta"}
            )

        self.trainable_params = self.param_count()

        self.is_training = False

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

    def param_count(self):
        count = self.weights.shape[0] * self.weights.shape[1]
        count += self.units

        if self.batch_norm is not False:
            count += 2 * self.units

        return count

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
