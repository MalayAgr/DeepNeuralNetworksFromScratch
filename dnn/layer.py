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

    def forward_step(self, X):
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

    def backprop_step(self, dA, bs):
        activation_grads = self.ip_layer.activation.calculate_derivatives(self.norm)

        d_norm = self.ip_layer.compute_dZ(dA, activation_grads)

        grads = {
            "gamma": np.sum(d_norm * self.Z_hat, axis=1, keepdims=True),
            "beta": np.sum(d_norm, axis=1, keepdims=True),
        }

        self.ip_layer.gradients.update(grads)

        dZ_hat = d_norm * self.gamma

        dZ_hat_sum = np.sum(dZ_hat, axis=1, keepdims=True)
        dZ_hat_prod = self.Z_hat * np.sum(dZ_hat * self.Z_hat, axis=1, keepdims=True)

        return (bs * dZ_hat - dZ_hat_sum - dZ_hat_prod) / (bs * self.std)


class Layer:
    def __init__(
        self, ip, units, activation, initializer="he", batch_norm=False, dropout=1.0
    ):
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

        self.dropout = dropout

        self.trainable_params = self.param_count()

        self.is_training = False

        self.linear = None
        self.dropout_mask = None
        self.activations = None
        self.dZ = None
        self.gradients = {}

    def __str__(self):
        activation_cls = self.activation.__class__.__name__
        return f"{self.__class__.__name__}(units={self.units}, activation={activation_cls})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def init_activation(activation):
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
            activation_ip = self.batch_norm.forward_step(linear)

        activations = self.activation.calculate_activations(activation_ip)

        if self.dropout < 1.0 and self.is_training:
            mask = np.random.rand(*activations.shape) < self.dropout
            activations *= mask
            activations /= self.dropout

            self.dropout_mask = mask

        self.linear, self.activations = linear, activations

        return activations

    def compute_dA(self, dA_params):
        if isinstance(dA_params, self.__class__):
            return np.matmul(dA_params.weights.T, dA_params.dZ)
        return dA_params

    @staticmethod
    def compute_dZ(dA, activation_grads):
        if len(activation_grads.shape) > 2:
            return np.sum(dA * activation_grads, axis=1)
        return dA * activation_grads

    def backprop_step(self, dA_params):
        dA = self.compute_dA(dA_params)

        if self.dropout < 1.0:
            dA = dA * self.dropout_mask
            dA /= self.dropout

        ip = self.get_ip()

        if self.batch_norm is not False:
            dZ = self.batch_norm.backprop_step(dA, ip.shape[-1])
        else:
            activation_grads = self.activation.calculate_derivatives(self.linear)
            dZ = self.compute_dZ(dA, activation_grads)

        gradients = {
            "weights": np.matmul(dZ, ip.T) / ip.shape[-1],
            "biases": np.sum(dZ, keepdims=True, axis=1) / ip.shape[-1],
        }

        self.gradients.update(gradients)
        self.dZ = dZ

        return gradients
