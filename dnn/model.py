import numpy as np

from .activations import activation_factory

from .loss import loss_factory


class Layer:
    def __init__(self, ip, units, activation):
        self.ip = ip
        self.units = units
        self.ip_shape = self.get_ip_shape(ip)
        self.train_size = self.ip_shape[-1]
        self.activation = self.init_activation(activation)
        self.weights, self.biases = self.init_params()

    def __str__(self):
        return f"{self.__class__.__name__}(ip_shape={self.ip_shape}, units={self.units}, activation={self.activation.__class__.__name__})"

    def __repr__(self):
        return self.__str__()

    def get_ip_shape(self, ip):
        if isinstance(ip, self.__class__) and hasattr(ip, "activations"):
            return ip.activations.shape
        elif isinstance(ip, self.__class__):
            return ip.units, ip.train_size
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

    def update_params(self, lr):
        self.weights -= lr * self.gradients["weights"]
        self.biases -= lr * self.gradients["biases"]


class Model:
    def __init__(self, ip_shape, layer_sizes, activations):
        if len(activations) != len(layer_sizes):
            raise AttributeError(
                "activations and layer_sizes should have the same length"
            )

        self.ip_shape = ip_shape
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.num_layers = len(layer_sizes)
        self.model = self._build_model()

    def __str__(self):
        layers = ",".join([str(l) for l in self.model])
        return f"{self.__class__.__name__}(InputLayer{self.ip_shape}, {layers})"

    def __repr__(self):
        return self.__str__()

    def _build_model(self):
        ip = np.random.randn(*self.ip_shape)

        model = ()
        for size, activation in zip(self.layer_sizes, self.activations):
            layer = Layer(ip=ip, units=size, activation=activation)
            model += (layer,)
            ip = layer

        return model

    def _forward_propagation(self, X):
        if self.ip_shape != X.shape:
            raise ValueError("The input does not have the expected shape")

        self.model[0].ip = X

        for idx in range(self.num_layers):
            self.model[idx].forward_step()

        return self.model[-1].activations

    def _backpropagation(self, preds):
        dA = self.loss.compute_derivatives(preds)

        for idx in reversed(range(self.num_layers)):
            self.model[idx].layer_gradients(dA)
            dA = self.model[idx]

    def _update_params(self, lr):
        for idx in range(self.num_layers):
            self.model[idx].update_params(lr)

    def _validate_labels_shape(self, X, Y):
        if X.shape[-1] != Y.shape[-1]:
            raise ValueError("Y needs to have the same number of samples as X")

    def train(
        self,
        X,
        Y,
        iterations=1000,
        lr=1e-3,
        loss="bse",
        show_loss=True,
        show_loss_freq=100,
        force_build=True,
    ):

        if force_build:
            self.model = self._build_model()

        self._validate_labels_shape(X, Y)

        self.loss = loss_factory(loss=loss, Y=Y)

        history = []
        for i in range(iterations):
            preds = self._forward_propagation(X)
            loss = self.loss.compute_loss(preds)
            history.append(loss)
            self._backpropagation(preds)
            self._update_params(lr=lr)

            if show_loss and (i + 1) % show_loss_freq == 0:
                print(f"Loss after {(i + 1)} iteration(s): {loss: .9f}")

        return history

    def predict(self, X):
        return self._forward_propagation(X)
