from abc import ABC, abstractmethod

import numpy as np

from dnn.layer import Layer

from .utils import generate_batches, loss_factory


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01, *args, **kwargs):
        self.lr = learning_rate

    @staticmethod
    def get_layer_dA(dA_params):
        if isinstance(dA_params, Layer):
            next_weights = dA_params.weights
            next_dZ = dA_params.dZ

            return np.matmul(next_weights.T, next_dZ)
        return dA_params

    def layer_gradients(self, layer, dA_params):
        layer_dA = self.get_layer_dA(dA_params)

        dZ = layer_dA * layer.activation.calculate_derivatives(layer.linear)

        gradients = {
            "weights": np.matmul(dZ, layer.get_ip().T) / self.train_size,
            "biases": np.sum(dZ, keepdims=True, axis=1) / self.train_size,
        }

        layer.dZ = dZ
        layer.gradients = gradients

        return gradients

    def backprop(self, model, loss, preds):
        dA = loss.compute_derivatives(preds)

        for layer in reversed(model.layers):
            self.layer_gradients(layer, dA)
            dA = layer

    @abstractmethod
    def optimize(self, model, X, Y, batch_size, epochs, shuffle=True):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, *args, **kwargs):
        momentum = kwargs.pop("momentum", 0)

        if not 0 <= momentum <= 1:
            raise AttributeError("momentum should be between 0 and 1")

        self.momentum = momentum
        super().__init__(learning_rate=learning_rate, *args, **kwargs)

    @staticmethod
    def init_velocities(model):
        for layer in model.layers:
            layer.velocities = {
                "weights": np.zeros(shape=layer.weights.shape),
                "biases": np.zeros(shape=layer.biases.shape),
            }

    def update_layer_velocities(self, layer):
        dW = layer.gradients["weights"]
        db = layer.gradients["biases"]

        W_vel = layer.velocities["weights"]
        b_vel = layer.velocities["biases"]

        layer.velocities["weights"] = self.momentum * W_vel + (1 - self.momentum) * dW
        layer.velocities["biases"] = self.momentum * b_vel + (1 - self.momentum) * db

    def update_params_momentum(self, model):
        for layer in model.layers:
            self.update_layer_velocities(layer)
            layer.weights -= self.lr * layer.velocities["weights"]
            layer.biases -= self.lr * layer.velocities["biases"]

    def update_params_no_momentum(self, model):
        for layer in model.layers:
            layer.weights -= self.lr * layer.gradients["weights"]
            layer.biases -= self.lr * layer.gradients["biases"]

    def get_update_function(self, model):
        if self.momentum == 0:
            return self.update_params_no_momentum

        self.init_velocities(model)
        return self.update_params_momentum

    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        cost, history = 0, []

        update_func = self.get_update_function(model)

        for epoch in range(epochs):
            batches = generate_batches(X, Y, batch_size=batch_size, shuffle=shuffle)
            for batch_X, batch_Y, size in batches:
                self.train_size = size
                # Forward pass
                preds = model.predict(batch_X)
                # Compute cost
                loss_func = loss_factory(loss, batch_Y)
                cost = loss_func.compute_loss(preds)
                # Backprop
                self.backprop(model, loss=loss_func, preds=preds)
                # Update params
                update_func(model)

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history
