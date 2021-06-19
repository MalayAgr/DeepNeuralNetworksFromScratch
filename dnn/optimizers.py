from abc import ABC, abstractmethod

import numpy as np

from dnn.layer import Layer

from .utils import generate_batches, loss_factory


class Optimizer(ABC):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        self.lr = learning_rate
        self.train_size = None

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
    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        pass


class SGD(Optimizer):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
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


class RMSProp(Optimizer):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        rho = kwargs.pop("rho", 0.9)

        if not 0 <= rho <= 1:
            raise AttributeError("rho should be between 0 and 1")

        self.rho = rho
        self.epsilon = kwargs.pop("epislon", 1e-7)
        super().__init__(learning_rate=learning_rate, *args, **kwargs)

    @staticmethod
    def init_rms(model):
        for layer in model.layers:
            layer.rms = {
                "weights": np.zeros(shape=layer.weights.shape),
                "biases": np.zeros(shape=layer.biases.shape),
            }

    def update_layer_rms(self, layer):
        dW = layer.gradients["weights"]
        db = layer.gradients["biases"]

        W_rms = layer.rms["weights"]
        b_rms = layer.rms["biases"]

        layer.rms["weights"] = self.rho * W_rms + (1 - self.rho) * np.square(dW)
        layer.rms["biases"] = self.rho * b_rms + (1 - self.rho) * np.square(db)

    def get_update(self, grad, rms):
        return self.lr * (grad / (np.sqrt(rms) + self.epsilon))

    def update_params(self, model):
        for layer in model.layers:
            self.update_layer_rms(layer)
            layer.weights -= self.get_update(
                layer.gradients["weights"], layer.rms["weights"]
            )
            layer.biases -= self.get_update(
                layer.gradients["biases"], layer.rms["biases"]
            )

    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        cost, history = 0, []

        self.init_rms(model)

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
                self.update_params(model)

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history


class Adam(Optimizer):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        momentum = kwargs.pop("alpha", 0.9)

        if not 0 <= momentum <= 1:
            raise AttributeError("alpha should be between 0 and 1")

        rho = kwargs.pop("beta", 0.999)

        if not 0 <= rho <= 1:
            raise AttributeError("beta should be between 0 and 1")

        self.momentum = momentum
        self.rho = rho
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        super().__init__(learning_rate=learning_rate, *args, **kwargs)

    @staticmethod
    def init_moments(model):
        for layer in model.layers:
            layer.m1 = {
                "weights": np.zeros(shape=layer.weights.shape),
                "biases": np.zeros(shape=layer.biases.shape),
            }
            layer.m2 = {
                "weights": np.zeros(shape=layer.weights.shape),
                "biases": np.zeros(shape=layer.biases.shape),
            }

    def update_layer_m1(self, layer):
        dW = layer.gradients["weights"]
        db = layer.gradients["biases"]

        W_m1 = layer.m1["weights"]
        b_m1 = layer.m1["biases"]

        layer.m1["weights"] = self.momentum * W_m1 + (1 - self.momentum) * dW
        layer.m1["biases"] = self.momentum * b_m1 + (1 - self.momentum) * db

    def update_layer_m2(self, layer):
        dW = layer.gradients["weights"]
        db = layer.gradients["biases"]

        W_m2 = layer.m2["weights"]
        b_m2 = layer.m2["biases"]

        layer.m2["weights"] = self.rho * W_m2 + (1 - self.rho) * np.square(dW)
        layer.m2["biases"] = self.rho * b_m2 + (1 - self.rho) * np.square(db)

    def get_update(self, m1, m2, t):
        m1 = np.divide(m1, 1.0 - self.momentum ** t)
        m2 = np.divide(m2, 1.0 - self.rho ** t)

        return self.lr * (m1 / (np.sqrt(m2) + self.epsilon))

    def update_params(self, model, t):
        for layer in model.layers:
            self.update_layer_m1(layer)
            self.update_layer_m2(layer)

            layer.weights -= self.get_update(
                layer.m1["weights"], layer.m2["weights"], t
            )
            layer.biases -= self.get_update(layer.m1["biases"], layer.m2["biases"], t)

    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        cost, history = 0, []

        self.init_moments(model)

        t = 0
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
                t += 1
                self.update_params(model, t)

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history
