from abc import ABC, abstractmethod

import numpy as np

from dnn.activations import Softmax
from dnn.layer import Layer

from .utils import generate_batches, loss_factory, rgetattr, rsetattr


class Optimizer(ABC):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        self.lr = learning_rate
        self.train_size = None

    def backprop(self, model, loss, preds):
        dA = loss.compute_derivatives(preds)

        for layer in reversed(model.layers):
            if not hasattr(layer, "param_map"):
                raise AttributeError("No param_map found.")
            layer.backprop_step(dA)
            dA = layer

    @abstractmethod
    def optimize(
        self, model, X, Y, batch_size, epochs, *args, loss="bce", shuffle=True, **kwargs
    ):
        pass


class BaseMiniBatchGD(Optimizer):
    def init_zeros_from_param_map(self, layer):
        return {
            key: np.zeros(shape=rgetattr(layer, attr).shape)
            for key, attr in layer.param_map.items()
        }

    def update_params(self, model, *args, **kwargs):
        for layer in model.layers:
            updates = self.compute_update(layer, *args, **kwargs)
            print(layer)
            for key, attr in layer.param_map.items():
                current_val = rgetattr(layer, attr)
                print("Key:", key)
                print("Previous value:", current_val)
                rsetattr(layer, attr, current_val - self.lr * updates[key])
                print("New value:", rgetattr(layer, attr), "\n\n")

    def mini_batch_step(self, model, batch_X, batch_Y, loss, *args, **kwargs):
        preds = model.predict(batch_X)

        loss_func = loss_factory(loss, batch_Y)
        cost = loss_func.compute_loss(preds)

        self.backprop(model, loss=loss_func, preds=preds)
        self.update_params(model, *args, **kwargs)

        return cost

    def optimize(
        self, model, X, Y, batch_size, epochs, *args, loss="bce", shuffle=True, **kwargs
    ):
        history = []
        step_count = 0
        for epoch in range(epochs):
            batches = generate_batches(X, Y, batch_size=batch_size, shuffle=shuffle)
            for batch_X, batch_Y, size in batches:
                step_count += 1
                cost = self.mini_batch_step(
                    model,
                    batch_X=batch_X,
                    batch_Y=batch_Y,
                    loss=loss,
                    step_count=step_count,
                    *args,
                    **kwargs,
                )

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history

    @abstractmethod
    def compute_update(self, layer, *args, **kwargs):
        pass


class SGD(BaseMiniBatchGD):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        momentum = kwargs.pop("momentum", 0)

        if not 0 <= momentum <= 1:
            raise AttributeError("momentum should be between 0 and 1")

        self.momentum = momentum
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    def init_velocities(self, model):
        for layer in model.layers:
            layer.velocities = self.init_zeros_from_param_map(layer)

    def update_layer_velocities(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.momentum * layer.velocities[key]
            rhs = (1 - self.momentum) * grad
            layer.velocities[key] = lhs + rhs

    def compute_update_momentum(self, layer, *args, **kwargs):
        self.update_layer_velocities(layer)
        return layer.velocities

    def compute_update(self, layer, *args, **kwargs):
        return layer.gradients

    def optimize(
        self, model, X, Y, batch_size, epochs, *args, loss="bce", shuffle=True, **kwargs
    ):
        if self.momentum > 0:
            self.init_velocities(model)
            self.compute_update = self.compute_update_momentum

        return super().optimize(
            model, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )


class RMSProp(BaseMiniBatchGD):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        rho = kwargs.pop("rho", 0.9)

        if not 0 <= rho <= 1:
            raise AttributeError("rho should be between 0 and 1")

        self.rho = rho
        self.epsilon = kwargs.pop("epislon", 1e-7)
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    def init_rms(self, model):
        for layer in model.layers:
            layer.rms = self.init_zeros_from_param_map(layer)

    def update_layer_rms(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.rho * layer.rms[key]
            rhs = (1 - self.rho) * np.square(grad)
            layer.rms[key] = lhs + rhs

    def compute_update(self, layer, *args, **kwargs):
        self.update_layer_rms(layer)

        return {
            key: grad / (np.sqrt(layer.rms[key]) + self.epsilon)
            for key, grad in layer.gradients.items()
        }

    def optimize(
        self, model, X, Y, batch_size, epochs, *args, loss="bce", shuffle=True, **kwargs
    ):
        self.init_rms(model)

        return super().optimize(
            model, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )


class Adam(BaseMiniBatchGD):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        momentum = kwargs.pop("beta_1", 0.9)

        if not 0 <= momentum <= 1:
            raise AttributeError("alpha should be between 0 and 1")

        rho = kwargs.pop("beta_2", 0.999)

        if not 0 <= rho <= 1:
            raise AttributeError("beta should be between 0 and 1")

        self.momentum = momentum
        self.rho = rho
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    def init_moments(self, model):
        for layer in model.layers:
            layer.m1 = self.init_zeros_from_param_map(layer)
            layer.m2 = self.init_zeros_from_param_map(layer)

    def update_layer_moments(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.momentum * layer.m1[key]
            rhs = (1 - self.momentum) * grad
            layer.m1[key] = lhs + rhs

            lhs = self.rho * layer.m2[key]
            rhs = (1 - self.rho) * np.square(grad)
            layer.m2[key] = lhs + rhs

    def compute_update(self, layer, *args, **kwargs):
        t = kwargs.pop("step_count", None)

        if t is None:
            raise ValueError("No t found for bias correction")

        self.update_layer_moments(layer)

        updates = {}
        for key in layer.gradients:
            m1 = np.divide(layer.m1[key], 1.0 - self.momentum ** t)
            m2 = np.divide(layer.m2[key], 1.0 - self.rho ** t)

            updates[key] = m1 / (np.sqrt(m2) + self.epsilon)

        return updates

    def optimize(
        self, model, X, Y, batch_size, epochs, *args, loss="bce", shuffle=True, **kwargs
    ):
        self.init_moments(model)

        return super().optimize(
            model, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )
