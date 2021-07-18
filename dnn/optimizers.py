from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dnn.layers.base_layer import BaseLayer
from dnn.loss import Loss

from .utils import (
    backprop,
    compute_l2_cost,
    generate_batches,
    loss_factory,
    rgetattr,
    rsetattr,
)


class Optimizer(ABC):
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs) -> None:
        self.lr = learning_rate

    @abstractmethod
    def optimize(
        self,
        model: "Model",
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        epochs: int,
        *args,
        loss: str = "bce",
        shuffle: bool = True,
        **kwargs,
    ) -> list:
        pass


class BaseMiniBatchGD(Optimizer):
    @staticmethod
    def init_zeros_from_params(layer: BaseLayer) -> dict:
        shapes = layer.get_param_shapes(layer.param_keys)

        return {
            key: np.zeros(shape=shape, dtype=np.float32)
            for key, shape in shapes.items()
        }

    def update_params(self, model: "Model", *args, **kwargs) -> None:
        for layer in model.trainable_layers:
            updates = self.compute_update(layer, *args, **kwargs)
            updates = {key: self.lr * value for key, value in updates.items()}
            layer.update_params(updates)
            layer.gradients = {}

    def mini_batch_step(
        self,
        model: "Model",
        batch_X: np.ndarray,
        batch_Y: np.ndarray,
        loss: Loss,
        *args,
        **kwargs,
    ) -> float:
        preds = model.predict(batch_X)

        cost = loss.compute_loss(batch_Y, preds)

        reg_param = kwargs.pop("reg_param", 0.0)

        if reg_param > 0.0:
            cost = compute_l2_cost(model, reg_param=reg_param, cost=cost)

        backprop(model, loss=loss, labels=batch_Y, preds=preds, reg_param=reg_param)

        self.update_params(model, *args, **kwargs)

        return cost

    def optimize(
        self,
        model: "Model",
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        epochs: int,
        *args,
        loss: str = "bce",
        shuffle: bool = True,
        **kwargs,
    ) -> list:
        history, step_count = [], 0

        loss_func = loss_factory(loss)

        for epoch in range(epochs):
            batches = generate_batches(X, Y, batch_size=batch_size, shuffle=shuffle)

            print(f"Epoch {epoch + 1}/{epochs}:")

            for step, (batch_X, batch_Y, size) in enumerate(batches):
                step_count += 1

                cost = self.mini_batch_step(
                    model,
                    batch_X=batch_X,
                    batch_Y=batch_Y,
                    loss=loss_func,
                    step_count=step_count,
                    batch_size=size,
                    *args,
                    **kwargs,
                )

                log_msg = f"\r  Step {step + 1}: Train loss = {cost: .5f}"
                print(log_msg, end="", flush=True)

            print()
            history.append(cost)

        return history

    @abstractmethod
    def compute_update(self, layer: BaseLayer, *args, **kwargs) -> dict:
        pass


class SGD(BaseMiniBatchGD):
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs) -> None:
        momentum = kwargs.pop("momentum", 0)

        if not 0 <= momentum <= 1:
            raise AttributeError("momentum should be between 0 and 1")

        self.momentum = momentum
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    def init_velocities(self, model: "Model") -> None:
        for layer in model.trainable_layers:
            layer.velocities = self.init_zeros_from_params(layer)

    def update_layer_velocities(self, layer: BaseLayer) -> None:
        for key, grad in layer.gradients.items():
            layer.velocities[key] *= self.momentum
            layer.velocities[key] += (1 - self.momentum) * grad

    def compute_update_momentum(self, layer: BaseLayer, *args, **kwargs) -> dict:
        self.update_layer_velocities(layer)
        return layer.velocities

    def compute_update(self, layer: BaseLayer, *args, **kwargs) -> dict:
        return layer.gradients

    def optimize(
        self,
        model: "Model",
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        epochs: int,
        *args,
        loss: str = "bce",
        shuffle: bool = True,
        **kwargs,
    ) -> list:
        if self.momentum > 0:
            self.init_velocities(model)
            self.compute_update = self.compute_update_momentum

        return super().optimize(
            model, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )


class RMSProp(BaseMiniBatchGD):
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs) -> None:
        rho = kwargs.pop("rho", 0.9)

        if not 0 <= rho <= 1:
            raise AttributeError("rho should be between 0 and 1")

        self.rho = rho
        self.epsilon = kwargs.pop("epislon", 1e-7)
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    def init_rms(self, model: "Model"):
        for layer in model.trainable_layers:
            layer.rms = self.init_zeros_from_params(layer)

    def update_layer_rms(self, layer: BaseLayer) -> None:
        for key, grad in layer.gradients.items():
            layer.rms[key] *= self.rho
            layer.rms[key] += (1 - self.rho) * np.square(grad)

    def compute_update(self, layer: BaseLayer, *args, **kwargs) -> dict:
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
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs) -> None:
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

    def init_moments(self, model: "Model") -> None:
        for layer in model.trainable_layers:
            layer.m1 = self.init_zeros_from_params(layer)
            layer.m2 = self.init_zeros_from_params(layer)

    def update_layer_moments(self, layer: BaseLayer) -> None:
        for key, grad in layer.gradients.items():
            layer.m1[key] *= self.momentum
            layer.m1[key] += (1 - self.momentum) * grad

            layer.m2[key] *= self.rho
            layer.m2[key] += (1 - self.rho) * np.square(grad)

    def compute_update(self, layer: BaseLayer, *args, **kwargs) -> dict:
        t = kwargs.pop("step_count", None)

        if t is None:
            raise ValueError("No t found for bias correction")

        self.update_layer_moments(layer)

        updates = {}
        for key in layer.gradients:
            m1 = np.divide(layer.m1[key], 1.0 - self.momentum ** t)
            updates[key] = m1

            m1 = np.divide(layer.m2[key], 1.0 - self.rho ** t)
            updates[key] /= np.sqrt(m1) + self.epsilon

        return updates

    def optimize(
        self,
        model: "Model",
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        epochs: int,
        *args,
        loss: str = "bce",
        shuffle: bool = True,
        **kwargs,
    ) -> list:
        self.init_moments(model)

        return super().optimize(
            model, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )
