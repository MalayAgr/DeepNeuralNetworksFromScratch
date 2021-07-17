from __future__ import annotations

import numpy as np

from dnn.layers.base_layer import BaseLayer
from dnn.optimizers import Optimizer
from dnn.types import LayerInput


class Model:
    def __init__(self, ip_layer: LayerInput, op_layer: LayerInput):
        self.ip_layer = ip_layer
        self.op_layer = op_layer

        self.layers, self.trainable_layers = self._deconstruct(ip_layer, op_layer)

        self.opt = None

    @staticmethod
    def _deconstruct(
        ip_layer: LayerInput, op_layer: LayerInput
    ) -> tuple[list[BaseLayer], list[BaseLayer]]:
        layers, trainable_layers = [], []

        layer = op_layer

        while layer is not ip_layer:
            layers.append(layer)

            if layer.trainable is True:
                trainable_layers.append(layer)

            layer = layer.ip_layer

        layer.requires_dX = False

        return layers[::-1], trainable_layers[::-1]

    def __str__(self) -> str:
        layers = ", ".join([str(l) for l in self.layers])
        return f"{self.__class__.__name__}({self.ip_layer}, {layers})"

    def __repr__(self) -> str:
        return self.__str__()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.ip_layer.ip = X

        for layer in self.layers:
            layer.forward_step()

        return self.op_layer.output()

    def compile(self, opt: Optimizer):
        if not isinstance(opt, Optimizer):
            raise ValueError("opt should be an instance of a subclass of Optimizer.")

        for layer in self.layers:
            layer.build()

        self.opt = opt

    def set_operation_mode(self, training: bool) -> None:
        for layer in self.layers:
            layer.is_training = training

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        epochs: int,
        *args,
        loss: str = "bce",
        shuffle: bool = True,
        **kwargs,
    ) -> list:
        if X.shape[-1] != Y.shape[-1]:
            raise ValueError("X and Y should have the same number of samples.")

        if Y.shape[0] != self.layers[-1].units:
            msg = "Y should have the same number of rows as number of units in the final layer."
            raise ValueError(msg)

        self.set_operation_mode(training=True)

        history = self.opt.optimize(
            self, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )

        self.set_operation_mode(training=False)

        return history
