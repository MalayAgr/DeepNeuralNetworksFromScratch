from __future__ import annotations

from typing import Any, List, Tuple, Union
from contextlib import ContextDecorator

import numpy as np
from dnn import Input
from dnn.layers import BaseLayer
from dnn.loss import Loss
from dnn.utils import loss_factory

from .graph.core import ComputationGraph
from .model_utils import (
    build_graph_for_model,
    flatten_layers,
    get_data_generator,
    validate_labels_against_outputs,
    validate_labels_against_samples,
)
from .optimizers import Optimizer


class Model:
    def __init__(
        self,
        inputs: Union[List[Input], Input],
        outputs: Union[List[BaseLayer], BaseLayer],
        *args,
        graph: ComputationGraph = None,
        **kwargs,
    ) -> None:
        if not isinstance(inputs, List):
            inputs = [inputs]

        if not isinstance(outputs, List):
            outputs = [outputs]

        self.inputs = inputs
        self.outputs = outputs

        layers: List[BaseLayer] = []
        flatten_layers(inputs=inputs, outputs=outputs, accumulator=layers)
        self.layers = layers

        self._graph = graph

        self._built = False

        self._compiled = False

        self.is_training = False

        self.opt: Optimizer = None
        self.losses: List[Loss] = None

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, value: bool):
        self._built = value

    def build(self) -> Any:
        self._graph = build_graph_for_model(
            layers=self.layers,
            inputs=self.inputs,
            outputs=self.outputs,
            graph=self._graph,
        )
        self.built = True

    def fetch_layer(self, name: str = None, idx: int = None) -> BaseLayer:
        num_layers = len(self.layers)

        if name is not None and idx is not None:
            raise ValueError("Specify only one of name or idx at a time.")

        if idx is not None:
            if num_layers <= idx:
                raise ValueError(
                    f"{idx} is out of bounds since the model "
                    f"has only {num_layers} layers."
                )

            return self.layers[idx]

        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(f"No layer with name {name} exists in the model.")

        raise ValueError("Specify either a name or an index to fetch a layer.")

    def _forward_step(self, inputs: List[np.ndarray]) -> Tuple[np.ndarray]:
        if not isinstance(inputs, List):
            raise TypeError("Expected a list of inputs.")

        if len(inputs) != len(self.inputs):
            raise ValueError(
                "Unexpected number of inputs passed to the model. "
                f"It expected {len(self.inputs)} but got {len(inputs)}."
            )

        if not self.built:
            self.build()

        for ip, X in zip(self.inputs, inputs):
            ip.ip = X

        return self._graph.forward_propagation()

    def predict(
        self, inputs: Union[np.ndarray, List[np.ndarray]], training: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:

        if not isinstance(inputs, List):
            inputs = [inputs]

        with TrainingContext(self, training=training) as _:
            op = self._forward_step(inputs=inputs)

            if len(self.outputs) == 1:
                op = op[0]
        return op

    def compile(
        self, opt: Optimizer, loss: Union[str, Loss, List[str], List[Loss]]
    ) -> None:
        if not isinstance(opt, Optimizer):
            raise TypeError(
                f"Expected an instance of Optimizer but got {type(opt)} instead."
            )

        if not isinstance(loss, List):
            loss = [loss]

        self.opt = opt
        self.losses = [loss_factory(l) if isinstance(l, str) else l for l in loss]

        self._compiled = True

    def train_step(
        self, batch_X: List[np.ndarray], batch_Y: List[np.ndarray], sizes: List[int]
    ) -> float:
        preds = self._forward_step(batch_X)

        cost, grads = 0, []

        num_losses = len(self.losses)

        for idx, (y, pred) in enumerate(zip(batch_Y, preds)):
            loss = self.losses[0] if num_losses == 1 else self.losses[idx]
            cost += loss.compute_loss(y, pred)
            grads.append(loss.compute_derivatives(y, pred))

        self.opt.minimize(self._graph, initial_grads=grads)

        return cost

    def train(
        self,
        X: Union[List[np.ndarray], np.ndarray],
        Y: Union[List[np.ndarray], np.ndarray],
        batch_size: int,
        epochs: int,
        shuffle: bool = True,
        verbosity: int = 1,
    ) -> List[float]:
        if not self._compiled:
            raise RuntimeError("Compile the model before training it.")

        if not isinstance(X, List):
            X = [X]

        if not isinstance(Y, List):
            Y = [Y]

        validate_labels_against_samples(X, Y)
        validate_labels_against_outputs(Y, self.outputs)

        if verbosity not in [0, 1]:
            raise ValueError("Unexpected verbosity level. Can only be 0 or 1.")

        history = []

        with TrainingContext(self, training=True) as _:
            for epoch in range(epochs):
                batches = get_data_generator(
                    X, Y, batch_size=batch_size, shuffle=shuffle
                )

                print(f"Epoch {epoch + 1}/{epochs}:")

                for step, (batch_X, batch_Y, sizes) in enumerate(batches):
                    cost = self.train_step(batch_X, batch_Y, sizes)

                    log_msg = (
                        f"\r  Step {step + 1}: Train loss = {cost: .5f}"
                        if verbosity == 1
                        else f"\r  Train loss = {cost: .5f}"
                    )
                    print(log_msg, end="", flush=True)

                print()
                history.append(cost)

        return history


class TrainingContext(ContextDecorator):
    def __init__(self, model: Model, training: bool = False) -> None:
        self.model = model
        self.training = training

    def _set_operation_mode(self, training: bool = False) -> None:
        model = self.model

        if model.is_training is not training:
            model.is_training = training
            for layer in model.layers:
                layer.is_training = training

    def __enter__(self):
        self._set_operation_mode(training=self.training)

    def __exit__(self, *args, **kwargs):
        self._set_operation_mode()
