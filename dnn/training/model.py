from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any, List, Tuple, Union

import numpy as np

from dnn import Input
from dnn.layers import BaseLayer
from dnn.loss import Loss
from dnn.utils import get_data_generator, loss_factory

from . import model_utils as mutils
from .graph.core import ComputationGraph
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

        layers = mutils.discover_layers(inputs=inputs, outputs=outputs)
        self.layers = layers
        self._list_layers = list(layers.values())

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
        self._graph = mutils.build_graph_for_model(
            layers=self._list_layers,
            inputs=self.inputs,
            outputs=self.outputs,
            graph=self._graph,
        )
        self.built = True

    def count_params(self):
        return sum(layer.count_params() for layer in self._list_layers)

    def fetch_layer(self, name: str = None, idx: int = None) -> BaseLayer:
        if name is not None and idx is not None:
            raise ValueError("Specify only one of name or idx at a time.")

        if idx is not None:
            num_layers = len(self.layers)
            if not 0 <= idx < num_layers:
                raise ValueError(
                    f"{idx} is out of bounds since the model "
                    f"has only {num_layers} layers."
                )

            return self._list_layers[idx]

        if name is not None:
            layer = self.layers.get(name, None)
            if layer is None:
                raise ValueError(f"No layer with name {name} exists in the model.")
            return layer

        raise ValueError("Specify either a name or an index to fetch a layer.")

    def _forward(self, inputs: List[np.ndarray]) -> Tuple[np.ndarray]:
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
            op = self._forward(inputs=inputs)

            if len(self.outputs) == 1:
                op = op[0]
        return op

    def compile(
        self, opt: Optimizer, loss: Union[str, Loss, List[Union[str, Loss]]]
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
        preds = self._forward(batch_X)

        cost, grads = 0, []

        num_losses = len(self.losses)

        for idx, (y, pred) in enumerate(zip(batch_Y, preds)):
            loss = self.losses[0] if num_losses == 1 else self.losses[idx]
            cost += loss.compute_loss(y, pred)
            grads.append(loss.compute_derivatives(y, pred))

        self.opt.minimize(self._graph, initial_grads=grads)

        return cost

    def train_loop(
        self,
        X: List[np.ndarray],
        Y: List[np.ndarray],
        epochs: int,
        batch_size: int,
        shuffle: bool,
        verbosity: int,
    ) -> List[float]:
        history: List[float] = []

        for epoch in range(epochs):
            batches = get_data_generator(X, Y, batch_size=batch_size, shuffle=shuffle)

            cost = 0

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

        mutils.validate_labels_against_samples(X, Y)
        mutils.validate_labels_against_outputs(Y, self.outputs)

        if verbosity not in [0, 1]:
            raise ValueError("Unexpected verbosity level. Can only be 0 or 1.")

        with TrainingContext(self, training=True) as _:
            history = self.train_loop(
                X=X,
                Y=Y,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                verbosity=verbosity,
            )

        return history


class TrainingContext(ContextDecorator):
    def __init__(self, model: Model, training: bool = False) -> None:
        self.model = model
        self.training = training

    def _set_operation_mode(self, training: bool = False) -> None:
        model = self.model

        if model.is_training is not training:
            model.is_training = training
            for layer in model.layers.values():
                layer.is_training = training

    def __enter__(self):
        self._set_operation_mode(training=self.training)

    def __exit__(self, *args, **kwargs):
        self._set_operation_mode()
