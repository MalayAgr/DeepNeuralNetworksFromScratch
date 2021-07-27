from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np
from dnn import Input
from dnn.layers.base_layer import BaseLayer
from dnn.training.graph.core import ComputationGraph

from .model_utils import build_graph_for_model, flatten_layers


class Model:
    def __init__(
        self,
        inputs: Union[List[Input], Input],
        outputs: Union[List[BaseLayer], BaseLayer],
        *args,
        name: str = None,
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

        self.opt = None
        self.loss_func = None

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

    def predict(
        self, inputs: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:

        if not self.built:
            self.build()

        if not isinstance(inputs, List):
            inputs = [inputs]

        if len(inputs) != len(self.inputs):
            raise ValueError(
                "Unexpected number of inputs passed to the model. "
                f"It expected {len(self.inputs)} but got {len(inputs)}."
            )

        for ip, X in zip(self.inputs, inputs):
            ip.ip = X

        op = self._graph.forward_propagation()

        if len(self.outputs) == 1:
            op = op[0]
        return op
