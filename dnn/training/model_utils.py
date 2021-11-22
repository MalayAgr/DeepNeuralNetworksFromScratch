from collections import deque
from collections.abc import Iterator
from typing import Dict, List, Tuple, Union

import numpy as np

from dnn import Input
from dnn.layers import BaseLayer
from dnn.layers.base_layer import MultiInputBaseLayer

from .graph.core import ComputationGraph
from .graph.nodes import LayerNode



def discover_layers(
    inputs: List[Input], outputs: List[BaseLayer]
) -> Dict[str, BaseLayer]:
    queue = deque(outputs)

    layers = {}

    while queue:
        layer = queue.popleft()

        ips = layer.ip_layer
        if not isinstance(layer, MultiInputBaseLayer):
            ips = [ips]

        queue.extend(ip for ip in ips if ip not in inputs)

        if layer not in layers:
            layers[layer.name] = layer

    return layers


def is_a_source_layer(
    layer: Union[BaseLayer, MultiInputBaseLayer], inputs: List[Input]
) -> bool:
    if isinstance(layer, MultiInputBaseLayer):
        return all(ip in inputs for ip in layer.ip_layer)
    return layer.ip_layer in inputs


def build_graph_for_model(
    layers: List[BaseLayer],
    inputs: List[Input],
    outputs: List[BaseLayer],
    graph: ComputationGraph = None,
) -> ComputationGraph:
    graph = ComputationGraph() if graph is None else graph

    for layer in layers:
        source = is_a_source_layer(layer, inputs)
        layer.requires_dX = not source
        sink = layer in outputs
        graph.add_node(LayerNode(layer, source=source, sink=sink))

    return graph


def validate_labels_against_outputs(labels: List[np.ndarray], outputs: List[BaseLayer]):
    if any(y.shape[:-1] != op.output_shape()[:-1] for y, op in zip(labels, outputs)):
        raise ValueError(
            "Each set of labels should have the same "
            "dimensions as the respective output layer."
        )


def validate_labels_against_samples(
    samples: List[np.ndarray], labels: List[np.ndarray]
):
    if any(x.shape[-1] != y.shape[-1] for x, y in zip(samples, labels)):
        raise ValueError(
            "There should be an equal number of training examples in each X, Y pair."
        )
