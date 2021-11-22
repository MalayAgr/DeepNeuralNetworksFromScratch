from collections import deque
from typing import Dict, List

import numpy as np

from dnn import Input
from dnn.layers.base_layer import BaseLayerType

from .graph.core import ComputationGraph
from .graph.nodes import LayerNode


def discover_layers(
    inputs: List[Input], outputs: List[BaseLayerType]
) -> Dict[str, BaseLayerType]:
    queue = deque(outputs)

    layers: Dict[str, BaseLayerType] = {}

    while queue:
        layer = queue.popleft()

        ips = layer.ip_layer
        if not isinstance(ips, list):
            ips = [ips]

        queue.extend(ip for ip in ips if ip not in inputs)

        if layer not in layers:
            layers[layer.name] = layer

    return layers


def is_a_source_layer(layer: BaseLayerType, inputs: List[Input]) -> bool:
    ips = layer.ip_layer
    if isinstance(ips, list):
        return all(ip in inputs for ip in ips)
    return ips in inputs


def build_graph_for_model(
    layers: List[BaseLayerType],
    inputs: List[Input],
    outputs: List[BaseLayerType],
    graph: ComputationGraph = None,
) -> ComputationGraph:
    graph = ComputationGraph() if graph is None else graph

    for layer in layers:
        source = is_a_source_layer(layer, inputs)
        layer.requires_dX = not source
        sink = layer in outputs
        graph.add_node(LayerNode(layer, source=source, sink=sink))

    return graph


def validate_labels_against_outputs(
    labels: List[np.ndarray], outputs: List[BaseLayerType]
) -> None:
    if any(y.shape[:-1] != op.output_shape()[:-1] for y, op in zip(labels, outputs)):
        msg = "Each set of labels should have the same dimensions as the respective output layer."
        raise ValueError(msg)


def validate_labels_against_samples(
    samples: List[np.ndarray], labels: List[np.ndarray]
) -> None:
    if any(x.shape[-1] != y.shape[-1] for x, y in zip(samples, labels)):
        msg = "There should be an equal number of training examples in each X, Y pair."
        raise ValueError(msg)
