from typing import List

from dnn import Input
from dnn.layers import BaseLayer
from dnn.layers.base_layer import MultiInputBaseLayer

from .graph.core import ComputationGraph
from .graph.nodes import LayerNode


def flatten_layers(
    inputs: List[Input], outputs: List[BaseLayer], accumulator: List
) -> None:
    for layer in outputs:
        print(layer)
        if layer in inputs:
            return

        ips = layer.ip_layer
        if not isinstance(layer, MultiInputBaseLayer):
            ips = [ips]

        flatten_layers(inputs=inputs, outputs=ips, accumulator=accumulator)

        if layer not in accumulator:
            accumulator.append(layer)


def is_a_source_layer(layer: BaseLayer, inputs: List[Input]) -> bool:
    ips = layer.ip_layer
    if isinstance(layer, MultiInputBaseLayer):
        return all(ip in inputs for ip in ips)
    return ips in inputs


def build_graph_for_model(
    layers: List[BaseLayer],
    inputs: List[Input],
    outputs: List[BaseLayer],
    graph: ComputationGraph = None,
) -> ComputationGraph:
    graph = ComputationGraph() if graph is None else graph

    for layer in layers:
        source = is_a_source_layer(layer, inputs)
        sink = layer in outputs
        graph.add_node(LayerNode(layer, source=source, sink=sink))

    return graph
