from typing import Generator, List, Tuple

import numpy as np
from dnn import Input
from dnn.layers import BaseLayer
from dnn.layers.base_layer import MultiInputBaseLayer
from dnn.utils import generate_batches

from .graph.core import ComputationGraph
from .graph.nodes import LayerNode

BatchGenerator = Generator[Tuple[np.ndarray, np.ndarray, int], None, None]
UnpackReturnType = Generator[
    Tuple[List[np.ndarray], List[np.ndarray], List[int]], None, None
]


def flatten_layers(
    inputs: List[Input], outputs: List[BaseLayer], accumulator: List
) -> None:
    for layer in outputs:
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


def _unpack_data_generators(generators: Tuple[BatchGenerator]) -> UnpackReturnType:
    for input_batches in zip(*generators):
        batch_X, batch_Y, sizes = [], [], []
        for input_batch in input_batches:
            batch_X.append(input_batch[0])
            batch_Y.append(input_batch[1])
            sizes.append(input_batch[2])
        yield batch_X, batch_Y, sizes


def get_data_generator(
    X: List[np.ndarray], Y: List[np.ndarray], batch_size: int, shuffle: bool = True
):
    generators = tuple(
        generate_batches(x, y, batch_size=batch_size, shuffle=shuffle)
        for x, y in zip(X, Y)
    )
    return _unpack_data_generators(generators=generators)
