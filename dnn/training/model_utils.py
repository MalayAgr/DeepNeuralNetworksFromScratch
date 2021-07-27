from typing import List
from .graph.nodes import LayerNode
from .graph.core import ComputationGraph

from dnn import Input
from dnn.layers import BaseLayer


def _add_layer_to_graph(stop_layer: Input, layer: BaseLayer, graph: ComputationGraph):
    if layer is stop_layer:
        return

    source = False

    ips = layer.ip_layer
    if not isinstance(ips, List):
        if ips is stop_layer:
            source = True
        ips = [ips]

    node = LayerNode(layer, source=source)
    graph.add_node(node)

    for ip in ips:
        _add_layer_to_graph(stop_layer=stop_layer, layer=ip, graph=graph)


def build_graph_for_model(
    ip_layer: Input, op_layer: BaseLayer, graph: ComputationGraph = None
):
    graph = ComputationGraph() if graph is None else graph
    _add_layer_to_graph(stop_layer=ip_layer, layer=op_layer, graph=graph)

    return graph
