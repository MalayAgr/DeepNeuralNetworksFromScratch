from collections import defaultdict
from collections.abc import Iterator
from typing import Dict, List, Tuple, Union

import numpy as np

from .nodes import Node


class ComputationGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, List[str]] = defaultdict(list)

        self._ordering: List[str] = []

    def __str__(self) -> str:
        num_nodes = len(self.nodes)
        return f"{self.__class__.__name__}(num_nodes={num_nodes})"

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, key: Union[Node, str]) -> bool:
        if isinstance(key, Node):
            key = key.name
        return key in self.nodes

    @property
    def topological_order(self) -> List[str]:
        return self._ordering

    def add_node(self, node: Node) -> None:
        if node in self:
            return

        name = node.name

        self.nodes[name] = node

        name = node.name

        parents = node.parents

        if parents is None:
            return

        for parent in parents:
            self.adj[parent].append(name)

    def fetch_node(self, name: str) -> Node:
        node = self.nodes.get(name, None)
        if node is None:
            raise ValueError("No node with the given name found in the graph.")
        return node

    def _sink_nodes(self) -> Iterator[Tuple[str, Node]]:
        return ((name, node) for name, node in self.nodes.items() if node.is_sink)

    def _reset_topological_order(self):
        self._ordering = []

        for node in self.nodes.values():
            node.visited = False

    def _dfs_visit_node(self, u):
        for name in self.adj[u.name]:
            v = self.fetch_node(name)
            if v.visited is False:
                self._dfs_visit_node(v)

        u.visited = True
        self._ordering.append(u.name)

    def _topological_sort(self, reset=True):
        if reset is True:
            self._reset_topological_order()

        for node in self.nodes.values():
            if node.visited is False:
                self._dfs_visit_node(node)

        self._ordering.reverse()

    def forward_propagation(self) -> Tuple[np.ndarray]:
        topological_order = self.topological_order

        if not topological_order:
            self._topological_sort(reset=False)
            topological_order = self.topological_order

        for name in topological_order:
            node = self.fetch_node(name)
            node.forward()

        return tuple(node.forward_output() for _, node in self._sink_nodes())

    def _pass_gradients_to_parents(
        self, node: Node, grads: Union[np.ndarray, Tuple[np.ndarray]]
    ) -> None:
        parents = node.parents

        if len(parents) == 1:
            if not isinstance(grads, np.ndarray):
                raise TypeError(
                    f"Expected a single numpy array for {node.name} "
                    "but got a sequence containing one array."
                )

            grads = (grads,)

        for name, grad in zip(parents, grads):
            parent = self.fetch_node(name)
            parent.backprop_grad += grad

    def _backprop_single_node(self, name: str, backprop_grad: np.ndarray = None):
        node = self.fetch_node(name)

        if backprop_grad is not None:
            node.backprop_grad = backprop_grad

        grads = node.backprop()
        node.backprop_grad = 0

        if not node.is_source:
            self._pass_gradients_to_parents(node, grads)

        node_weights = node.get_trainable_weight_values()

        return ((weight, grad) for weight, grad in zip(node_weights, node.gradients))

    def backprop(self, grads: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        topological_order = self.topological_order

        if not topological_order:
            raise AttributeError(
                "You must run forward propagation before running backpropagation."
            )

        sink_nodes = tuple(name for name, _ in self._sink_nodes())

        if len(grads) != len(sink_nodes):
            raise ValueError(
                "Unexpected number of gradients received. Expected "
                f"{len(sink_nodes)} but got {len(grads)}."
            )

        weights_and_grads, sink_count = [], 0

        for name in reversed(topological_order):
            if name in sink_nodes:
                node_w_and_g = self._backprop_single_node(name, grads[sink_count])
                sink_count += 1
            else:
                node_w_and_g = self._backprop_single_node(name)

            weights_and_grads.extend(node_w_and_g)

        return weights_and_grads
