from collections import defaultdict
from collections.abc import Iterator
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from .nodes import Node


class ComputationGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, Set[str]] = defaultdict(set)

        self._order: List[str] = []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={len(self.nodes)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, key: Union[Node, str]) -> bool:
        if isinstance(key, Node):
            key = key.name
        return key in self.nodes

    def add_node(self, node: Node) -> None:
        if node in self:
            return

        name = node.name

        self.nodes[name] = node
        parents = node.parents

        if parents is None:
            return

        for parent in parents:
            self.adj[parent].add(name)

    def fetch_node(self, name: str) -> Node:
        node = self.nodes.get(name, None)
        if node is None:
            raise ValueError("No node with the given name found in the graph.")
        return node

    def _sink_nodes(self) -> Iterator[Tuple[str, Node]]:
        return ((name, node) for name, node in self.nodes.items() if node.is_sink)

    def _topological_sort(self):
        adj = self.adj
        seen = set()
        stack, order = [], []
        q = [node.name for node in self.nodes.values() if node.is_source is True]

        while q:
            v = q.pop()
            if v not in seen:
                seen.add(v)
                q.extend(adj[v])

                while stack and v not in adj[stack[-1]]:
                    order.append(stack.pop())

                stack.append(v)

        self._order = stack + order[::-1]

    @property
    def topological_order(self) -> List[str]:
        if not self._order:
            self._topological_sort()
        return self._order

    def forward_propagation(self) -> Tuple[np.ndarray]:
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
        print(sink_nodes)

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
