from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

from .nodes import Node


class ComputationGraph:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self._node_names: List[str] = []
        self.adj: Dict[str, List[str]] = defaultdict(list)

        self._ordering: List[str] = []

    def __str__(self) -> str:
        num_nodes = len(self._node_names)
        return f"{self.__class__.__name__}(num_nodes={num_nodes})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def topological_order(self) -> List[str]:
        return self._ordering

    def add_node(self, node: Node):
        self.nodes.append(node)

        name = node.name

        self._node_names.append(name)

        parents = node.parents

        if parents is None:
            return

        for parent in parents:
            self.adj[parent].append(name)

    def fetch_node(self, name: str) -> Node:
        try:
            idx = self._node_names.index(name)
        except ValueError:
            raise ValueError("No node with the given name found in the graph.")

        return self.nodes[idx]

    def _reset_topological_order(self):
        self._ordering = []

        for node in self.nodes:
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

        for node in self.nodes:
            if node.visited is False:
                self._dfs_visit_node(node)

        self._ordering.reverse()

    def forward_propagation(self):
        self._topological_sort()

        for name in self.topological_order:
            node = self.fetch_node(name)
            node.forward()

        return node.forward_output()

    def _pass_gradients_to_parents(
        self, node: Node, grads: Union[np.ndarray, Tuple[np.ndarray]]
    ) -> None:
        parents = node.parents

        if len(parents) == 1:
            if not isinstance(grads, np.ndarray):
                raise TypeError(
                    f"Expected a single numpy array for {node.name} but got a tuple of values instead."
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

        if not node.is_source:
            self._pass_gradients_to_parents(node, grads)

        node_weights = node.get_trainable_weight_values()

        return [(weight, grad) for weight, grad in zip(node_weights, node.gradients)]

    def backprop(self, grad: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not self.topological_order:
            raise AttributeError(
                "You must run forward propagation before running backpropagation."
            )

        weights_and_grads = self._backprop_single_node(
            self.topological_order[-1], backprop_grad=grad
        )

        for name in reversed(self.topological_order[:-1]):
            node_w_and_g = self._backprop_single_node(name)
            weights_and_grads.extend(node_w_and_g)

        return weights_and_grads
