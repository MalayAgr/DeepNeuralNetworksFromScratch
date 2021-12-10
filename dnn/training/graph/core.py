from __future__ import annotations

from collections import Iterator, defaultdict
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from ..optimizers import WeightsGradientsType
from .nodes import Node


class ComputationGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, Set[str]] = defaultdict(set)

        self._order: List[str] = []
        self._sink_nodes: List[Node] = None

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
        node = self.nodes.get(name)
        if node is None:
            raise ValueError(f"No node with the name {name!r} found in the graph.")
        return node

    @property
    def sink_nodes(self) -> List[Node]:
        if self._sink_nodes is None:
            self._sink_nodes = [node for node in self.nodes.values() if node.is_sink]
        return self._sink_nodes

    def _tsort(self):
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

        return stack + order[::-1]

    @property
    def topological_order(self) -> List[str]:
        if not self._order:
            self._order = self._tsort()
        return self._order

    def forward_propagation(self) -> Tuple[np.ndarray]:
        t_order = self.topological_order

        for name in t_order:
            node = self.fetch_node(name)
            node.forward()

        return tuple(node.forward_output() for node in self.sink_nodes)

    def _pass_grads_to_parents(
        self, node: Node, grads: Union[np.ndarray, Tuple[np.ndarray]]
    ) -> None:
        parents = node.parents

        if len(parents) == 1:
            if not isinstance(grads, np.ndarray):
                msg = f"Expected a single numpy array for node {node.name!r} but got a sequence instead."
                raise TypeError(msg)

            grads = (grads,)

        for name, grad in zip(parents, grads):
            parent = self.fetch_node(name)
            parent.backprop_grad += grad

    def _backprop_node(self, name: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        node = self.fetch_node(name)

        grads = node.backprop()
        node.backprop_grad = 0

        if not node.is_source:
            self._pass_grads_to_parents(node, grads)

        node_weights = node.get_trainable_weight_values()

        return ((weight, grad) for weight, grad in zip(node_weights, node.gradients))

    def backprop(self, grads: List[np.ndarray]) -> WeightsGradientsType:
        t_order = self.topological_order

        if not t_order:
            msg = "You must run forward propagation before running backpropagation."
            raise AttributeError(msg)

        sink_nodes = self.sink_nodes

        if (grads_len := len(grads)) != (nodes_len := len(sink_nodes)):
            msg = f"Unexpected number of gradients received. Expected {nodes_len} but got {grads_len}."
            raise ValueError(msg)

        for grad, node in zip(grads, sink_nodes):
            node.backprop_grad = grad

        weights_and_grads = []

        for name in reversed(t_order):
            node_w_and_g = self._backprop_node(name)
            weights_and_grads.extend(node_w_and_g)

        return weights_and_grads
