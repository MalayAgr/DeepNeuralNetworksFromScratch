from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator

import numpy as np

from ..optimizers import WeightsGradientsType
from .nodes import Node


class ComputationGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.adj: dict[str, set[str]] = defaultdict(set)

        self._order: list[str] = []
        self._sink_nodes: list[Node] = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={len(self.nodes)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, key: Node | str) -> bool:
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
    def sink_nodes(self) -> list[Node]:
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
    def topological_order(self) -> list[str]:
        if not self._order:
            self._order = self._tsort()
        return self._order

    def forward_propagation(self) -> tuple[np.ndarray]:
        t_order = self.topological_order

        for name in t_order:
            node = self.fetch_node(name)
            node.forward()

        return tuple(node.forward_output() for node in self.sink_nodes)

    def _pass_grads_to_parents(
        self, node: Node, grads: np.ndarray | tuple[np.ndarray]
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

    def _backprop_node(self, name: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        node = self.fetch_node(name)

        grads = node.backprop()
        node.backprop_grad = 0

        if not node.is_source:
            self._pass_grads_to_parents(node, grads)

        yield from zip(node.weights, node.gradients)

    def backprop(self, grads: list[np.ndarray]) -> WeightsGradientsType:
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

        func = self._backprop_node

        return list(itertools.chain(*(func(name) for name in reversed(t_order))))
