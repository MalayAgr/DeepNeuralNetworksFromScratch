from abc import ABC

import numpy as np

from dnn.loss import Loss
from dnn.training.graph.core import ComputationGraph


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 1e-2) -> None:
        self.learning_rate = learning_rate

    def minimize(self, graph: ComputationGraph, initial_grad: np.ndarray):
        pass
