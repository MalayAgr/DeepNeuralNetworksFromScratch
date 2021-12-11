from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union


class LearningRateScheduler(ABC):
    def __init__(self, initial_lr: float) -> None:
        self.lr0 = initial_lr

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(lr0={self.lr0})"

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def lr(self, iteration: int) -> float:
        """Method to calculate the learning rate for the current iteration."""


LearningRateType = Union[float, LearningRateScheduler]
