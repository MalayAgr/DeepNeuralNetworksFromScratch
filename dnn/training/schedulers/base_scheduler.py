from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    def __init__(self, initial_lr: float) -> None:
        self.lr0 = initial_lr

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(lr0={self.lr0})"

    @abstractmethod
    def lr(self, iteration: int) -> float:
        """Method to calculate the learning rate for the current iteration."""
