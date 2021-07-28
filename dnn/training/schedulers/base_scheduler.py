from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    def __init__(self, initial_lr: float) -> None:
        self.lr0 = initial_lr

    @abstractmethod
    def lr(self, iteration: int) -> float:
        """Method to calculate the learning rate for the current iteration."""
