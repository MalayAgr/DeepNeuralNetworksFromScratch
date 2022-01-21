from __future__ import annotations

from math import floor
from typing import Callable

from .base_scheduler import LearningRateScheduler


class TimeDecay(LearningRateScheduler):
    def __init__(
        self,
        initial_lr: float,
        decay_rate: float,
        decay_steps: int,
        staircase: bool = False,
    ) -> None:
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.multiplier_func = self.make_multiplier_func(staircase)

    @staticmethod
    def make_multiplier_func(staircase: bool) -> Callable[[int, int], int | float]:
        def non_staircased(iteration: int, decay_steps: int) -> float:
            return iteration / decay_steps

        def staircased(iteration: int, decay_steps: int) -> int:
            return floor(iteration / decay_steps)

        return staircased if staircase else non_staircased

    def lr(self, iteration: int) -> float:
        multiplier = self.multiplier_func(iteration, self.decay_steps)
        return self.lr0 / (1.0 + self.decay_rate * multiplier)
