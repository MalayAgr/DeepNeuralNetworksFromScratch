from __future__ import annotations

from typing import Callable

from .base_scheduler import LearningRateScheduler


class ExponentialDecay(LearningRateScheduler):
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
        self.exp_func = self.make_exp_function(staircase)

    @staticmethod
    def make_exp_function(staircase: bool) -> Callable[[int, int], int | float]:
        def staircased(iteration: int, decay_steps: int) -> int:
            return iteration // decay_steps

        def non_staircased(iteration: int, decay_steps: int) -> float:
            return iteration / decay_steps

        return staircased if staircase else non_staircased

    def lr(self, iteration: int) -> float:
        exp = self.exp_func(iteration, self.decay_steps)
        return self.lr0 * (self.decay_rate ** exp)
