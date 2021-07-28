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

    def lr(self, iteration: int) -> float:
        exp = (
            iteration / self.decay_steps
            if self.staircase
            else iteration // self.decay_steps
        )

        return self.lr0 * (self.decay_rate ** exp)
