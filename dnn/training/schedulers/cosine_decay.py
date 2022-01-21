from __future__ import annotations

import math

from .base_scheduler import LearningRateScheduler


class CosineDecay(LearningRateScheduler):
    def __init__(self, initial_lr: float, decay_steps: int, alpha: float = 0.0) -> None:
        super().__init__(initial_lr)
        self.decay_steps = decay_steps
        self.alpha = alpha

    def lr(self, iteration: int) -> float:
        decay_steps, alpha = self.decay_steps, self.alpha
        step = min(iteration, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return self.lr0 * decayed
