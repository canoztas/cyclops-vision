from __future__ import annotations

import numpy as np
from cyclops.attacks.base import Attack, register
from cyclops.adapters.base import ModelAdapter


class PGDAttack(Attack):
    name = "pgd"
    requires_gradients = True

    def run(
        self,
        adapter: ModelAdapter,
        x: np.ndarray,
        y: np.ndarray,
        eps: float = 4 / 255,
        steps: int = 10,
        step_size: float | None = None,
        rand_start: bool = True,
        **_: dict,
    ) -> np.ndarray:
        if step_size is None:
            step_size = eps / 4
        x0 = x.copy()
        if rand_start:
            x = np.clip(x + np.random.uniform(-eps, eps, size=x.shape), 0.0, 1.0)
        for _ in range(steps):
            grads = adapter.gradients(x, y)
            x = x + step_size * np.sign(grads)
            x = np.minimum(np.maximum(x, x0 - eps), x0 + eps)  # project L∞
            x = np.clip(x, 0.0, 1.0)
        return x


register(PGDAttack())
