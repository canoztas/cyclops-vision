from __future__ import annotations

import numpy as np
from cyclops.attacks.base import Attack, register
from cyclops.adapters.base import ModelAdapter


class FGSMAttack(Attack):
    name = "fgsm"
    requires_gradients = True

    def run(
        self,
        adapter: ModelAdapter,
        x: np.ndarray,
        y: np.ndarray,
        eps: float = 2 / 255,
        **_: dict,
    ) -> np.ndarray:
        grads = adapter.gradients(x, y)
        x_adv = np.clip(x + eps * np.sign(grads), 0.0, 1.0)
        return x_adv


register(FGSMAttack())
