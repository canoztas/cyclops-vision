from __future__ import annotations

from typing import Iterable, List
import numpy as np

from .base import Transform


class Compose(Transform):
    def __init__(self, transforms: Iterable[Transform]):
        self.transforms: List[Transform] = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x
        for t in self.transforms:
            out = t(out)
        return out
