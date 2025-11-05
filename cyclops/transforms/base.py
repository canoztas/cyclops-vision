from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...
