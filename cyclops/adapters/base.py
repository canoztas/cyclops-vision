from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ModelAdapter(ABC):
    task: str  # "classify" (MVP)
    input_size: Tuple[int, int]

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def gradients(self, x: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
        """Return ∂loss/∂x; raise NotImplementedError if not supported."""
        ...
