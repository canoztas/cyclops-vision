from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from cyclops.adapters.base import ModelAdapter


class Attack(ABC):
    name: str
    requires_gradients: bool = True

    @abstractmethod
    def run(self, adapter: ModelAdapter, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> np.ndarray:
        ...


ATTACKS: Dict[str, Attack] = {}

def register(attack: Attack) -> None:
    ATTACKS[attack.name] = attack
