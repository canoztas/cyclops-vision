# TODO: Post-MVP detector adapter base interface
from __future__ import annotations

from abc import ABC, abstractmethod


class DetectorAdapter(ABC):
    task: str  # "detect"

    @abstractmethod
    def predict(self, x):  # noqa: ANN001
        ...
