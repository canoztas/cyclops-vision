# TODO: Post-MVP PyTorch adapter skeleton
from __future__ import annotations

from .base import ModelAdapter


class TorchAdapter(ModelAdapter):
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001, ANN002
        raise NotImplementedError("Torch adapter not implemented (post-MVP)")

    def predict(self, x):  # type: ignore[override]
        raise NotImplementedError

    def gradients(self, x, y_onehot):  # type: ignore[override]
        raise NotImplementedError
