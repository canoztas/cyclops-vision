from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2

from .base import Transform


class Resize(Transform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (N,H,W,C) in [0,1]
        h, w = self.size
        out = np.empty((x.shape[0], h, w, x.shape[3]), dtype=np.float32)
        for i in range(x.shape[0]):
            img = (x[i] * 255.0).astype(np.uint8)
            resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            out[i] = resized.astype(np.float32) / 255.0
        return out
