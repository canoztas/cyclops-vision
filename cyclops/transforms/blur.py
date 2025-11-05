from __future__ import annotations

import numpy as np
import cv2

from .base import Transform


class Blur(Transform):
    def __init__(self, sigma: float = 0.0):
        self.sigma = float(sigma)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.sigma <= 0:
            return x
        out = np.empty_like(x)
        ksize = max(1, int(self.sigma * 3) * 2 + 1)
        for i in range(x.shape[0]):
            img = (x[i] * 255.0).astype(np.uint8)
            blurred = cv2.GaussianBlur(img, (ksize, ksize), self.sigma)
            out[i] = blurred.astype(np.float32) / 255.0
        return out
