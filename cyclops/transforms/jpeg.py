from __future__ import annotations

import io
import numpy as np
from PIL import Image

from .base import Transform


class JPEG(Transform):
    def __init__(self, quality: int = 95):
        self.quality = int(quality)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            img = (x[i] * 255.0).astype(np.uint8)
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=self.quality, subsampling=0)
            buf.seek(0)
            dec = Image.open(buf).convert("RGB")
            out[i] = np.asarray(dec, dtype=np.uint8).astype(np.float32) / 255.0
        return out
