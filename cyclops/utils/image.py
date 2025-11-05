from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
    return arr


def save_image(path: str | Path, array01: np.ndarray) -> None:
    arr = (np.clip(array01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def make_grid(pairs: list[Tuple[np.ndarray, np.ndarray]], max_items: int = 8) -> np.ndarray:
    # Simple horizontal concatenation per pair, then vertical stack
    items = pairs[:max_items]
    rows = [np.concatenate([a, b], axis=1) for a, b in items]
    return np.concatenate(rows, axis=0) if rows else np.zeros((1, 1, 3), dtype=np.uint8)
