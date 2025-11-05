from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np

from cyclops.utils.image import load_image


def load_folder_dataset(root: str | Path, input_size: Tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root_path = Path(root)
    class_names = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    images: List[np.ndarray] = []
    labels: List[int] = []
    for idx, cname in enumerate(class_names):
        for img_path in sorted((root_path / cname).glob("*.jpg")) + sorted((root_path / cname).glob("*.png")):
            arr = load_image(img_path)
            if input_size is not None:
                # resize via simple PIL
                from PIL import Image  # lazy import

                h, w = input_size
                arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((w, h))).astype(np.float32) / 255.0
            images.append(arr)
            labels.append(idx)
    if not images:
        raise ValueError(f"No images found under {root}")
    x = np.stack(images, axis=0).astype(np.float32)
    num_classes = len(class_names)
    y_indices = np.array(labels, dtype=np.int64)
    y_onehot = np.eye(num_classes, dtype=np.float32)[y_indices]
    return x, y_onehot, class_names
