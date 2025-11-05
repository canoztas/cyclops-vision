from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def set_seed(seed: int = 123) -> None:
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def download_mobilenetv2_h5(path: str = "mobilenetv2_imagenet.h5") -> str:
    import tensorflow as tf
    set_seed(123)
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    model.save(path)
    print(f"Saved model to {path}")
    return path


def download_images() -> List[str]:
    import requests, os

    urls = [
        # Wikimedia Commons direct images (generally accessible)
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg",  # dog
        "https://upload.wikimedia.org/wikipedia/commons/4/40/Wild_Daisy.jpg",
    ]
    out_dir = Path(".cache/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    headers = {"User-Agent": "cyclops-vision/0.1 (https://example.com)"}
    for i, u in enumerate(urls):
        try:
            r = requests.get(u, headers=headers, timeout=30)
            r.raise_for_status()
            fname = out_dir / f"img_{i}.jpg"
            with open(fname, "wb") as f:
                f.write(r.content)
            paths.append(str(fname))
        except Exception as e:
            print(f"Failed to download {u}: {e}")
            continue
    if not paths:
        raise RuntimeError("Failed to download any images")
    print(f"Downloaded {len(paths)} images")
    return paths


def preprocess_for_mobilenet(img_path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    import tensorflow as tf
    from PIL import Image

    img = Image.open(img_path).convert("RGB").resize((size[1], size[0]))
    arr = np.asarray(img).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def build_dataset_from_predictions(model_path: str, image_paths: List[str], out_root: str = "examples/imagenet_like") -> None:
    import tensorflow as tf
    from PIL import Image

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)

    # Predict labels and split into two buckets by top-1 class id parity (even/odd) for balance
    buckets = {"class0": [], "class1": []}
    for p in image_paths:
        x = preprocess_for_mobilenet(p)
        logits = model(tf.convert_to_tensor(x[None, ...], dtype=tf.float32), training=False).numpy()[0]
        pred = int(np.argmax(logits))
        bucket = "class0" if (pred % 2 == 0) else "class1"
        buckets[bucket].append(p)

    # Ensure both classes have at least one image by duplicating if needed
    if not buckets["class0"] and buckets["class1"]:
        buckets["class0"].append(buckets["class1"][0])
    if not buckets["class1"] and buckets["class0"]:
        buckets["class1"].append(buckets["class0"][0])

    for cls, paths in buckets.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB").resize((224, 224))
            img.save(d / f"img_{i}.jpg")
    print(f"Wrote dataset to {out_root}")


if __name__ == "__main__":
    model_path = download_mobilenetv2_h5()
    image_paths = download_images()
    build_dataset_from_predictions(model_path, image_paths)
    print("Done.")
