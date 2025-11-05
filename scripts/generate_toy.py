from __future__ import annotations

from pathlib import Path
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


def build_and_save_toy_model(path: str = "toy_model.h5", input_shape=(224, 224, 3), num_classes: int = 2) -> None:
    import tensorflow as tf
    set_seed(123)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(inputs)
    x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
    model = tf.keras.Model(inputs, outputs)
    model.save(path)
    print(f"Saved model to {path}")


def generate_dataset(root: str = "examples/imagenet_like", size=(224, 224), per_class: int = 6) -> None:
    from PIL import Image, ImageDraw
    set_seed(123)
    h, w = size
    root_path = Path(root)
    for cls in ["class0", "class1"]:
        (root_path / cls).mkdir(parents=True, exist_ok=True)
    # class0: centered white square on dark background
    for i in range(per_class):
        img = Image.new("RGB", (w, h), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        pad = 30 + (i % 3) * 5
        draw.rectangle([pad, pad, w - pad, h - pad], fill=(235, 235, 235))
        img.save(root_path / "class0" / f"img_{i}.png")
    # class1: diagonal white stripe on dark background
    for i in range(per_class):
        img = Image.new("RGB", (w, h), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        for d in range(-20, 20):
            draw.line([(0, (h//2)+d), (w, (h//2)+d-40)], fill=(235, 235, 235), width=2)
        img.save(root_path / "class1" / f"img_{i}.png")
    print(f"Wrote dataset under {root}")


if __name__ == "__main__":
    build_and_save_toy_model()
    generate_dataset()
    print("Done.")
