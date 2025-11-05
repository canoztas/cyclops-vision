from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional import for packaging
    tf = None  # type: ignore

from .base import ModelAdapter


class KerasH5Adapter(ModelAdapter):
    def __init__(self, path: str, task: str = "classify", input_size: Tuple[int, int] = (224, 224)):
        if tf is None:
            raise RuntimeError("TensorFlow is required for KerasH5Adapter. Install tensorflow-cpu.")
        self.model = tf.keras.models.load_model(path, compile=False)
        self.task = task
        self.input_size = input_size
        self._loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        return self.model(x_tf, training=False).numpy()

    def gradients(self, x: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_onehot, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            logits = self.model(x_tf, training=False)
            loss = self._loss(y_tf, logits)
        g = tape.gradient(loss, x_tf)
        return g.numpy()
