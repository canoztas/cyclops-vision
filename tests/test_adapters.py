from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore

from cyclops.adapters.keras_h5 import KerasH5Adapter


pytestmark = pytest.mark.skipif(tf is None, reason="TensorFlow not available")


def build_toy_model(input_shape=(32, 32, 3), num_classes: int = 2):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(4, 3, activation="relu")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def test_adapter_predict_and_gradients():
    model = build_toy_model()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "toy.h5")
        model.save(path)
        adapter = KerasH5Adapter(path, input_size=(32, 32))
        x = np.random.rand(4, 32, 32, 3).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.random.randint(0, 2, size=(4,))]
        logits = adapter.predict(x)
        assert logits.shape == (4, 2)
        grads = adapter.gradients(x, y)
        assert grads.shape == x.shape
