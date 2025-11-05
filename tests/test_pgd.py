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
from cyclops.attacks.pgd import PGDAttack
from cyclops.metrics.classify import asr


pytestmark = pytest.mark.skipif(tf is None, reason="TensorFlow not available")


def build_toy_model(input_shape=(16, 16, 3), num_classes: int = 2):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def test_pgd_runs_and_asr_non_negative():
    model = build_toy_model()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "toy.h5")
        model.save(path)
        adapter = KerasH5Adapter(path, input_size=(16, 16))
        x = np.random.rand(8, 16, 16, 3).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.random.randint(0, 2, size=(8,))]
        logits_clean = adapter.predict(x)
        x_adv = PGDAttack().run(adapter, x, y, eps=4/255, steps=5)
        logits_adv = adapter.predict(x_adv)
        assert asr(logits_clean, logits_adv) >= 0.0
