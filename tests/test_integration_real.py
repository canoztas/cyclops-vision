from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # type: ignore

from cyclops.adapters.keras_h5 import KerasH5Adapter
from cyclops.attacks.fgsm import FGSMAttack
from cyclops.datasets.folder import load_folder_dataset


requires_tf = pytest.mark.skipif(tf is None, reason="TensorFlow not available")


@requires_tf
def test_real_model_and_images(tmp_path: Path):
    # Use the fetch script to download assets
    import runpy

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "fetch_real_assets.py"
    assert script_path.exists(), "fetch_real_assets.py not found"
    runpy.run_path(str(script_path))

    model_path = repo_root / "mobilenetv2_imagenet.h5"
    data_root = repo_root / "examples" / "imagenet_like"

    assert model_path.exists(), "Model was not downloaded"
    assert data_root.exists(), "Dataset was not created"

    adapter = KerasH5Adapter(str(model_path), input_size=(224, 224))
    x, _, _ = load_folder_dataset(str(data_root), input_size=(224, 224))

    logits = adapter.predict(x)
    assert logits.shape[0] == x.shape[0]

    # Build 1000-class one-hot targets from model's own predictions
    preds = np.argmax(logits, axis=1)
    y1000 = np.eye(logits.shape[1], dtype=np.float32)[preds]

    x_adv = FGSMAttack().run(adapter, x, y1000, eps=2/255)
    logits_adv = adapter.predict(x_adv)
    assert logits_adv.shape == logits.shape

    # Clean up big model file to keep workspace light (optional)
    try:
        os.remove(model_path)
    except Exception:
        pass
