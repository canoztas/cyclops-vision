from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf  # noqa: WPS433

        tf.random.set_seed(seed)
    except Exception:
        pass
