from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional import for packaging
    ort = None  # type: ignore

from .base import ModelAdapter


class ONNXAdapter(ModelAdapter):
    def __init__(self, path: str, task: str = "classify", input_size: Tuple[int, int] = (224, 224), nchw: bool | None = None):
        if ort is None:
            raise RuntimeError("onnxruntime is required for ONNXAdapter. Install onnxruntime.")
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])  # simple CPU default
        self.task = task
        self.input_size = input_size
        self._nchw = nchw
        self._in_name = self.session.get_inputs()[0].name
        self._out_name = self.session.get_outputs()[0].name

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Convert NHWC (default) to NCHW if model expects it
        input_array = x
        if self._nchw is None:
            # Heuristic: if expected shape channels-first
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and (shape[1] in (1, 3)):
                self._nchw = True
            else:
                self._nchw = False
        if self._nchw:
            input_array = np.transpose(x, (0, 3, 1, 2))
        outputs = self.session.run([self._out_name], {self._in_name: input_array.astype(np.float32)})
        return outputs[0]

    def gradients(self, x: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError("ONNXAdapter does not support gradients in MVP.")
