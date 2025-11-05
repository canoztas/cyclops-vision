from __future__ import annotations

import numpy as np


def top1_accuracy(y_true_onehot: np.ndarray, logits: np.ndarray) -> float:
    true_idx = np.argmax(y_true_onehot, axis=1)
    pred_idx = np.argmax(logits, axis=1)
    return float(np.mean(true_idx == pred_idx))


def asr(clean_logits: np.ndarray, adv_logits: np.ndarray) -> float:
    clean_idx = np.argmax(clean_logits, axis=1)
    adv_idx = np.argmax(adv_logits, axis=1)
    return float(np.mean(clean_idx != adv_idx))
