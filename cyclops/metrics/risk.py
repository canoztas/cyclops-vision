from __future__ import annotations

from typing import Dict

# Documented constants for weights by transform severity (example values)
TRANSFORM_WEIGHTS: Dict[str, float] = {
    "jpeg95": 0.8,
    "jpeg80": 1.0,
    "blur0": 0.7,
    "blur1": 0.9,
}


def risk_score(weighted_accuracy_under_attack: float) -> float:
    # Scale to 0-100 where higher is riskier
    return 100.0 - (weighted_accuracy_under_attack * 100.0)
