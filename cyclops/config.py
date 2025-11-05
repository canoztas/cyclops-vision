from __future__ import annotations

from typing import List, Tuple

from cyclops.transforms import Compose, Resize, JPEG, Blur, Transform


def parse_size(token: str) -> Tuple[int, int]:
    # token like 224x224
    h_str, w_str = token.lower().split("x")
    return int(h_str), int(w_str)


def parse_transform_spec(spec: str) -> Transform:
    # Example: "resize=224x224,jpeg=95,blur=1"
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    transforms: List[Transform] = []
    for part in parts:
        if part.startswith("resize="):
            size = parse_size(part.split("=", 1)[1])
            transforms.append(Resize(size))
        elif part.startswith("jpeg="):
            q = int(part.split("=", 1)[1])
            transforms.append(JPEG(q))
        elif part.startswith("blur="):
            s = float(part.split("=", 1)[1])
            transforms.append(Blur(s))
    return Compose(transforms)
