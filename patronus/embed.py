from __future__ import annotations

import numpy as np


def embed_text(text: str) -> np.ndarray:
    raise NotImplementedError


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    raise NotImplementedError
