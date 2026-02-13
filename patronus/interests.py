from __future__ import annotations

import numpy as np

from patronus.config import Config
from patronus.embed import embed_batch


def load_interest_vectors(config: Config) -> dict[str, np.ndarray]:
    keys = list(config.topics.keys())
    descriptions = [config.topics[k].description for k in keys]
    embeddings = embed_batch(descriptions, model=config.embedding.model)
    return dict(zip(keys, embeddings))
