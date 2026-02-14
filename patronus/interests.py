from __future__ import annotations

import numpy as np

from patronus.config import Config
from patronus.embed import embed_batch


class InterestsSource:
    def get_context(self, config: Config) -> str:
        if not config.topics:
            return ""
        parts: list[str] = []
        for topic in config.topics.values():
            parts.append(f"{topic.name}: {topic.description}")
        return "\n\n".join(parts)

    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        if not config.topics:
            return None
        return load_interest_vectors(config)


def load_interest_vectors(config: Config) -> dict[str, np.ndarray]:
    keys = list(config.topics.keys())
    descriptions = [config.topics[k].description for k in keys]
    embeddings = embed_batch(descriptions, model=config.embedding.model)
    return dict(zip(keys, embeddings))
