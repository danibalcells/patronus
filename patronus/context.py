from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from patronus.config import Config

logger = logging.getLogger(__name__)


@runtime_checkable
class PersonalizationSource(Protocol):
    def get_context(self, config: Config) -> str: ...
    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None: ...


@dataclass
class Context:
    prose: str = ""
    vectors: dict[str, np.ndarray] = field(default_factory=dict)


def merge_sources(
    sources: list[PersonalizationSource],
    config: Config,
    notion_force_refresh: bool = False,
) -> Context:
    prose_parts: list[str] = []
    vectors: dict[str, np.ndarray] = {}

    for source in sources:
        try:
            from patronus.notion import NotionSource
            if isinstance(source, NotionSource):
                context_str = source.get_context(config, force_refresh=notion_force_refresh)
            else:
                context_str = source.get_context(config)
            if context_str:
                prose_parts.append(context_str)
        except Exception:
            logger.warning("Failed to get context from %s", type(source).__name__, exc_info=True)
            continue

        try:
            vecs = source.get_interest_vectors(config)
            if vecs:
                vectors.update(vecs)
        except Exception:
            logger.warning("Failed to get interest vectors from %s", type(source).__name__, exc_info=True)
            continue

    return Context(prose="\n\n".join(prose_parts), vectors=vectors)
