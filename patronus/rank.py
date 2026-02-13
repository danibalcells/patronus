from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from patronus.db import Item, deserialize_embedding

_DEFAULT_DAYS = 30.0


@dataclass
class ScoredItem:
    item: Item
    score: float
    matched_topic: str
    raw_similarity: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _days_old(timestamp: Optional[str]) -> float:
    if not timestamp:
        return _DEFAULT_DAYS
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return _DEFAULT_DAYS


def rank_unread(
    items: list[Item],
    interest_vectors: dict[str, np.ndarray],
    *,
    decay_base: float = 0.998,
) -> list[ScoredItem]:
    scored: list[ScoredItem] = []

    for item in items:
        if item.embedding is None:
            continue

        embedding = deserialize_embedding(item.embedding)

        best_similarity = -1.0
        best_topic = ""

        for topic_key, centroid in interest_vectors.items():
            sim = _cosine_similarity(embedding, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_topic = topic_key

        days = _days_old(item.timestamp)
        final_score = best_similarity * (decay_base ** days)

        scored.append(
            ScoredItem(
                item=item,
                score=final_score,
                matched_topic=best_topic,
                raw_similarity=best_similarity,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def select_digest(
    scored_items: list[ScoredItem],
    *,
    size: int = 7,
    max_per_topic: dict[str, int] | int = 3,
) -> list[ScoredItem]:
    selected: list[ScoredItem] = []
    topic_counts: dict[str, int] = {}

    for item in scored_items:
        if len(selected) >= size:
            break

        if isinstance(max_per_topic, dict):
            limit = max_per_topic.get(item.matched_topic, 1)
        else:
            limit = max_per_topic

        count = topic_counts.get(item.matched_topic, 0)
        if count >= limit:
            continue

        selected.append(item)
        topic_counts[item.matched_topic] = count + 1

    return selected
