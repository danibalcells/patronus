from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from patronus.db import Item, serialize_embedding
from patronus.rank import ScoredItem, _cosine_similarity, _days_old, rank_unread, select_digest


def _make_item(
    url: str = "https://example.com/1",
    embedding: np.ndarray | None = None,
    timestamp: str | None = None,
    topic_cluster: str | None = None,
) -> Item:
    item = Item(
        id="test-id",
        url=url,
        source_type="rss",
        timestamp=timestamp,
        topic_cluster=topic_cluster,
    )
    if embedding is not None:
        item.embedding = serialize_embedding(embedding)
    return item


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero_returns_zero(self) -> None:
        z = np.zeros(3, dtype=np.float32)
        assert _cosine_similarity(z, z) == 0.0


class TestDaysOld:
    def test_none_timestamp_returns_default(self) -> None:
        assert _days_old(None) == 30.0

    def test_invalid_format_returns_default(self) -> None:
        assert _days_old("not-a-date") == 30.0

    def test_recent_timestamp(self) -> None:
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert _days_old(ts) < 1.0

    def test_old_timestamp(self) -> None:
        assert _days_old("2020-01-01T00:00:00Z") > 365


class TestRankUnread:
    def test_sorts_by_score_descending(self) -> None:
        high_sim = _unit_vec(1.0, 0.0, 0.0)
        low_sim = _unit_vec(0.0, 1.0, 0.0)
        centroid = _unit_vec(1.0, 0.0, 0.0)

        items = [
            _make_item(url="https://low.com", embedding=low_sim, timestamp="2026-02-13T00:00:00Z"),
            _make_item(url="https://high.com", embedding=high_sim, timestamp="2026-02-13T00:00:00Z"),
        ]

        scored = rank_unread(items, {"topic": centroid})
        assert scored[0].item.url == "https://high.com"
        assert scored[1].item.url == "https://low.com"
        assert scored[0].score > scored[1].score

    def test_skips_items_without_embeddings(self) -> None:
        centroid = _unit_vec(1.0, 0.0)
        items = [
            _make_item(url="https://no-emb.com", embedding=None),
            _make_item(url="https://has-emb.com", embedding=_unit_vec(1.0, 0.0)),
        ]
        scored = rank_unread(items, {"t": centroid})
        assert len(scored) == 1
        assert scored[0].item.url == "https://has-emb.com"

    def test_picks_best_topic(self) -> None:
        item_emb = _unit_vec(0.0, 1.0)
        centroids = {
            "topic_a": _unit_vec(1.0, 0.0),
            "topic_b": _unit_vec(0.0, 1.0),
        }
        items = [_make_item(embedding=item_emb, timestamp="2026-02-13T00:00:00Z")]
        scored = rank_unread(items, centroids)
        assert scored[0].matched_topic == "topic_b"

    def test_recency_boost(self) -> None:
        emb = _unit_vec(1.0, 0.0)
        centroid = _unit_vec(1.0, 0.0)

        recent = _make_item(url="https://recent.com", embedding=emb, timestamp="2026-02-13T00:00:00Z")
        old = _make_item(url="https://old.com", embedding=emb, timestamp="2025-01-01T00:00:00Z")

        scored = rank_unread([old, recent], {"t": centroid})
        assert scored[0].item.url == "https://recent.com"
        assert scored[0].score > scored[1].score
        assert scored[0].raw_similarity == pytest.approx(scored[1].raw_similarity, abs=1e-6)

    def test_empty_items(self) -> None:
        assert rank_unread([], {"t": _unit_vec(1.0)}) == []


class TestSelectDigest:
    def _make_scored(self, topic: str, score: float) -> ScoredItem:
        item = _make_item(url=f"https://{topic}-{score}.com")
        return ScoredItem(item=item, score=score, matched_topic=topic, raw_similarity=score)

    def test_respects_size_limit(self) -> None:
        topics = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        items = [self._make_scored(topics[i], 0.9 - i * 0.01) for i in range(10)]
        selected = select_digest(items, size=5)
        assert len(selected) == 5

    def test_respects_max_per_topic(self) -> None:
        items = [
            self._make_scored("ml", 0.95),
            self._make_scored("ml", 0.90),
            self._make_scored("ml", 0.85),
            self._make_scored("ml", 0.80),
            self._make_scored("phil", 0.75),
        ]
        selected = select_digest(items, size=7, max_per_topic=3)
        ml_count = sum(1 for s in selected if s.matched_topic == "ml")
        assert ml_count == 3
        assert len(selected) == 4

    def test_diverse_selection(self) -> None:
        items = [
            self._make_scored("ml", 0.95),
            self._make_scored("ml", 0.93),
            self._make_scored("ml", 0.91),
            self._make_scored("ml", 0.89),
            self._make_scored("phil", 0.88),
            self._make_scored("phil", 0.86),
            self._make_scored("spain", 0.84),
        ]
        selected = select_digest(items, size=7, max_per_topic=3)
        topics = [s.matched_topic for s in selected]
        assert topics.count("ml") == 3
        assert topics.count("phil") == 2
        assert topics.count("spain") == 1
        assert len(selected) == 6

    def test_empty_input(self) -> None:
        assert select_digest([]) == []

    def test_fewer_items_than_size(self) -> None:
        items = [self._make_scored("a", 0.9), self._make_scored("b", 0.8)]
        selected = select_digest(items, size=7)
        assert len(selected) == 2

    def test_maintains_score_order(self) -> None:
        items = [
            self._make_scored("a", 0.9),
            self._make_scored("b", 0.8),
            self._make_scored("c", 0.7),
        ]
        selected = select_digest(items, size=3)
        assert selected[0].score > selected[1].score > selected[2].score
