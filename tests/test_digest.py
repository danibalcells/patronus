from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from patronus.config import (
    Config,
    DigestConfig,
    EmbeddingConfig,
    PollingConfig,
    SummarizationConfig,
    TelegramConfig,
    TopicConfig,
)
from patronus.db import Database, Item, serialize_embedding
from patronus.digest import (
    Digest,
    DigestItem,
    _apply_repeat_penalty,
    _escape_markdown_v2,
    format_telegram,
    generate_digest,
)
from patronus.rank import ScoredItem


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_config(**overrides: object) -> Config:
    return Config(
        digest=DigestConfig(**(overrides.get("digest", {}))),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        summarization=SummarizationConfig(),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="Technical AI/ML", description="ML research"),
            "phil": TopicConfig(name="Philosophy", description="Philosophy of mind"),
        },
    )


def _make_item(
    item_id: str = "item-1",
    url: str = "https://example.com/1",
    title: str = "Test Article",
    source: str = "Test Blog",
    text: str = "Article content here.",
    embedding: np.ndarray | None = None,
    timestamp: str | None = "2026-02-13T00:00:00Z",
    digest_history: str = "[]",
) -> Item:
    item = Item(
        id=item_id,
        url=url,
        source_type="rss",
        title=title,
        source=source,
        text=text,
        timestamp=timestamp,
        digest_history=digest_history,
    )
    if embedding is not None:
        item.embedding = serialize_embedding(embedding)
    return item


def _make_scored(
    item: Item | None = None,
    score: float = 0.9,
    matched_topic: str = "ml",
) -> ScoredItem:
    if item is None:
        item = _make_item(embedding=_unit_vec(1.0, 0.0))
    return ScoredItem(
        item=item,
        score=score,
        matched_topic=matched_topic,
        raw_similarity=score,
    )


class TestApplyRepeatPenalty:
    def test_no_history_no_change(self) -> None:
        scored = [_make_scored(score=0.9)]
        result = _apply_repeat_penalty(scored, repeat_penalty=0.85)
        assert result[0].score == pytest.approx(0.9)

    def test_single_digest_applies_once(self) -> None:
        item = _make_item(digest_history=json.dumps(["2026-02-12"]))
        scored = [_make_scored(item=item, score=0.9)]
        result = _apply_repeat_penalty(scored, repeat_penalty=0.85)
        assert result[0].score == pytest.approx(0.9 * 0.85)

    def test_two_digests_applies_twice(self) -> None:
        item = _make_item(digest_history=json.dumps(["2026-02-11", "2026-02-12"]))
        scored = [_make_scored(item=item, score=0.9)]
        result = _apply_repeat_penalty(scored, repeat_penalty=0.85)
        assert result[0].score == pytest.approx(0.9 * 0.85 ** 2)

    def test_reorders_after_penalty(self) -> None:
        fresh = _make_item(item_id="fresh", url="https://fresh.com")
        stale = _make_item(
            item_id="stale",
            url="https://stale.com",
            digest_history=json.dumps(["2026-02-12"]),
        )
        scored = [
            _make_scored(item=stale, score=0.91),
            _make_scored(item=fresh, score=0.90),
        ]
        result = _apply_repeat_penalty(scored, repeat_penalty=0.85)
        assert result[0].item.id == "fresh"


class TestEscapeMarkdownV2:
    def test_escapes_special_chars(self) -> None:
        assert _escape_markdown_v2("hello_world") == "hello\\_world"
        assert _escape_markdown_v2("a*b") == "a\\*b"
        assert _escape_markdown_v2("1.2") == "1\\.2"

    def test_plain_text_unchanged(self) -> None:
        assert _escape_markdown_v2("hello world") == "hello world"


class TestFormatTelegram:
    def _make_digest(self, items: list[DigestItem] | None = None) -> Digest:
        if items is None:
            items = []
        return Digest(items=items, generated_at="2026-02-13T08:00:00Z")

    def test_empty_digest(self) -> None:
        config = _make_config()
        result = format_telegram(self._make_digest(), config)
        assert "No items" in result

    def test_contains_date(self) -> None:
        config = _make_config()
        item = DigestItem(scored_item=_make_scored(), summary="A summary.")
        result = format_telegram(self._make_digest([item]), config)
        assert "2026\\-02\\-13" in result

    def test_groups_by_topic(self) -> None:
        config = _make_config()
        ml_item = DigestItem(
            scored_item=_make_scored(
                item=_make_item(item_id="ml-1", url="https://ml.com", title="ML Paper"),
                matched_topic="ml",
            ),
            summary="ML summary.",
        )
        phil_item = DigestItem(
            scored_item=_make_scored(
                item=_make_item(item_id="p-1", url="https://phil.com", title="Phil Essay"),
                matched_topic="phil",
            ),
            summary="Phil summary.",
        )
        result = format_telegram(self._make_digest([ml_item, phil_item]), config)
        assert "Technical AI/ML" in result
        assert "Philosophy" in result

    def test_item_has_title_link_source_summary(self) -> None:
        config = _make_config()
        item = DigestItem(
            scored_item=_make_scored(
                item=_make_item(title="Great Article", source="Some Blog", url="https://example.com"),
            ),
            summary="This is relevant.",
        )
        result = format_telegram(self._make_digest([item]), config)
        assert "Great Article" in result
        assert "https://example.com" in result
        assert "Some Blog" in result
        assert "This is relevant" in result


class TestDigestDataclass:
    def test_item_count(self) -> None:
        d = Digest(items=[DigestItem(scored_item=_make_scored(), summary="s")], generated_at="now")
        assert d.item_count == 1

    def test_empty_item_count(self) -> None:
        d = Digest(items=[], generated_at="now")
        assert d.item_count == 0


class TestGenerateDigest:
    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_full_pipeline(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        centroid = _unit_vec(1.0, 0.0, 0.0)
        mock_interests.return_value = {"ml": centroid}
        mock_summarize.return_value = "Generated summary."

        emb = _unit_vec(0.9, 0.1, 0.0)
        db.add_item(
            url="https://example.com/article",
            source_type="rss",
            title="Test Article",
            source="Test Blog",
            text="Some text",
            embedding=emb,
            timestamp="2026-02-13T00:00:00Z",
        )

        digest = generate_digest(config, db)

        assert digest.item_count == 1
        assert digest.items[0].summary == "Generated summary."
        assert digest.items[0].scored_item.matched_topic == "ml"

        mock_interests.assert_called_once_with(config)
        mock_summarize.assert_called_once()

        digests = db.get_latest_digests(1)
        assert len(digests) == 1
        assert digests[0].item_count == 1

        digest_items = db.get_digest_items(digests[0].id)
        assert len(digest_items) == 1
        assert digest_items[0].summary == "Generated summary."

        db.close()

    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_empty_unread(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        mock_interests.return_value = {"ml": _unit_vec(1.0, 0.0)}

        digest = generate_digest(config, db)

        assert digest.item_count == 0
        mock_summarize.assert_not_called()
        db.close()

    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_summarize_failure_does_not_crash(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        centroid = _unit_vec(1.0, 0.0)
        mock_interests.return_value = {"ml": centroid}
        mock_summarize.side_effect = RuntimeError("API down")

        emb = _unit_vec(1.0, 0.0)
        db.add_item(
            url="https://example.com/fail",
            source_type="rss",
            title="Failing Article",
            text="text",
            embedding=emb,
            timestamp="2026-02-13T00:00:00Z",
        )

        digest = generate_digest(config, db)

        assert digest.item_count == 1
        assert digest.items[0].summary == ""
        db.close()

    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_records_digest_history(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        centroid = _unit_vec(1.0, 0.0)
        mock_interests.return_value = {"ml": centroid}
        mock_summarize.return_value = "summary"

        emb = _unit_vec(1.0, 0.0)
        item_id = db.add_item(
            url="https://example.com/hist",
            source_type="rss",
            title="Article",
            text="text",
            embedding=emb,
            timestamp="2026-02-13T00:00:00Z",
        )

        generate_digest(config, db)

        item = db.get_item(item_id)
        assert item is not None
        history = json.loads(item.digest_history)
        assert len(history) == 1
        db.close()

    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_repeat_penalty_applied(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(digest={"size": 1, "max_per_topic": 1, "repeat_penalty": 0.5})

        centroid = _unit_vec(1.0, 0.0)
        mock_interests.return_value = {"ml": centroid}
        mock_summarize.return_value = "summary"

        emb = _unit_vec(1.0, 0.0)
        db.add_item(
            url="https://stale.com",
            source_type="rss",
            title="Stale",
            text="text",
            embedding=emb,
            timestamp="2026-02-13T00:00:00Z",
        )
        item_stale = db.get_item_by_url("https://stale.com")
        assert item_stale is not None
        db.update_digest_history(item_stale.id, "2026-02-12")

        emb2 = _unit_vec(0.95, 0.05)
        db.add_item(
            url="https://fresh.com",
            source_type="rss",
            title="Fresh",
            text="text",
            embedding=emb2,
            timestamp="2026-02-13T00:00:00Z",
        )

        digest = generate_digest(config, db)

        assert digest.item_count == 1
        assert digest.items[0].scored_item.item.url == "https://fresh.com"
        db.close()
