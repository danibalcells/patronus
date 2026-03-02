from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from patronus.db import Database, Feed, Item, deserialize_embedding, serialize_embedding


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "test.sqlite3"))


@pytest.fixture()
def sample_embedding() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(1536).astype(np.float32)


class TestEmbeddingSerialization:
    def test_round_trip(self, sample_embedding: np.ndarray) -> None:
        blob = serialize_embedding(sample_embedding)
        recovered = deserialize_embedding(blob)
        assert np.array_equal(sample_embedding, recovered)

    def test_float64_cast_to_float32(self) -> None:
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        blob = serialize_embedding(emb)
        recovered = deserialize_embedding(blob)
        assert recovered.dtype == np.float32
        assert np.allclose(emb, recovered)

    def test_empty_embedding(self) -> None:
        emb = np.array([], dtype=np.float32)
        blob = serialize_embedding(emb)
        recovered = deserialize_embedding(blob)
        assert recovered.shape == (0,)


class TestDatabaseInit:
    def test_creates_tables(self, db: Database) -> None:
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "items" in tables
        assert "feeds" in tables

    def test_context_manager(self, tmp_path: Path) -> None:
        with Database(str(tmp_path / "ctx.sqlite3")) as db:
            db.add_feed(url="https://example.com/feed")


class TestItems:
    def test_add_and_get(self, db: Database) -> None:
        item_id = db.add_item(
            url="https://example.com/1",
            source_type="rss",
            title="Test",
            author="Author",
            source="Blog",
        )
        item = db.get_item(item_id)
        assert item is not None
        assert isinstance(item, Item)
        assert item.title == "Test"
        assert item.author == "Author"
        assert item.source == "Blog"
        assert item.source_type == "rss"
        assert item.read is False
        assert item.ingested_at is not None

    def test_add_minimal(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/min", source_type="manual")
        item = db.get_item(item_id)
        assert item is not None
        assert item.title is None
        assert item.embedding is None

    def test_get_by_url(self, db: Database) -> None:
        url = "https://example.com/dedup"
        db.add_item(url=url, source_type="rss")
        assert db.get_item_by_url(url) is not None
        assert db.get_item_by_url("https://nonexistent.com") is None

    def test_duplicate_url_rejected(self, db: Database) -> None:
        db.add_item(url="https://example.com/dup", source_type="rss")
        with pytest.raises(IntegrityError):
            db.add_item(url="https://example.com/dup", source_type="manual")

    def test_mark_read(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/read", source_type="rss")
        assert db.get_item(item_id).read is False
        db.mark_read(item_id)
        assert db.get_item(item_id).read is True

    def test_unread_items_ordered_by_timestamp(self, db: Database) -> None:
        db.add_item(url="https://example.com/old", source_type="rss", timestamp="2026-01-01T00:00:00Z")
        db.add_item(url="https://example.com/new", source_type="rss", timestamp="2026-02-01T00:00:00Z")
        db.add_item(url="https://example.com/mid", source_type="rss", timestamp="2026-01-15T00:00:00Z")

        unread = db.get_unread_items()
        assert len(unread) == 3
        assert unread[0].url == "https://example.com/new"
        assert unread[1].url == "https://example.com/mid"
        assert unread[2].url == "https://example.com/old"

    def test_unread_excludes_read(self, db: Database) -> None:
        iid = db.add_item(url="https://example.com/a", source_type="rss")
        db.add_item(url="https://example.com/b", source_type="rss")
        db.mark_read(iid)
        assert len(db.get_unread_items()) == 1

    def test_embedding_store_and_retrieve(self, db: Database, sample_embedding: np.ndarray) -> None:
        item_id = db.add_item(url="https://example.com/emb1", source_type="arxiv")
        assert db.get_embedding(item_id) is None

        db.store_embedding(item_id, sample_embedding)
        recovered = db.get_embedding(item_id)
        assert recovered is not None
        assert np.array_equal(sample_embedding, recovered)

    def test_embedding_at_insert(self, db: Database, sample_embedding: np.ndarray) -> None:
        item_id = db.add_item(
            url="https://example.com/emb2",
            source_type="rss",
            embedding=sample_embedding,
        )
        recovered = db.get_embedding(item_id)
        assert np.array_equal(sample_embedding, recovered)

    def test_digest_history(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/digest", source_type="rss")
        item = db.get_item(item_id)
        assert json.loads(item.digest_history) == []

        db.update_digest_history(item_id, "2026-02-10")
        db.update_digest_history(item_id, "2026-02-11")
        item = db.get_item(item_id)
        assert json.loads(item.digest_history) == ["2026-02-10", "2026-02-11"]

    def test_digest_history_nonexistent_item(self, db: Database) -> None:
        db.update_digest_history("nonexistent", "2026-02-10")

    def test_item_type_defaults_to_article(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/post", source_type="rss")
        item = db.get_item(item_id)
        assert item is not None
        assert item.item_type == "article"

    def test_item_type_stored(self, db: Database) -> None:
        item_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )
        item = db.get_item(item_id)
        assert item is not None
        assert item.item_type == "tweet"

    def test_source_item_id_defaults_to_none(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/post", source_type="rss")
        item = db.get_item(item_id)
        assert item is not None
        assert item.source_item_id is None

    def test_source_item_id_stored(self, db: Database) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )
        child_id = db.add_item(
            url="https://arxiv.org/abs/2026.12345",
            source_type="rss",
            item_type="paper",
            source_item_id=parent_id,
        )
        child = db.get_item(child_id)
        assert child is not None
        assert child.source_item_id == parent_id

    def test_source_item_id_fk_enforced(self, db: Database) -> None:
        with pytest.raises(IntegrityError):
            db.add_item(
                url="https://example.com/orphan",
                source_type="rss",
                source_item_id="nonexistent-id",
            )


class TestFeeds:
    def test_add_and_get(self, db: Database) -> None:
        fid = db.add_feed(url="https://example.com/feed", name="Example", category="tech")
        feed = db.get_feed(fid)
        assert feed is not None
        assert isinstance(feed, Feed)
        assert feed.name == "Example"
        assert feed.category == "tech"
        assert feed.active is True
        assert feed.last_polled is None

    def test_get_by_url(self, db: Database) -> None:
        db.add_feed(url="https://example.com/feed2")
        assert db.get_feed_by_url("https://example.com/feed2") is not None
        assert db.get_feed_by_url("https://nonexistent.com") is None

    def test_duplicate_url_rejected(self, db: Database) -> None:
        db.add_feed(url="https://example.com/dup")
        with pytest.raises(IntegrityError):
            db.add_feed(url="https://example.com/dup")

    def test_active_feeds(self, db: Database) -> None:
        db.add_feed(url="https://example.com/a", name="A")
        fid_b = db.add_feed(url="https://example.com/b", name="B")
        db.add_feed(url="https://example.com/c", name="C")
        db.deactivate_feed(fid_b)

        active = db.get_active_feeds()
        assert len(active) == 2
        urls = [f.url for f in active]
        assert "https://example.com/b" not in urls

    def test_update_poll(self, db: Database) -> None:
        fid = db.add_feed(url="https://example.com/poll")
        db.update_feed_poll(fid, last_polled="2026-02-12T10:00:00Z", etag='"abc"', last_modified="Wed, 12 Feb 2026")
        feed = db.get_feed(fid)
        assert feed.last_polled == "2026-02-12T10:00:00Z"
        assert feed.etag == '"abc"'
        assert feed.last_modified == "Wed, 12 Feb 2026"

    def test_update_poll_defaults_to_now(self, db: Database) -> None:
        fid = db.add_feed(url="https://example.com/poll2")
        db.update_feed_poll(fid)
        feed = db.get_feed(fid)
        assert feed.last_polled is not None

    def test_seed_from_file(self, db: Database, tmp_path: Path) -> None:
        feed_file = tmp_path / "feeds.txt"
        feed_file.write_text("https://a.com/feed\nhttps://b.com/feed\n\nhttps://c.com/feed\n")

        count = db.seed_feeds_from_file(str(feed_file))
        assert count == 3
        assert len(db.get_active_feeds()) == 3

    def test_seed_skips_duplicates(self, db: Database, tmp_path: Path) -> None:
        db.add_feed(url="https://a.com/feed")
        feed_file = tmp_path / "feeds.txt"
        feed_file.write_text("https://a.com/feed\nhttps://b.com/feed\n")

        count = db.seed_feeds_from_file(str(feed_file))
        assert count == 1
        assert len(db.get_active_feeds()) == 2


class TestRecentlyDigestedItems:
    def _save_digest_with_item(self, db: Database, item_id: str, days_ago: int = 0) -> None:
        generated_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.save_digest(
            generated_at=generated_at,
            item_count=1,
            formatted_text=None,
            items=[{"item_id": item_id, "summary": "s", "score": 1.0, "matched_topic": ""}],
        )

    def test_empty_when_no_digests(self, db: Database) -> None:
        db.add_item(url="https://example.com/a", source_type="rss")
        assert db.get_recently_digested_item_ids() == set()

    def test_item_in_recent_digest_is_returned(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/a", source_type="rss")
        self._save_digest_with_item(db, item_id, days_ago=0)
        assert item_id in db.get_recently_digested_item_ids()

    def test_item_digested_within_lookback_is_returned(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/a", source_type="rss")
        self._save_digest_with_item(db, item_id, days_ago=29)
        assert item_id in db.get_recently_digested_item_ids()

    def test_item_digested_outside_lookback_not_returned(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/a", source_type="rss")
        self._save_digest_with_item(db, item_id, days_ago=31)
        assert item_id not in db.get_recently_digested_item_ids()

    def test_only_recent_items_returned(self, db: Database) -> None:
        recent_id = db.add_item(url="https://example.com/recent", source_type="rss")
        old_id = db.add_item(url="https://example.com/old", source_type="rss")
        self._save_digest_with_item(db, recent_id, days_ago=5)
        self._save_digest_with_item(db, old_id, days_ago=45)

        result = db.get_recently_digested_item_ids()
        assert recent_id in result
        assert old_id not in result

    def test_item_returned_once_even_if_digested_multiple_times_recently(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/a", source_type="rss")
        self._save_digest_with_item(db, item_id, days_ago=1)
        self._save_digest_with_item(db, item_id, days_ago=2)
        result = db.get_recently_digested_item_ids()
        assert item_id in result
        assert list(result).count(item_id) == 1

    def test_custom_lookback_days(self, db: Database) -> None:
        item_id = db.add_item(url="https://example.com/a", source_type="rss")
        self._save_digest_with_item(db, item_id, days_ago=10)

        assert item_id in db.get_recently_digested_item_ids(lookback_days=30)
        assert item_id not in db.get_recently_digested_item_ids(lookback_days=5)
