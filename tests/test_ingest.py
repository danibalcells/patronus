from __future__ import annotations

from pathlib import Path
from time import struct_time
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from patronus.db import Database
from patronus.ingest import (
    _arxiv_abs_to_pdf,
    _arxiv_fetch_url,
    _display_author,
    _entry_author,
    _extract_full_text,
    _extract_links_from_html,
    _extract_tweet_content,
    _filter_allowed_links,
    _is_allowed_link,
    _normalize_url,
    _parse_timestamp,
    _parse_tweet_html,
    classify_item_type,
    ingest_linked_items,
    ingest_url,
    poll_feeds,
)


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "test.sqlite3"))


def _make_time_struct(year: int = 2026, month: int = 2, day: int = 10) -> struct_time:
    return struct_time((year, month, day, 12, 0, 0, 0, 41, 0))


def _make_entry(
    link: str = "https://example.com/post",
    title: str = "Test Post",
    author: Optional[str] = "Alice",
    summary: str = "A summary",
    published_parsed: Optional[struct_time] = None,
) -> SimpleNamespace:
    entry = SimpleNamespace(
        link=link,
        title=title,
        summary=summary,
        published_parsed=published_parsed or _make_time_struct(),
        updated_parsed=None,
    )
    if author is not None:
        entry.author = author
    return entry


def _make_parsed_feed(
    entries: list | None = None,
    feed_title: str = "Test Feed",
    status: int = 200,
    etag: Optional[str] = None,
    modified: Optional[str] = None,
) -> SimpleNamespace:
    if entries is None:
        entries = [_make_entry()]
    return SimpleNamespace(
        entries=entries,
        feed=SimpleNamespace(title=feed_title),
        status=status,
        etag=etag,
        modified=modified,
    )


class TestEntryAuthor:
    def test_simple_author(self) -> None:
        entry = SimpleNamespace(author="Alice")
        assert _entry_author(entry) == "Alice"

    def test_authors_list_dict(self) -> None:
        entry = SimpleNamespace(authors=[{"name": "Bob", "email": "bob@example.com"}])
        assert _entry_author(entry) == "Bob"

    def test_authors_list_dict_email_fallback(self) -> None:
        entry = SimpleNamespace(authors=[{"email": "bob@example.com"}])
        assert _entry_author(entry) == "bob@example.com"

    def test_authors_list_object(self) -> None:
        entry = SimpleNamespace(authors=[SimpleNamespace(name="Carol")])
        assert _entry_author(entry) == "Carol"

    def test_author_detail(self) -> None:
        entry = SimpleNamespace(
            author="",
            authors=[],
            author_detail=SimpleNamespace(name="Dave", email="dave@example.com"),
        )
        assert _entry_author(entry) == "Dave"

    def test_author_detail_email_fallback(self) -> None:
        entry = SimpleNamespace(
            author="",
            authors=[],
            author_detail=SimpleNamespace(name=None, email="dave@example.com"),
        )
        assert _entry_author(entry) == "dave@example.com"

    def test_no_author(self) -> None:
        entry = SimpleNamespace()
        assert _entry_author(entry) is None

    def test_empty_author(self) -> None:
        entry = SimpleNamespace(author="", authors=[], author_detail=None)
        assert _entry_author(entry) is None


class TestParseTimestamp:
    def test_published_parsed(self) -> None:
        entry = SimpleNamespace(
            published_parsed=_make_time_struct(2026, 2, 10), updated_parsed=None
        )
        ts = _parse_timestamp(entry)
        assert ts is not None
        assert ts.startswith("2026-02-10")

    def test_updated_parsed_fallback(self) -> None:
        entry = SimpleNamespace(
            published_parsed=None,
            updated_parsed=_make_time_struct(2026, 1, 15),
        )
        ts = _parse_timestamp(entry)
        assert ts is not None
        assert ts.startswith("2026-01-15")

    def test_no_timestamp(self) -> None:
        entry = SimpleNamespace(published_parsed=None, updated_parsed=None)
        assert _parse_timestamp(entry) is None

    def test_published_takes_precedence(self) -> None:
        entry = SimpleNamespace(
            published_parsed=_make_time_struct(2026, 3, 1),
            updated_parsed=_make_time_struct(2026, 3, 5),
        )
        ts = _parse_timestamp(entry)
        assert ts is not None
        assert ts.startswith("2026-03-01")


class TestDisplayAuthor:
    def test_both_different(self) -> None:
        assert _display_author("Alice", "Blog") == "Alice - Blog"

    def test_same_name(self) -> None:
        assert _display_author("Blog", "Blog") == "Blog"

    def test_same_case_insensitive(self) -> None:
        assert _display_author("blog", "Blog") == "blog"

    def test_no_author(self) -> None:
        assert _display_author(None, "Blog") == "Blog"

    def test_no_feed_title(self) -> None:
        assert _display_author("Alice", None) == "Alice"

    def test_both_none(self) -> None:
        assert _display_author(None, None) is None


class TestExtractFullText:
    @patch("patronus.ingest.trafilatura")
    def test_success(self, mock_traf: MagicMock) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.extract.return_value = "Extracted article text"
        result = _extract_full_text("https://example.com/post")
        assert result == "Extracted article text"
        mock_traf.fetch_url.assert_called_once_with("https://example.com/post")

    @patch("patronus.ingest.trafilatura")
    def test_fetch_returns_none(self, mock_traf: MagicMock) -> None:
        mock_traf.fetch_url.return_value = None
        assert _extract_full_text("https://example.com/post") is None

    @patch("patronus.ingest.trafilatura")
    def test_extract_returns_none(self, mock_traf: MagicMock) -> None:
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = None
        assert _extract_full_text("https://example.com/post") is None

    @patch("patronus.ingest.trafilatura")
    def test_exception_returns_none(self, mock_traf: MagicMock) -> None:
        mock_traf.fetch_url.side_effect = Exception("network error")
        assert _extract_full_text("https://example.com/post") is None

    @patch("patronus.ingest.trafilatura")
    def test_truncates_long_text(self, mock_traf: MagicMock) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.extract.return_value = "x" * 30_000
        result = _extract_full_text("https://example.com/post")
        assert result is not None
        assert len(result) == 25_000


class TestPollFeeds:
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_ingests_new_items(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(link="https://example.com/1", title="Post 1"),
                _make_entry(link="https://example.com/2", title="Post 2"),
            ]
        )
        mock_extract.return_value = "Full article text"
        embedding = np.ones(1536, dtype=np.float32)
        mock_embed.return_value = [embedding, embedding]

        ids = poll_feeds(db)

        assert len(ids) == 2
        item1 = db.get_item_by_url("https://example.com/1")
        assert item1 is not None
        assert item1.title == "Post 1"
        assert item1.source_type == "rss"
        assert item1.text == "Full article text"
        assert item1.embedding is not None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_deduplicates_by_url(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")
        db.add_item(url="https://example.com/existing", source_type="rss")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(link="https://example.com/existing", title="Old"),
                _make_entry(link="https://example.com/new", title="New"),
            ]
        )
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db)

        assert len(ids) == 1
        assert db.get_item_by_url("https://example.com/new") is not None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_falls_back_to_summary(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(
                    link="https://example.com/1",
                    title="Post",
                    summary="Feed summary text",
                )
            ]
        )
        mock_extract.return_value = None
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db)

        assert len(ids) == 1
        item = db.get_item_by_url("https://example.com/1")
        assert item is not None
        assert item.text == "Feed summary text"

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_handles_304_not_modified(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[], status=304, etag='"abc"'
        )

        ids = poll_feeds(db)

        assert len(ids) == 0
        mock_extract.assert_not_called()
        mock_embed.assert_not_called()

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_updates_feed_poll_metadata(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        feed_id = db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://example.com/1")],
            etag='"new-etag"',
            modified="Thu, 13 Feb 2026",
        )
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        feed = db.get_feed(feed_id)
        assert feed is not None
        assert feed.etag == '"new-etag"'
        assert feed.last_modified == "Thu, 13 Feb 2026"
        assert feed.last_polled is not None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_passes_etag_and_modified_to_feedparser(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        feed_id = db.add_feed(url="https://feed.example.com/rss", name="TestFeed")
        db.update_feed_poll(
            feed_id, etag='"old-etag"', last_modified="Wed, 12 Feb 2026"
        )

        mock_fp.parse.return_value = _make_parsed_feed(entries=[])
        poll_feeds(db)

        mock_fp.parse.assert_called_once_with(
            "https://feed.example.com/rss",
            etag='"old-etag"',
            modified="Wed, 12 Feb 2026",
        )

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_stores_items_without_embeddings_on_embed_failure(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://example.com/1")]
        )
        mock_extract.return_value = "Text"
        mock_embed.side_effect = Exception("API error")

        ids = poll_feeds(db)

        assert len(ids) == 1
        item = db.get_item_by_url("https://example.com/1")
        assert item is not None
        assert item.embedding is None
        assert item.text == "Text"

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_skips_entries_without_link(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        no_link = SimpleNamespace(
            link=None,
            title="No Link",
            summary="summary",
            published_parsed=_make_time_struct(),
            updated_parsed=None,
            author="Alice",
        )
        mock_fp.parse.return_value = _make_parsed_feed(entries=[no_link])
        mock_embed.return_value = []

        ids = poll_feeds(db)
        assert len(ids) == 0

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_continues_on_feed_parse_error(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://bad.example.com/rss", name="Bad")
        db.add_feed(url="https://good.example.com/rss", name="Good")

        def parse_side_effect(url: str, **kwargs: object) -> SimpleNamespace:
            if "bad" in url:
                raise Exception("parse failure")
            return _make_parsed_feed(
                entries=[_make_entry(link="https://example.com/good-post")]
            )

        mock_fp.parse.side_effect = parse_side_effect
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db)
        assert len(ids) == 1

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_no_active_feeds(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        ids = poll_feeds(db)
        assert ids == []
        mock_fp.parse.assert_not_called()

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_author_formatting(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(
                    link="https://example.com/1", author="Alice", title="Post"
                )
            ],
            feed_title="Cool Blog",
        )
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        item = db.get_item_by_url("https://example.com/1")
        assert item is not None
        assert item.author == "Alice - Cool Blog"

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_skips_embedding_for_items_without_text(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        entry_no_text = _make_entry(link="https://example.com/1", title="Post")
        entry_no_text.summary = None
        delattr(entry_no_text, "summary")

        mock_fp.parse.return_value = _make_parsed_feed(entries=[entry_no_text])
        mock_extract.return_value = None
        mock_embed.return_value = []

        ids = poll_feeds(db)

        assert len(ids) == 1
        item = db.get_item_by_url("https://example.com/1")
        assert item is not None
        assert item.text is None
        assert item.embedding is None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_limit_caps_entries_per_feed(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(link="https://example.com/1", title="Post 1"),
                _make_entry(link="https://example.com/2", title="Post 2"),
                _make_entry(link="https://example.com/3", title="Post 3"),
            ]
        )
        mock_extract.return_value = "Text"
        embedding = np.ones(1536, dtype=np.float32)
        mock_embed.return_value = [embedding, embedding]

        ids = poll_feeds(db, limit=2)

        assert len(ids) == 2
        assert db.get_item_by_url("https://example.com/1") is not None
        assert db.get_item_by_url("https://example.com/2") is not None
        assert db.get_item_by_url("https://example.com/3") is None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_limit_counts_only_new_entries(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")
        db.add_item(url="https://example.com/existing", source_type="rss")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(link="https://example.com/existing", title="Old"),
                _make_entry(link="https://example.com/new1", title="New 1"),
                _make_entry(link="https://example.com/new2", title="New 2"),
                _make_entry(link="https://example.com/new3", title="New 3"),
            ]
        )
        mock_extract.return_value = "Text"
        embedding = np.ones(1536, dtype=np.float32)
        mock_embed.return_value = [embedding, embedding]

        ids = poll_feeds(db, limit=2)

        assert len(ids) == 2
        assert db.get_item_by_url("https://example.com/new1") is not None
        assert db.get_item_by_url("https://example.com/new2") is not None
        assert db.get_item_by_url("https://example.com/new3") is None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_feed_limit_caps_feeds_processed(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed1.example.com/rss", name="A Feed")
        db.add_feed(url="https://feed2.example.com/rss", name="B Feed")
        db.add_feed(url="https://feed3.example.com/rss", name="C Feed")

        def parse_side_effect(url: str, **kwargs: object) -> SimpleNamespace:
            slug = url.split("//")[1].split(".")[0]
            return _make_parsed_feed(
                entries=[_make_entry(link=f"https://example.com/{slug}-post")],
                feed_title=slug,
            )

        mock_fp.parse.side_effect = parse_side_effect
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db, feed_limit=2)

        assert len(ids) == 2
        assert mock_fp.parse.call_count == 2

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_feed_limit_none_processes_all(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed1.example.com/rss", name="A Feed")
        db.add_feed(url="https://feed2.example.com/rss", name="B Feed")
        db.add_feed(url="https://feed3.example.com/rss", name="C Feed")

        def parse_side_effect(url: str, **kwargs: object) -> SimpleNamespace:
            slug = url.split("//")[1].split(".")[0]
            return _make_parsed_feed(
                entries=[_make_entry(link=f"https://example.com/{slug}-post")],
                feed_title=slug,
            )

        mock_fp.parse.side_effect = parse_side_effect
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db)

        assert len(ids) == 3
        assert mock_fp.parse.call_count == 3

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_feed_limit_greater_than_total_is_fine(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed1.example.com/rss", name="A Feed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://example.com/1")]
        )
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db, feed_limit=100)

        assert len(ids) == 1
        assert mock_fp.parse.call_count == 1

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_skip_embed_stores_without_embeddings(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="TestFeed")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://example.com/1", title="Post")]
        )
        mock_extract.return_value = "Full article text"

        ids = poll_feeds(db, skip_embed=True)

        assert len(ids) == 1
        mock_embed.assert_not_called()
        item = db.get_item_by_url("https://example.com/1")
        assert item is not None
        assert item.text == "Full article text"
        assert item.embedding is None


class TestIngestUrl:
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_ingests_manual_url(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Article text",
            "title": "Article Title",
            "author": "Author Name",
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        item_id = ingest_url(db, "https://example.com/article")

        assert item_id is not None
        item = db.get_item(item_id)
        assert item is not None
        assert item.source_type == "manual"
        assert item.title == "Article Title"
        assert item.author == "Author Name"
        assert item.text == "Article text"
        assert item.embedding is not None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_deduplicates(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        db.add_item(url="https://example.com/exists", source_type="rss")
        result = ingest_url(db, "https://example.com/exists")
        assert result is None
        mock_traf.fetch_url.assert_not_called()

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_stores_without_embedding_on_failure(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Text",
            "title": "Title",
            "author": None,
        }
        mock_embed.side_effect = Exception("API error")

        item_id = ingest_url(db, "https://example.com/article")

        assert item_id is not None
        item = db.get_item(item_id)
        assert item is not None
        assert item.text == "Text"
        assert item.embedding is None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_handles_extraction_failure(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = None
        mock_traf.bare_extraction.return_value = None

        item_id = ingest_url(db, "https://example.com/article")

        assert item_id is not None
        item = db.get_item(item_id)
        assert item is not None
        assert item.text is None
        assert item.embedding is None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_skips_embedding_when_no_text(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.bare_extraction.return_value = {
            "text": "",
            "title": "Empty",
            "author": None,
        }

        ingest_url(db, "https://example.com/empty")

        mock_embed.assert_not_called()


class TestClassifyItemType:
    def test_x_com_tweet(self) -> None:
        assert classify_item_type("https://x.com/user/status/123") == "tweet"

    def test_twitter_com_tweet(self) -> None:
        assert classify_item_type("https://twitter.com/user/status/456") == "tweet"

    def test_www_x_com_tweet(self) -> None:
        assert classify_item_type("https://www.x.com/user/status/123") == "tweet"

    def test_arxiv_paper(self) -> None:
        assert classify_item_type("https://arxiv.org/abs/2026.12345") == "paper"

    def test_www_arxiv_paper(self) -> None:
        assert classify_item_type("https://www.arxiv.org/abs/2026.12345") == "paper"

    def test_openreview_paper(self) -> None:
        assert classify_item_type("https://openreview.net/forum?id=abc") == "paper"

    def test_blog_defaults_to_article(self) -> None:
        assert classify_item_type("https://example.com/post") == "article"

    def test_substack_defaults_to_article(self) -> None:
        assert classify_item_type("https://someone.substack.com/p/title") == "article"

    def test_empty_url(self) -> None:
        assert classify_item_type("") == "article"

    def test_malformed_url(self) -> None:
        assert classify_item_type("not-a-url") == "article"


class TestItemTypeInPollFeeds:
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_tweet_url_classified(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://rss.app/feeds/twitter.xml", name="Twitter")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(link="https://x.com/user/status/123", title="A tweet"),
            ]
        )
        mock_extract.return_value = "Tweet text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        item = db.get_item_by_url("https://x.com/user/status/123")
        assert item is not None
        assert item.item_type == "tweet"

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_blog_url_classified_as_article(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="Blog")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[
                _make_entry(link="https://blog.example.com/post", title="A post"),
            ]
        )
        mock_extract.return_value = "Post text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        item = db.get_item_by_url("https://blog.example.com/post")
        assert item is not None
        assert item.item_type == "article"


class TestItemTypeInIngestUrl:
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_arxiv_url_classified_as_paper(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Paper abstract",
            "title": "A Paper",
            "author": "Author",
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        item_id = ingest_url(db, "https://arxiv.org/abs/2026.12345")

        assert item_id is not None
        item = db.get_item(item_id)
        assert item is not None
        assert item.item_type == "paper"

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_generic_url_classified_as_article(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Article text",
            "title": "An Article",
            "author": "Author",
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        item_id = ingest_url(db, "https://example.com/article")

        assert item_id is not None
        item = db.get_item(item_id)
        assert item is not None
        assert item.item_type == "article"


RSSAPP_TWEET_HTML = (
    'Honestly once "write a an episode of Seinfeld about quantum mechanics"'
    " worked, the rest was kind of obvious."
    "<hr/>"
    "<blockquote>"
    "<b>Shane Gu (@shaneguML)</b>"
    'honestly once "let\'s think step by step" worked, the rest of kind of obvious'
    "<footer>"
    "&mdash; <cite>"
    ' <a href="https://x.com/shaneguML/status/2022408843062579305">'
    "https://x.com/shaneguML/status/2022408843062579305</a>"
    "</footer>"
    "</blockquote>"
)

RSSAPP_TWEET_WITH_IMAGE_HTML = (
    '"But Cars Don\'t Actually Run," Says Increasingly Nervous Horse'
    " For the 7th Time This Year"
    "<hr/>"
    "<blockquote>"
    "<b>AI Notkilleveryoneism Memes ⏸\ufe0f (@AISafetyMemes)</b>"
    "humans don't actually think, they just imitate others<br>"
    "<br>"
    "humans don't actually think, they're just math<br>"
    "<br>"
    "humans don't...<footer>"
    "&mdash; <cite>"
    ' <a href="https://x.com/AISafetyMemes/status/1863618182398820596">'
    "https://x.com/AISafetyMemes/status/1863618182398820596</a>"
    "</footer>"
    "</blockquote>"
)


class TestParseTweetHtml:
    def test_extracts_text_from_simple_tweet(self) -> None:
        text, links = _parse_tweet_html(RSSAPP_TWEET_HTML)
        assert "Seinfeld about quantum mechanics" in text
        assert "Shane Gu" in text
        assert "let's think step by step" in text

    def test_strips_html_tags(self) -> None:
        text, _links = _parse_tweet_html(RSSAPP_TWEET_HTML)
        assert "<blockquote>" not in text
        assert "<b>" not in text
        assert "<hr/>" not in text
        assert "<a " not in text

    def test_unescapes_html_entities(self) -> None:
        text, _links = _parse_tweet_html(RSSAPP_TWEET_HTML)
        assert "\u2014" in text
        assert "&mdash;" not in text

    def test_extracts_links(self) -> None:
        _text, links = _parse_tweet_html(RSSAPP_TWEET_HTML)
        assert len(links) == 1
        assert links[0] == "https://x.com/shaneguML/status/2022408843062579305"

    def test_handles_br_as_newlines(self) -> None:
        text, _links = _parse_tweet_html(RSSAPP_TWEET_WITH_IMAGE_HTML)
        assert "imitate others\n" in text

    def test_collapses_excessive_newlines(self) -> None:
        text, _links = _parse_tweet_html(RSSAPP_TWEET_WITH_IMAGE_HTML)
        assert "\n\n\n" not in text

    def test_extracts_links_from_quoted_tweet(self) -> None:
        _text, links = _parse_tweet_html(RSSAPP_TWEET_WITH_IMAGE_HTML)
        assert "https://x.com/AISafetyMemes/status/1863618182398820596" in links

    def test_abridged_url_replaced_with_full_href(self) -> None:
        html = (
            'Check out this post: '
            '<a href="https://bounded-regret.ghost.io/building-a-thing">'
            "bounded-regret.ghost.io/buil\u2026"
            "</a>"
        )
        text, links = _parse_tweet_html(html)
        assert "bounded-regret.ghost.io/buil\u2026" not in text
        assert "https://bounded-regret.ghost.io/building-a-thing" in text
        assert links == ["https://bounded-regret.ghost.io/building-a-thing"]

    def test_non_abridged_link_text_unchanged(self) -> None:
        html = (
            '<a href="https://x.com/user/status/123">'
            "https://x.com/user/status/123"
            "</a>"
        )
        text, links = _parse_tweet_html(html)
        assert "https://x.com/user/status/123" in text
        assert links == ["https://x.com/user/status/123"]

    def test_non_url_link_text_unchanged(self) -> None:
        html = '<a href="https://x.com/hashtag/AI">#AI</a> is exciting'
        text, links = _parse_tweet_html(html)
        assert "#AI" in text
        assert "https://x.com/hashtag/AI" not in text

    def test_empty_html(self) -> None:
        text, links = _parse_tweet_html("")
        assert text == ""
        assert links == []

    def test_plain_text_passthrough(self) -> None:
        text, links = _parse_tweet_html("Just a plain tweet with no HTML")
        assert text == "Just a plain tweet with no HTML"
        assert links == []


class TestExtractTweetContent:
    def test_extracts_from_summary(self) -> None:
        entry_data = {"summary": RSSAPP_TWEET_HTML}
        text, links = _extract_tweet_content(entry_data)
        assert text is not None
        assert "Seinfeld about quantum mechanics" in text
        assert "<blockquote>" not in text
        assert len(links) == 1

    def test_returns_none_without_summary(self) -> None:
        text, links = _extract_tweet_content({"summary": None})
        assert text is None
        assert links == []
        text, links = _extract_tweet_content({})
        assert text is None
        assert links == []

    def test_returns_none_for_empty_summary(self) -> None:
        text, links = _extract_tweet_content({"summary": ""})
        assert text is None
        assert links == []


class TestTweetExtractionInPollFeeds:
    @patch("patronus.ingest.ingest_linked_items", return_value=[])
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_tweet_uses_rss_description_not_trafilatura(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://rss.app/feeds/twitter.xml", name="Twitter AI")

        tweet_entry = _make_entry(
            link="https://x.com/adamimos/status/2022511507482005987",
            title='@adamimos: Honestly once "write a an episode of Seinfeld"...',
            author="@adamimos",
            summary=RSSAPP_TWEET_HTML,
        )

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[tweet_entry], feed_title="AI"
        )
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        ids = poll_feeds(db)

        mock_extract.assert_not_called()
        assert len(ids) == 1
        item = db.get_item_by_url(
            "https://x.com/adamimos/status/2022511507482005987"
        )
        assert item is not None
        assert item.item_type == "tweet"
        assert "Seinfeld about quantum mechanics" in (item.text or "")
        assert "JavaScript" not in (item.text or "")

    @patch("patronus.ingest.ingest_linked_items", return_value=[])
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_mixed_tweets_and_articles(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://rss.app/feeds/twitter.xml", name="Twitter AI")
        db.add_feed(url="https://blog.example.com/rss", name="Blog")

        tweet_entry = _make_entry(
            link="https://x.com/user/status/123",
            title="A tweet",
            summary="<b>Tweet content</b> with <a href=\"https://arxiv.org/abs/1234\">a link</a>",
        )
        article_entry = _make_entry(
            link="https://blog.example.com/post",
            title="A blog post",
            summary="Blog summary",
        )

        def parse_side_effect(url: str, **kwargs: object) -> SimpleNamespace:
            if "twitter" in url:
                return _make_parsed_feed(
                    entries=[tweet_entry], feed_title="Twitter AI"
                )
            return _make_parsed_feed(
                entries=[article_entry], feed_title="Blog"
            )

        mock_fp.parse.side_effect = parse_side_effect
        mock_extract.return_value = "Full article text from trafilatura"
        mock_embed.return_value = [
            np.ones(1536, dtype=np.float32),
            np.ones(1536, dtype=np.float32),
        ]

        ids = poll_feeds(db)

        assert len(ids) == 2

        tweet_item = db.get_item_by_url("https://x.com/user/status/123")
        assert tweet_item is not None
        assert tweet_item.item_type == "tweet"
        assert "Tweet content" in (tweet_item.text or "")
        assert "<b>" not in (tweet_item.text or "")

        article_item = db.get_item_by_url("https://blog.example.com/post")
        assert article_item is not None
        assert article_item.item_type == "article"
        assert article_item.text == "Full article text from trafilatura"

        mock_extract.assert_called_once_with("https://blog.example.com/post")


class TestExtractLinksFromHtml:
    def test_extracts_href_links(self) -> None:
        html = '<a href="https://arxiv.org/abs/1234">paper</a> and <a href="https://example.com">site</a>'
        links = _extract_links_from_html(html)
        assert links == ["https://arxiv.org/abs/1234", "https://example.com"]

    def test_ignores_non_http_links(self) -> None:
        html = '<a href="mailto:x@y.com">email</a> <a href="/relative">rel</a>'
        links = _extract_links_from_html(html)
        assert links == []

    def test_empty_html(self) -> None:
        assert _extract_links_from_html("") == []

    def test_no_links(self) -> None:
        assert _extract_links_from_html("<p>No links here</p>") == []

    def test_handles_malformed_html(self) -> None:
        html = '<a href="https://example.com">unclosed'
        links = _extract_links_from_html(html)
        assert links == ["https://example.com"]


class TestNormalizeUrl:
    def test_arxiv_abs_to_pdf(self) -> None:
        assert _normalize_url("https://arxiv.org/abs/2301.12345") == "https://arxiv.org/pdf/2301.12345"

    def test_arxiv_pdf_unchanged(self) -> None:
        assert _normalize_url("https://arxiv.org/pdf/2301.12345") == "https://arxiv.org/pdf/2301.12345"

    def test_non_arxiv_unchanged(self) -> None:
        assert _normalize_url("https://example.com/pdf/doc") == "https://example.com/pdf/doc"

    def test_arxiv_other_paths_unchanged(self) -> None:
        assert _normalize_url("https://arxiv.org/html/2301.12345") == "https://arxiv.org/html/2301.12345"


class TestIsAllowedLink:
    def test_arxiv_allowed(self) -> None:
        assert _is_allowed_link("https://arxiv.org/abs/2301.12345") is True

    def test_openreview_allowed(self) -> None:
        assert _is_allowed_link("https://openreview.net/forum?id=abc") is True

    def test_anthropic_allowed(self) -> None:
        assert _is_allowed_link("https://www.anthropic.com/research/paper") is True

    def test_deepmind_allowed(self) -> None:
        assert _is_allowed_link("https://deepmind.google/blog/post") is True

    def test_openai_allowed(self) -> None:
        assert _is_allowed_link("https://openai.com/research/index") is True

    def test_substack_allowed(self) -> None:
        assert _is_allowed_link("https://thezvi.substack.com/p/post") is True
        assert _is_allowed_link("https://aisupremacy.substack.com/p/title") is True

    def test_twitter_not_allowed(self) -> None:
        assert _is_allowed_link("https://x.com/user/status/123") is False
        assert _is_allowed_link("https://twitter.com/user/status/123") is False

    def test_youtube_not_allowed(self) -> None:
        assert _is_allowed_link("https://youtube.com/watch?v=abc") is False

    def test_image_hosts_not_allowed(self) -> None:
        assert _is_allowed_link("https://pbs.twimg.com/media/abc.jpg") is False
        assert _is_allowed_link("https://imgur.com/abc") is False

    def test_random_domain_not_allowed(self) -> None:
        assert _is_allowed_link("https://example.com/post") is False

    def test_bare_substack_not_allowed(self) -> None:
        assert _is_allowed_link("https://substack.com") is False


class TestFilterAllowedLinks:
    def test_filters_to_allowed_only(self) -> None:
        urls = [
            "https://arxiv.org/abs/1234",
            "https://x.com/user/status/123",
            "https://thezvi.substack.com/p/post",
            "https://youtube.com/watch?v=abc",
        ]
        result = _filter_allowed_links(urls)
        assert result == [
            "https://arxiv.org/pdf/1234",
            "https://thezvi.substack.com/p/post",
        ]

    def test_deduplicates(self) -> None:
        urls = [
            "https://arxiv.org/abs/1234",
            "https://arxiv.org/abs/1234",
        ]
        result = _filter_allowed_links(urls)
        assert result == ["https://arxiv.org/pdf/1234"]

    def test_normalizes_arxiv_abs_and_pdf_to_same_url(self) -> None:
        urls = [
            "https://arxiv.org/pdf/1234",
            "https://arxiv.org/abs/1234",
        ]
        result = _filter_allowed_links(urls)
        assert len(result) == 1
        assert result[0] == "https://arxiv.org/pdf/1234"

    def test_empty_list(self) -> None:
        assert _filter_allowed_links([]) == []


class TestIngestLinkedItems:
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_ingests_linked_url(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )

        mock_extract.return_value = "Paper abstract text"
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ids = ingest_linked_items(db, parent_id, ["https://arxiv.org/abs/2301.12345"])

        assert len(ids) == 1
        item = db.get_item_by_url("https://arxiv.org/pdf/2301.12345")
        assert item is not None
        assert item.source_type == "link_extraction"
        assert item.item_type == "paper"
        assert item.source_item_id == parent_id
        assert item.text == "Paper abstract text"
        assert item.embedding is not None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_skips_existing_url(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )
        db.add_item(
            url="https://arxiv.org/pdf/2301.12345",
            source_type="rss",
            item_type="paper",
        )

        ids = ingest_linked_items(db, parent_id, ["https://arxiv.org/abs/2301.12345"])

        assert len(ids) == 0
        mock_extract.assert_not_called()

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_handles_extraction_failure(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )

        mock_extract.return_value = None

        ids = ingest_linked_items(db, parent_id, ["https://arxiv.org/abs/2301.12345"])

        assert len(ids) == 1
        item = db.get_item_by_url("https://arxiv.org/pdf/2301.12345")
        assert item is not None
        assert item.text is None
        assert item.embedding is None
        assert item.source_item_id == parent_id

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_skip_embed_flag(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )

        mock_extract.return_value = "Paper text"

        ids = ingest_linked_items(
            db, parent_id, ["https://arxiv.org/abs/2301.12345"], skip_embed=True
        )

        assert len(ids) == 1
        mock_embed.assert_not_called()
        item = db.get_item_by_url("https://arxiv.org/pdf/2301.12345")
        assert item is not None
        assert item.text == "Paper text"
        assert item.embedding is None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_continues_on_individual_failure(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123",
            source_type="rss",
            item_type="tweet",
        )

        mock_extract.side_effect = [Exception("network error"), "Paper text"]
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ids = ingest_linked_items(
            db,
            parent_id,
            [
                "https://arxiv.org/abs/1111.11111",
                "https://arxiv.org/abs/2222.22222",
            ],
        )

        assert len(ids) == 1
        assert db.get_item_by_url("https://arxiv.org/pdf/1111.11111") is None
        assert db.get_item_by_url("https://arxiv.org/pdf/2222.22222") is not None


class TestLinkExtractionInPollFeeds:
    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_tweet_with_arxiv_link_triggers_extraction(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://rss.app/feeds/twitter.xml", name="Twitter AI")

        tweet_html = (
            'Check out this paper '
            '<a href="https://arxiv.org/abs/2301.12345">https://arxiv.org/abs/2301.12345</a>'
        )
        tweet_entry = _make_entry(
            link="https://x.com/user/status/123",
            title="A tweet",
            summary=tweet_html,
        )

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[tweet_entry], feed_title="Twitter AI"
        )
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]
        mock_ingest_linked.return_value = []

        poll_feeds(db)

        mock_ingest_linked.assert_called_once()
        call_args = mock_ingest_linked.call_args
        assert call_args[0][2] == ["https://arxiv.org/pdf/2301.12345"]

    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_no_links_does_not_trigger_extraction(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="Blog")

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://example.com/post", summary="no links")]
        )
        mock_extract.return_value = "Text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        mock_ingest_linked.assert_not_called()

    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_non_allowed_links_filtered_out(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://rss.app/feeds/twitter.xml", name="Twitter AI")

        tweet_html = (
            'Look at this <a href="https://youtube.com/watch?v=abc">video</a> '
            'and <a href="https://x.com/other/status/456">tweet</a>'
        )
        tweet_entry = _make_entry(
            link="https://x.com/user/status/123",
            title="A tweet",
            summary=tweet_html,
        )

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[tweet_entry], feed_title="Twitter AI"
        )
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        mock_ingest_linked.assert_not_called()

    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_article_summary_links_extracted(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://feed.example.com/rss", name="Blog")

        summary_html = (
            'Great post referencing <a href="https://arxiv.org/abs/9999.99999">this paper</a>'
        )
        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://example.com/post", summary=summary_html)]
        )
        mock_extract.return_value = "Full text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]
        mock_ingest_linked.return_value = []

        poll_feeds(db)

        mock_ingest_linked.assert_called_once()
        call_args = mock_ingest_linked.call_args
        assert call_args[0][2] == ["https://arxiv.org/pdf/9999.99999"]


class TestLinkExtractionInIngestUrl:
    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_extracts_links_from_downloaded_html(
        self,
        mock_traf: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        html_with_link = (
            '<html><body>'
            '<a href="https://arxiv.org/abs/2301.12345">paper</a>'
            '</body></html>'
        )
        mock_traf.fetch_url.return_value = html_with_link
        mock_traf.bare_extraction.return_value = {
            "text": "Article mentioning a paper",
            "title": "Blog Post",
            "author": "Author",
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)
        mock_ingest_linked.return_value = ["linked-id"]

        item_id = ingest_url(db, "https://example.com/blog-post")

        assert item_id is not None
        mock_ingest_linked.assert_called_once()
        call_args = mock_ingest_linked.call_args
        assert call_args[0][0] == db
        assert call_args[0][1] == item_id
        assert call_args[0][2] == ["https://arxiv.org/pdf/2301.12345"]

    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_no_links_does_not_trigger_extraction(
        self,
        mock_traf: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        mock_traf.fetch_url.return_value = "<html><body>plain</body></html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Plain article",
            "title": "Title",
            "author": None,
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_url(db, "https://example.com/article")

        mock_ingest_linked.assert_not_called()

    @patch("patronus.ingest.ingest_linked_items")
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_no_extraction_when_download_fails(
        self,
        mock_traf: MagicMock,
        mock_embed: MagicMock,
        mock_ingest_linked: MagicMock,
        db: Database,
    ) -> None:
        mock_traf.fetch_url.return_value = None
        mock_traf.bare_extraction.return_value = None

        ingest_url(db, "https://example.com/article")

        mock_ingest_linked.assert_not_called()


class TestEndToEndLinkExtraction:
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_tweet_with_paper_and_tweet_links(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed_batch: MagicMock,
        mock_embed_text: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://rss.app/feeds/twitter.xml", name="Twitter AI")

        tweet_html = (
            "Interesting thread by @researcher on scaling laws "
            '<a href="https://x.com/researcher/status/999">https://x.com/researcher/status/999</a> '
            "and the paper backing it up "
            '<a href="https://arxiv.org/abs/2602.01234">https://arxiv.org/abs/2602.01234</a>'
        )
        tweet_entry = _make_entry(
            link="https://x.com/user/status/123",
            title="@user: Interesting thread...",
            author="@user",
            summary=tweet_html,
        )

        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[tweet_entry], feed_title="Twitter AI"
        )
        embedding = np.ones(1536, dtype=np.float32)
        mock_embed_batch.return_value = [embedding]
        mock_extract.return_value = "Abstract: We study scaling laws..."
        mock_embed_text.return_value = embedding

        ids = poll_feeds(db)

        assert len(ids) == 2

        tweet = db.get_item_by_url("https://x.com/user/status/123")
        assert tweet is not None
        assert tweet.item_type == "tweet"
        assert tweet.source_item_id is None
        assert tweet.source_type == "rss"

        paper = db.get_item_by_url("https://arxiv.org/pdf/2602.01234")
        assert paper is not None
        assert paper.item_type == "paper"
        assert paper.source_item_id == tweet.id
        assert paper.source_type == "link_extraction"
        assert paper.text == "Abstract: We study scaling laws..."
        assert paper.embedding is not None

        assert db.get_item_by_url("https://x.com/researcher/status/999") is None


class TestArxivUrlHelpers:
    def test_abs_to_pdf_converts_abs(self) -> None:
        assert _arxiv_abs_to_pdf("https://arxiv.org/abs/2301.12345") == "https://arxiv.org/pdf/2301.12345"

    def test_abs_to_pdf_leaves_pdf_unchanged(self) -> None:
        assert _arxiv_abs_to_pdf("https://arxiv.org/pdf/2301.12345") == "https://arxiv.org/pdf/2301.12345"

    def test_abs_to_pdf_leaves_non_arxiv_unchanged(self) -> None:
        assert _arxiv_abs_to_pdf("https://example.com/abs/paper") == "https://example.com/abs/paper"

    def test_abs_to_pdf_www_prefix(self) -> None:
        assert _arxiv_abs_to_pdf("https://www.arxiv.org/abs/2301.12345") == "https://www.arxiv.org/pdf/2301.12345"

    def test_abs_to_pdf_other_arxiv_paths_unchanged(self) -> None:
        assert _arxiv_abs_to_pdf("https://arxiv.org/html/2301.12345") == "https://arxiv.org/html/2301.12345"

    def test_fetch_url_converts_pdf_to_abs(self) -> None:
        assert _arxiv_fetch_url("https://arxiv.org/pdf/2301.12345") == "https://arxiv.org/abs/2301.12345"

    def test_fetch_url_leaves_abs_unchanged(self) -> None:
        assert _arxiv_fetch_url("https://arxiv.org/abs/2301.12345") == "https://arxiv.org/abs/2301.12345"

    def test_fetch_url_leaves_non_arxiv_unchanged(self) -> None:
        assert _arxiv_fetch_url("https://example.com/pdf/doc") == "https://example.com/pdf/doc"

    def test_fetch_url_www_prefix(self) -> None:
        assert _arxiv_fetch_url("https://www.arxiv.org/pdf/2301.12345") == "https://www.arxiv.org/abs/2301.12345"

    def test_fetch_url_other_arxiv_paths_unchanged(self) -> None:
        assert _arxiv_fetch_url("https://arxiv.org/html/2301.12345") == "https://arxiv.org/html/2301.12345"

    def test_roundtrip_abs_to_pdf_to_fetch(self) -> None:
        abs_url = "https://arxiv.org/abs/2301.12345"
        assert _arxiv_fetch_url(_arxiv_abs_to_pdf(abs_url)) == abs_url


class TestPollFeedsArxivUrls:
    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_arxiv_abs_url_stored_as_pdf(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://export.arxiv.org/rss/cs.LG", name="arXiv cs.LG")
        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://arxiv.org/abs/2301.12345", title="Paper")]
        )
        mock_extract.return_value = "Abstract text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        assert db.get_item_by_url("https://arxiv.org/pdf/2301.12345") is not None
        assert db.get_item_by_url("https://arxiv.org/abs/2301.12345") is None

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_arxiv_abs_url_fetched_via_abs(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://export.arxiv.org/rss/cs.LG", name="arXiv cs.LG")
        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://arxiv.org/abs/2301.12345", title="Paper")]
        )
        mock_extract.return_value = "Abstract text"
        mock_embed.return_value = [np.ones(1536, dtype=np.float32)]

        poll_feeds(db)

        mock_extract.assert_called_once_with("https://arxiv.org/abs/2301.12345")

    @patch("patronus.ingest.embed_batch")
    @patch("patronus.ingest._extract_full_text")
    @patch("patronus.ingest.feedparser")
    def test_arxiv_abs_deduplicates_against_existing_pdf(
        self,
        mock_fp: MagicMock,
        mock_extract: MagicMock,
        mock_embed: MagicMock,
        db: Database,
    ) -> None:
        db.add_feed(url="https://export.arxiv.org/rss/cs.LG", name="arXiv cs.LG")
        db.add_item(url="https://arxiv.org/pdf/2301.12345", source_type="rss")
        mock_fp.parse.return_value = _make_parsed_feed(
            entries=[_make_entry(link="https://arxiv.org/abs/2301.12345", title="Paper")]
        )

        ids = poll_feeds(db)

        assert len(ids) == 0
        mock_extract.assert_not_called()


class TestIngestLinkedItemsArxivUrls:
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_abs_link_stored_as_pdf(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123", source_type="rss", item_type="tweet"
        )
        mock_extract.return_value = "Abstract text"
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_linked_items(db, parent_id, ["https://arxiv.org/abs/2301.12345"])

        assert db.get_item_by_url("https://arxiv.org/pdf/2301.12345") is not None
        assert db.get_item_by_url("https://arxiv.org/abs/2301.12345") is None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_abs_link_fetched_via_abs_url(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123", source_type="rss", item_type="tweet"
        )
        mock_extract.return_value = "Abstract text"
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_linked_items(db, parent_id, ["https://arxiv.org/abs/2301.12345"])

        mock_extract.assert_called_once_with("https://arxiv.org/abs/2301.12345")

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_pdf_link_input_also_stored_as_pdf(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123", source_type="rss", item_type="tweet"
        )
        mock_extract.return_value = "Abstract text"
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_linked_items(db, parent_id, ["https://arxiv.org/pdf/2301.12345"])

        assert db.get_item_by_url("https://arxiv.org/pdf/2301.12345") is not None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest._extract_full_text")
    def test_pdf_link_fetched_via_abs_url(
        self, mock_extract: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        parent_id = db.add_item(
            url="https://x.com/user/status/123", source_type="rss", item_type="tweet"
        )
        mock_extract.return_value = "Abstract text"
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_linked_items(db, parent_id, ["https://arxiv.org/pdf/2301.12345"])

        mock_extract.assert_called_once_with("https://arxiv.org/abs/2301.12345")


class TestIngestUrlArxivUrls:
    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_abs_url_stored_as_pdf(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Abstract", "title": "Paper", "author": "Author"
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        item_id = ingest_url(db, "https://arxiv.org/abs/2301.12345")

        assert item_id is not None
        assert db.get_item_by_url("https://arxiv.org/pdf/2301.12345") is not None
        assert db.get_item_by_url("https://arxiv.org/abs/2301.12345") is None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_abs_url_fetched_via_abs(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Abstract", "title": "Paper", "author": None
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_url(db, "https://arxiv.org/abs/2301.12345")

        mock_traf.fetch_url.assert_called_once_with("https://arxiv.org/abs/2301.12345")

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_pdf_url_input_stored_as_pdf(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Abstract", "title": "Paper", "author": None
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        item_id = ingest_url(db, "https://arxiv.org/pdf/2301.12345")

        assert item_id is not None
        assert db.get_item_by_url("https://arxiv.org/pdf/2301.12345") is not None

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_pdf_url_input_fetched_via_abs(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.bare_extraction.return_value = {
            "text": "Abstract", "title": "Paper", "author": None
        }
        mock_embed.return_value = np.ones(1536, dtype=np.float32)

        ingest_url(db, "https://arxiv.org/pdf/2301.12345")

        mock_traf.fetch_url.assert_called_once_with("https://arxiv.org/abs/2301.12345")

    @patch("patronus.ingest.embed_text")
    @patch("patronus.ingest.trafilatura")
    def test_abs_url_deduplicates_against_existing_pdf(
        self, mock_traf: MagicMock, mock_embed: MagicMock, db: Database
    ) -> None:
        db.add_item(url="https://arxiv.org/pdf/2301.12345", source_type="rss")

        result = ingest_url(db, "https://arxiv.org/abs/2301.12345")

        assert result is None
        mock_traf.fetch_url.assert_not_called()
