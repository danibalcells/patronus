from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from patronus.notion_mirror import MirrorPage, NotionMirror, open_mirror


@pytest.fixture()
def mirror(tmp_path: Path) -> NotionMirror:
    return NotionMirror(str(tmp_path / "test_mirror.sqlite3"))


def _insert_page(
    mirror: NotionMirror,
    *,
    page_id: str = "page1",
    title: str = "Test Page",
    content: str = "Some content",
    source_db: str = "journal",
    url: str = "https://notion.so/page1",
    created_at: str = "2026-01-01T00:00:00Z",
    last_edited_at: str = "2026-02-01T00:00:00Z",
    embedding: np.ndarray | None = None,
) -> None:
    mirror.upsert_page(
        page_id=page_id,
        title=title,
        content=content,
        source_db=source_db,
        url=url,
        created_at=created_at,
        last_edited_at=last_edited_at,
        embedding=embedding,
    )


class TestNotionMirrorSchema:
    def test_creates_pages_table(self, mirror: NotionMirror) -> None:
        row = mirror._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pages'"
        ).fetchone()
        assert row is not None

    def test_creates_fts_table(self, mirror: NotionMirror) -> None:
        row = mirror._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='pages_fts'"
        ).fetchone()
        assert row is not None

    def test_creates_sync_meta_table(self, mirror: NotionMirror) -> None:
        row = mirror._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sync_meta'"
        ).fetchone()
        assert row is not None

    def test_context_manager(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "cm.sqlite3")
        with NotionMirror(db_path) as m:
            _insert_page(m)
        with NotionMirror(db_path) as m2:
            assert m2.page_count() == 1


class TestUpsertPage:
    def test_insert_and_count(self, mirror: NotionMirror) -> None:
        assert mirror.page_count() == 0
        _insert_page(mirror, page_id="p1")
        assert mirror.page_count() == 1

    def test_upsert_updates_existing(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", title="Old Title", content="old")
        _insert_page(mirror, page_id="p1", title="New Title", content="new")
        assert mirror.page_count() == 1
        rows = mirror._conn.execute("SELECT title, content FROM pages WHERE id='p1'").fetchall()
        assert rows[0]["title"] == "New Title"
        assert rows[0]["content"] == "new"

    def test_stores_embedding(self, mirror: NotionMirror) -> None:
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        _insert_page(mirror, page_id="p1", embedding=emb)
        row = mirror._conn.execute("SELECT embedding FROM pages WHERE id='p1'").fetchone()
        assert row["embedding"] is not None
        recovered = np.frombuffer(row["embedding"], dtype=np.float32)
        assert np.allclose(recovered, emb)

    def test_upsert_preserves_existing_embedding_when_none_provided(self, mirror: NotionMirror) -> None:
        emb = np.array([0.1, 0.2], dtype=np.float32)
        _insert_page(mirror, page_id="p1", content="original", embedding=emb)
        _insert_page(mirror, page_id="p1", content="updated", embedding=None)
        row = mirror._conn.execute("SELECT embedding FROM pages WHERE id='p1'").fetchone()
        assert row["embedding"] is not None


class TestSyncMeta:
    def test_get_last_synced_at_returns_none_when_missing(self, mirror: NotionMirror) -> None:
        assert mirror.get_last_synced_at("journal") is None

    def test_set_and_get_last_synced_at(self, mirror: NotionMirror) -> None:
        mirror.set_last_synced_at("journal", "2026-02-10T00:00:00Z")
        assert mirror.get_last_synced_at("journal") == "2026-02-10T00:00:00Z"

    def test_update_last_synced_at(self, mirror: NotionMirror) -> None:
        mirror.set_last_synced_at("journal", "2026-02-01T00:00:00Z")
        mirror.set_last_synced_at("journal", "2026-02-10T00:00:00Z")
        assert mirror.get_last_synced_at("journal") == "2026-02-10T00:00:00Z"

    def test_is_stale_when_no_sync_meta(self, mirror: NotionMirror) -> None:
        assert mirror.is_stale(max_age_hours=24) is True

    def test_is_stale_when_recently_synced(self, mirror: NotionMirror) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        mirror.set_last_synced_at("journal", now)
        assert mirror.is_stale(max_age_hours=24) is False

    def test_is_stale_when_old(self, mirror: NotionMirror) -> None:
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        mirror.set_last_synced_at("journal", old)
        assert mirror.is_stale(max_age_hours=24) is True


class TestSearch:
    def test_returns_matching_page(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", title="mechanistic interpretability", content="circuits and features")
        _insert_page(mirror, page_id="p2", title="cooking recipes", content="pasta and sauce")
        results = mirror.search("mechanistic")
        assert len(results) == 1
        assert results[0].id == "p1"

    def test_returns_mirror_page_shape(self, mirror: NotionMirror) -> None:
        _insert_page(
            mirror,
            page_id="p1",
            title="AI safety",
            content="deceptive alignment",
            source_db="notes",
            url="https://notion.so/p1",
            created_at="2026-01-01T00:00:00Z",
            last_edited_at="2026-02-10T00:00:00Z",
        )
        results = mirror.search("deceptive")
        assert len(results) == 1
        p = results[0]
        assert isinstance(p, MirrorPage)
        assert p.id == "p1"
        assert p.title == "AI safety"
        assert p.source_db == "notes"
        assert p.url == "https://notion.so/p1"
        assert p.created_at == "2026-01-01T00:00:00Z"
        assert p.last_edited_at == "2026-02-10T00:00:00Z"

    def test_content_snippet_truncated(self, mirror: NotionMirror) -> None:
        long_content = "word " * 200
        _insert_page(mirror, page_id="p1", content=long_content)
        results = mirror.search("word")
        assert len(results[0].content_snippet) <= 305

    def test_filter_by_source_dbs(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", title="journal entry", source_db="journal")
        _insert_page(mirror, page_id="p2", title="journal note", source_db="notes")
        results = mirror.search("journal", source_dbs=["journal"])
        ids = {r.id for r in results}
        assert "p1" in ids
        assert "p2" not in ids

    def test_limit(self, mirror: NotionMirror) -> None:
        for i in range(5):
            _insert_page(mirror, page_id=f"p{i}", title=f"result {i}", content="matching text")
        results = mirror.search("matching", limit=3)
        assert len(results) <= 3

    def test_returns_empty_when_no_match(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", title="cooking", content="pasta")
        results = mirror.search("zzznomatch")
        assert results == []


class TestGetRecent:
    def test_returns_all_pages_ordered_by_last_edited_desc(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", last_edited_at="2026-02-01T00:00:00Z")
        _insert_page(mirror, page_id="p2", last_edited_at="2026-02-10T00:00:00Z")
        _insert_page(mirror, page_id="p3", last_edited_at="2026-01-15T00:00:00Z")
        results = mirror.get_recent()
        assert [r.id for r in results] == ["p2", "p1", "p3"]

    def test_filter_by_source_dbs(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", source_db="journal")
        _insert_page(mirror, page_id="p2", source_db="notes")
        _insert_page(mirror, page_id="p3", source_db="library")
        results = mirror.get_recent(source_dbs=["journal", "notes"])
        ids = {r.id for r in results}
        assert ids == {"p1", "p2"}

    def test_filter_by_since(self, mirror: NotionMirror) -> None:
        _insert_page(mirror, page_id="p1", last_edited_at="2026-02-10T00:00:00Z")
        _insert_page(mirror, page_id="p2", last_edited_at="2026-01-01T00:00:00Z")
        since = datetime(2026, 2, 1, tzinfo=timezone.utc)
        results = mirror.get_recent(since=since)
        assert len(results) == 1
        assert results[0].id == "p1"

    def test_limit(self, mirror: NotionMirror) -> None:
        for i in range(10):
            _insert_page(mirror, page_id=f"p{i}", last_edited_at=f"2026-02-{i+1:02d}T00:00:00Z")
        results = mirror.get_recent(limit=5)
        assert len(results) == 5

    def test_returns_empty_when_no_pages(self, mirror: NotionMirror) -> None:
        assert mirror.get_recent() == []

    def test_returns_mirror_page_shape(self, mirror: NotionMirror) -> None:
        _insert_page(
            mirror,
            page_id="p1",
            title="My Note",
            content="Some text",
            source_db="notes",
            url="https://notion.so/p1",
            created_at="2026-01-01T00:00:00Z",
            last_edited_at="2026-02-10T00:00:00Z",
        )
        results = mirror.get_recent()
        assert len(results) == 1
        p = results[0]
        assert isinstance(p, MirrorPage)
        assert p.id == "p1"
        assert p.title == "My Note"


class TestOpenMirror:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = str(tmp_path / "nested" / "dir" / "mirror.sqlite3")
        with open_mirror(nested) as m:
            assert m.page_count() == 0
        assert Path(nested).exists()


class TestSyncScript:
    def _make_config(self, tmp_path: Path) -> object:
        from patronus.config import (
            AgentConfig,
            Config,
            DigestConfig,
            EmbeddingConfig,
            NotionConfig,
            PollingConfig,
            SummarizationConfig,
            TelegramConfig,
        )
        return Config(
            digest=DigestConfig(),
            polling=PollingConfig(),
            embedding=EmbeddingConfig(),
            summarization=SummarizationConfig(),
            telegram=TelegramConfig(),
            topics={},
            agent=AgentConfig(),
            notion=NotionConfig(
                database_ids={"journal": "db-journal-id", "notes": "db-notes-id"},
                lookback_days=14,
                fallback_lookback_days=30,
                min_entries_threshold=1,
                max_chars_per_entry=5000,
                summary_model="anthropic/claude-test",
            ),
            notion_token="secret_test_token",
        )

    def _make_notion_client(self, pages_by_db: dict[str, list[dict]]) -> MagicMock:
        client = MagicMock()
        ds_responses: dict[str, MagicMock] = {}
        for db_id, pages in pages_by_db.items():
            resp = MagicMock()
            resp.get.side_effect = lambda k, d=None, _r={"has_more": False, "results": pages}: _r.get(k, d)
            resp.__getitem__ = lambda self, k, _r={"has_more": False, "results": pages}: _r[k]
            ds_id = f"ds-{db_id}"
            ds_responses[ds_id] = resp

        def _db_retrieve(database_id: str) -> dict:
            ds_id = f"ds-{database_id}"
            return {"data_sources": [{"id": ds_id}]}

        def _ds_query(data_source_id: str, **kwargs: object) -> dict:
            pages = []
            for db_id, ps in pages_by_db.items():
                if f"ds-{db_id}" == data_source_id:
                    pages = ps
                    break
            return {"results": pages, "has_more": False}

        def _blocks_list(block_id: str, **kwargs: object) -> dict:
            return {"results": [], "has_more": False}

        client.databases.retrieve.side_effect = _db_retrieve
        client.data_sources.query.side_effect = _ds_query
        client.blocks.children.list.side_effect = _blocks_list
        return client

    def _make_page(self, page_id: str, title: str) -> dict:
        return {
            "id": page_id,
            "url": f"https://notion.so/{page_id}",
            "created_time": "2026-01-01T00:00:00Z",
            "last_edited_time": "2026-02-10T00:00:00Z",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": title}],
                }
            },
        }

    def test_sync_inserts_pages_into_mirror(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync
        config = self._make_config(tmp_path)
        pages = [self._make_page("page-abc", "My Journal Entry")]
        client = self._make_notion_client({"db-journal-id": pages, "db-notes-id": []})

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            with patch("scripts.sync_notion_mirror.resolve_data_source_id", wraps=lambda c, db_id, cache: f"ds-{db_id}"):
                with patch("notion_client.Client") as MockClient:
                    MockClient.return_value = client
                    with patch("patronus.notion.NotionClient", return_value=client):
                        counts = sync(mirror, config, full=True)

            assert mirror.page_count() >= 1

    def test_incremental_sync_uses_last_synced_at(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync
        config = self._make_config(tmp_path)
        pages = [self._make_page("page-1", "Entry")]
        client = self._make_notion_client({"db-journal-id": pages, "db-notes-id": []})

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            mirror.set_last_synced_at("journal", "2026-02-09T00:00:00Z")
            with patch("notion_client.Client", return_value=client):
                with patch("scripts.sync_notion_mirror.resolve_data_source_id", side_effect=lambda c, db_id, cache: f"ds-{db_id}"):
                    sync(mirror, config, full=False)
            ts = mirror.get_last_synced_at("journal")
            assert ts is not None
            assert ts > "2026-02-09T00:00:00Z"

    def test_full_sync_resets_lookback(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import _query_pages_raw
        config = self._make_config(tmp_path)
        client = MagicMock()
        client.data_sources.query.return_value = {"results": [], "has_more": False}
        cache: dict = {}
        with patch("scripts.sync_notion_mirror.resolve_data_source_id", return_value="ds-1"):
            _query_pages_raw(client, config, "db-journal-id", 36500, cache)
        call_kwargs = client.data_sources.query.call_args[1]
        cutoff_str = call_kwargs["filter"]["last_edited_time"]["after"]
        cutoff_dt = datetime.fromisoformat(cutoff_str)
        age_days = (datetime.now(timezone.utc) - cutoff_dt.replace(tzinfo=timezone.utc)).days
        assert age_days >= 36499

    def test_sync_returns_no_config(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync
        from patronus.config import Config, DigestConfig, EmbeddingConfig, PollingConfig, SummarizationConfig, TelegramConfig
        config = Config(
            digest=DigestConfig(),
            polling=PollingConfig(),
            embedding=EmbeddingConfig(),
            summarization=SummarizationConfig(),
            telegram=TelegramConfig(),
            topics={},
            notion=None,
        )
        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            counts = sync(mirror, config)
        assert counts == {}

    def test_sync_pages_upserts_specific_page(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync_pages
        config = self._make_config(tmp_path)

        page_id = "309609f4e9ac80beb304e247ddf09787"
        formatted_id = "309609f4-e9ac-80be-b304-e247ddf09787"
        page = self._make_page(formatted_id, "Meeting Notes")
        page["parent"] = {"database_id": "db-journal-id"}

        client = MagicMock()
        client.pages.retrieve.return_value = page
        client.blocks.children.list.return_value = {
            "results": [
                {
                    "id": "block-1",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "AI summary content"}]},
                    "has_children": False,
                }
            ],
            "has_more": False,
        }

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            with patch("notion_client.Client", return_value=client):
                count = sync_pages(mirror, config, [page_id])

        assert count == 1
        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            pages = mirror.get_recent()
            assert len(pages) == 1
            assert pages[0].title == "Meeting Notes"
            assert pages[0].source_db == "journal"
            assert "AI summary content" in pages[0].content_snippet

    def test_sync_pages_resolves_source_db_from_parent(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync_pages
        config = self._make_config(tmp_path)

        page_id = "aaaabbbbccccddddeeeeffffaaaabbbb"
        formatted_id = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
        page = self._make_page(formatted_id, "Notes Entry")
        page["parent"] = {"database_id": "db-notes-id"}

        client = MagicMock()
        client.pages.retrieve.return_value = page
        client.blocks.children.list.return_value = {"results": [], "has_more": False}

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            with patch("notion_client.Client", return_value=client):
                sync_pages(mirror, config, [page_id])

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            pages = mirror.get_recent()
            assert pages[0].source_db == "notes"

    def test_sync_pages_uses_unknown_source_db_for_unrecognised_parent(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync_pages
        config = self._make_config(tmp_path)

        page_id = "11112222333344445555666677778888"
        formatted_id = "11112222-3333-4444-5555-666677778888"
        page = self._make_page(formatted_id, "Orphan Page")
        page["parent"] = {"database_id": "some-other-db-id"}

        client = MagicMock()
        client.pages.retrieve.return_value = page
        client.blocks.children.list.return_value = {"results": [], "has_more": False}

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            with patch("notion_client.Client", return_value=client):
                count = sync_pages(mirror, config, [page_id])

        assert count == 1
        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            pages = mirror.get_recent()
            assert pages[0].source_db == "unknown"

    def test_sync_pages_skips_failed_page_retrieve(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync_pages
        config = self._make_config(tmp_path)

        client = MagicMock()
        client.pages.retrieve.side_effect = Exception("Page not found")

        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            with patch("notion_client.Client", return_value=client):
                count = sync_pages(mirror, config, ["deadbeefdeadbeefdeadbeefdeadbeef"])

        assert count == 0

    def test_sync_pages_returns_zero_when_no_notion_config(self, tmp_path: Path) -> None:
        from scripts.sync_notion_mirror import sync_pages
        from patronus.config import Config, DigestConfig, EmbeddingConfig, PollingConfig, SummarizationConfig, TelegramConfig
        config = Config(
            digest=DigestConfig(),
            polling=PollingConfig(),
            embedding=EmbeddingConfig(),
            summarization=SummarizationConfig(),
            telegram=TelegramConfig(),
            topics={},
            notion=None,
        )
        with NotionMirror(str(tmp_path / "mirror.sqlite3")) as mirror:
            count = sync_pages(mirror, config, ["somepageid12345678901234567890ab"])
        assert count == 0
