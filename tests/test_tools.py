from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from patronus.config import (
    Config,
    DigestConfig,
    EmbeddingConfig,
    NotionConfig,
    PollingConfig,
    TelegramConfig,
    TopicConfig,
)
from patronus.db import Database, Item, serialize_embedding
from patronus.notion_mirror import NotionMirror
from patronus.tools import ToolRegistry
from patronus.tools.arxiv import SearchArxiv
from patronus.tools.base import Tool, ToolResult
from patronus.tools.local import (
    SearchBySource,
    SearchByTopic,
    SearchRecent,
    SearchSimilar,
    register_local_tools,
)
from patronus.tools.notion import SearchNotion, register_notion_tools
from patronus.tools.openalex import GetCitingPapers, GetReferencedPapers, SearchOpenAlex, register_openalex_tools


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_config(**overrides: object) -> Config:
    return Config(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="Technical AI/ML", description="ML research"),
            "phil": TopicConfig(name="Philosophy", description="Philosophy of mind"),
        },
        **{k: v for k, v in overrides.items()},
    )


class TestToolResult:
    def test_to_text_with_items(self) -> None:
        result = ToolResult(
            items=[
                {"title": "Paper A", "url": "https://a.com", "id": "1"},
                {"title": "Paper B", "url": "https://b.com"},
            ],
            message="Found 2 items.",
        )
        text = result.to_text()
        assert "Found 2 items." in text
        assert "Paper A" in text
        assert "https://a.com" in text
        assert "Paper B" in text
        assert "ID: 1" in text

    def test_to_text_message_only(self) -> None:
        result = ToolResult(message="No results found.")
        assert result.to_text() == "No results found."

    def test_to_text_empty(self) -> None:
        result = ToolResult()
        assert result.to_text() == "No results found."

    def test_to_text_item_with_all_fields(self) -> None:
        result = ToolResult(items=[{
            "title": "Title",
            "url": "https://x.com",
            "source": "RSS",
            "author": "Alice",
            "item_type": "paper",
            "timestamp": "2026-02-15",
            "id": "abc",
            "snippet": "Some text...",
        }])
        text = result.to_text()
        assert "Title: Title" in text
        assert "URL: https://x.com" in text
        assert "Source: RSS" in text
        assert "Author: Alice" in text
        assert "Type: paper" in text
        assert "Date: 2026-02-15" in text
        assert "ID: abc" in text
        assert "Snippet: Some text..." in text


class TestToolRegistry:
    def _make_tool(self, name: str = "test_tool") -> Tool:
        class FakeTool(Tool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return f"A fake tool named {name}"

            @property
            def input_schema(self) -> dict:
                return {"type": "object", "properties": {}}

            def execute(self, **params: object) -> ToolResult:
                return ToolResult(message=f"Executed {name}")

        return FakeTool()

    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = self._make_tool("my_tool")
        registry.register(tool)

        assert registry.get("my_tool") is tool
        assert registry.get("nonexistent") is None

    def test_get_definitions(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_tool("tool_a"))
        registry.register(self._make_tool("tool_b"))

        defs = registry.get_definitions()
        assert len(defs) == 2
        names = {d["name"] for d in defs}
        assert names == {"tool_a", "tool_b"}
        assert all("description" in d for d in defs)
        assert all("input_schema" in d for d in defs)

    def test_tool_names(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_tool("a"))
        registry.register(self._make_tool("b"))
        assert set(registry.tool_names) == {"a", "b"}

    def test_execute_success(self) -> None:
        registry = ToolRegistry()
        registry.register(self._make_tool("my_tool"))
        result = registry.execute("my_tool")
        assert result.message == "Executed my_tool"

    def test_execute_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert "Unknown tool" in result.message

    def test_execute_catches_exception(self) -> None:
        class FailingTool(Tool):
            @property
            def name(self) -> str:
                return "fail"

            @property
            def description(self) -> str:
                return ""

            @property
            def input_schema(self) -> dict:
                return {}

            def execute(self, **params: object) -> ToolResult:
                raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register(FailingTool())
        result = registry.execute("fail")
        assert "failed with an internal error" in result.message

    def test_to_definition(self) -> None:
        tool = self._make_tool("my_tool")
        defn = tool.to_definition()
        assert defn["name"] == "my_tool"
        assert "fake tool" in defn["description"]
        assert "input_schema" in defn


class TestSearchSimilar:
    @patch("patronus.tools.local.embed_text")
    def test_returns_similar_items(self, mock_embed: MagicMock, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        emb1 = _unit_vec(1.0, 0.0, 0.0)
        emb2 = _unit_vec(0.0, 1.0, 0.0)
        db.add_item(url="https://a.com", source_type="rss", title="ML Paper",
                     text="ML content", embedding=emb1, timestamp="2026-02-15T00:00:00Z")
        db.add_item(url="https://b.com", source_type="rss", title="Philosophy Essay",
                     text="Philosophy content", embedding=emb2, timestamp="2026-02-15T00:00:00Z")

        mock_embed.return_value = _unit_vec(0.95, 0.05, 0.0)
        tool = SearchSimilar(config, db)
        result = tool.execute(query="machine learning")

        assert len(result.items) == 2
        assert result.items[0]["title"] == "ML Paper"
        db.close()

    @patch("patronus.tools.local.embed_text")
    def test_empty_query_returns_error(self, mock_embed: MagicMock, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        tool = SearchSimilar(config, db)
        result = tool.execute(query="")
        assert "required" in result.message.lower()
        db.close()

    @patch("patronus.tools.local.embed_text")
    def test_respects_n_param(self, mock_embed: MagicMock, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        for i in range(5):
            emb = _unit_vec(1.0, float(i) * 0.1, 0.0)
            db.add_item(url=f"https://item{i}.com", source_type="rss", title=f"Item {i}",
                         text="Content", embedding=emb, timestamp="2026-02-15T00:00:00Z")

        mock_embed.return_value = _unit_vec(1.0, 0.0, 0.0)
        tool = SearchSimilar(config, db)
        result = tool.execute(query="test", n=2)
        assert len(result.items) == 2
        db.close()

    def test_tool_definition(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        tool = SearchSimilar(config, db)
        assert tool.name == "search_similar"
        defn = tool.to_definition()
        assert "query" in defn["input_schema"]["properties"]
        db.close()


class TestSearchRecent:
    def test_returns_recent_items(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        now = datetime.now(timezone.utc)
        recent_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        old_ts = "2020-01-01T00:00:00Z"

        db.add_item(url="https://new.com", source_type="rss", title="New",
                     text="text", timestamp=recent_ts)
        db.add_item(url="https://old.com", source_type="rss", title="Old",
                     text="text", timestamp=old_ts)

        tool = SearchRecent(db)
        result = tool.execute(days=3)

        assert len(result.items) == 1
        assert result.items[0]["title"] == "New"
        db.close()

    def test_respects_n_limit(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        for i in range(5):
            db.add_item(url=f"https://item{i}.com", source_type="rss", title=f"Item {i}",
                         text="text", timestamp=ts)

        tool = SearchRecent(db)
        result = tool.execute(days=3, n=2)
        assert len(result.items) == 2
        db.close()

    def test_no_recent_items(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        db.add_item(url="https://old.com", source_type="rss", title="Old",
                     text="text", timestamp="2020-01-01T00:00:00Z")

        tool = SearchRecent(db)
        result = tool.execute(days=1)
        assert len(result.items) == 0
        db.close()

    def test_tool_metadata(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        tool = SearchRecent(db)
        assert tool.name == "search_recent"
        assert "recent" in tool.description.lower()
        db.close()


class TestSearchByTopic:
    def test_returns_items_matching_topic(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        item_id = db.add_item(url="https://ml.com", source_type="rss", title="ML Paper",
                               text="ML text", timestamp="2026-02-15T00:00:00Z")
        with db._session() as session:
            item = session.get(Item, item_id)
            item.topic_cluster = "ml"
            session.add(item)
            session.commit()

        db.add_item(url="https://phil.com", source_type="rss", title="Phil Essay",
                     text="Phil text", timestamp="2026-02-15T00:00:00Z")

        tool = SearchByTopic(config, db)
        result = tool.execute(topic="ml")

        assert len(result.items) == 1
        assert result.items[0]["title"] == "ML Paper"
        db.close()

    def test_empty_topic_returns_error(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        tool = SearchByTopic(config, db)
        result = tool.execute(topic="")
        assert "required" in result.message.lower()
        db.close()

    def test_description_includes_topics(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        tool = SearchByTopic(config, db)
        assert "ml" in tool.description
        assert "phil" in tool.description
        db.close()


class TestSearchBySource:
    def test_filter_by_source_type(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        db.add_item(url="https://rss.com", source_type="rss", title="RSS Item",
                     text="text", timestamp="2026-02-15T00:00:00Z")
        db.add_item(url="https://manual.com", source_type="manual", title="Manual Item",
                     text="text", timestamp="2026-02-15T00:00:00Z")

        tool = SearchBySource(db)
        result = tool.execute(source_type="rss")

        assert len(result.items) == 1
        assert result.items[0]["title"] == "RSS Item"
        db.close()

    def test_filter_by_source_name(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        db.add_item(url="https://a.com", source_type="rss", title="Blog A", source="AI Blog",
                     text="text", timestamp="2026-02-15T00:00:00Z")
        db.add_item(url="https://b.com", source_type="rss", title="Blog B", source="Cooking Blog",
                     text="text", timestamp="2026-02-15T00:00:00Z")

        tool = SearchBySource(db)
        result = tool.execute(source_name="AI")

        assert len(result.items) == 1
        assert result.items[0]["title"] == "Blog A"
        db.close()

    def test_no_filters(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        db.add_item(url="https://a.com", source_type="rss", title="Item A",
                     text="text", timestamp="2026-02-15T00:00:00Z")
        db.add_item(url="https://b.com", source_type="rss", title="Item B",
                     text="text", timestamp="2026-02-15T00:00:00Z")

        tool = SearchBySource(db)
        result = tool.execute()
        assert len(result.items) == 2
        db.close()


def _make_fake_feed(entries: list[dict]) -> object:
    class FakeEntry:
        def __init__(self, data: dict) -> None:
            self.id = data.get("id", "")
            self.title = data.get("title", "")
            self.summary = data.get("summary", "")
            self.published = data.get("published", "")
            self.authors = [type("A", (), {"name": n})() for n in data.get("authors", [])]
            self.tags = [{"term": t} for t in data.get("tags", [])]

    class FakeFeed:
        def __init__(self, entries_data: list[dict]) -> None:
            self.entries = [FakeEntry(e) for e in entries_data]

    return FakeFeed(entries)


FAKE_ENTRIES = [
    {
        "id": "http://arxiv.org/abs/2301.00001v2",
        "title": "Attention Is All You Need",
        "summary": "We propose the Transformer, a model based solely on attention mechanisms.",
        "published": "2023-01-15T00:00:00Z",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "tags": ["cs.LG", "cs.AI"],
    },
    {
        "id": "http://arxiv.org/abs/2301.00002v1",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "summary": "We introduce BERT for language representation pre-training.",
        "published": "2023-01-16T00:00:00Z",
        "authors": ["Devlin, J."],
        "tags": ["cs.CL"],
    },
]


class TestSearchArxiv:
    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_returns_results(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed(FAKE_ENTRIES)

        tool = SearchArxiv()
        result = tool.execute(query="transformer attention")

        assert len(result.items) == 2
        assert result.items[0]["title"] == "Attention Is All You Need"
        assert result.items[0]["url"] == "https://arxiv.org/abs/2301.00001"
        assert "Vaswani" in result.items[0]["author"]
        assert result.items[0]["item_type"] == "paper"
        assert result.items[0]["source"] == "arxiv"
        assert "transformer" in result.items[0]["snippet"].lower()

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_result_id_is_canonical_url(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed(FAKE_ENTRIES[:1])
        tool = SearchArxiv()
        result = tool.execute(query="attention")
        assert result.items[0]["id"] == "https://arxiv.org/abs/2301.00001"

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_does_not_write_to_db(self, mock_parse: MagicMock, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        mock_parse.return_value = _make_fake_feed(FAKE_ENTRIES[:1])
        SearchArxiv().execute(query="attention")
        assert db.get_item_by_url("https://arxiv.org/abs/2301.00001") is None
        db.close()

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_empty_query_returns_error(self, mock_parse: MagicMock) -> None:
        tool = SearchArxiv()
        result = tool.execute(query="")
        assert "required" in result.message.lower()
        mock_parse.assert_not_called()

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_no_results_returns_message(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        result = SearchArxiv().execute(query="zzznoresults")
        assert len(result.items) == 0
        assert "no" in result.message.lower()

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_api_failure_returns_error(self, mock_parse: MagicMock) -> None:
        mock_parse.side_effect = OSError("connection refused")
        result = SearchArxiv().execute(query="attention")
        assert len(result.items) == 0
        assert "failed" in result.message.lower()

    @patch("patronus.tools.arxiv.time.sleep")
    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_rate_limit_retries_once_then_fails(self, mock_parse: MagicMock, mock_sleep: MagicMock) -> None:
        class RateLimitedFeed:
            status = 429
            entries: list = []
            bozo = False

        mock_parse.return_value = RateLimitedFeed()
        result = SearchArxiv().execute(query="attention")

        assert mock_parse.call_count == 2
        mock_sleep.assert_called_once()
        assert "rate limit" in result.message.lower()
        assert result.items == []

    @patch("patronus.tools.arxiv.time.sleep")
    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_rate_limit_succeeds_on_retry(self, mock_parse: MagicMock, mock_sleep: MagicMock) -> None:
        class RateLimitedFeed:
            status = 429
            entries: list = []
            bozo = False

        mock_parse.side_effect = [RateLimitedFeed(), _make_fake_feed(FAKE_ENTRIES[:1])]
        result = SearchArxiv().execute(query="attention")

        assert mock_parse.call_count == 2
        mock_sleep.assert_called_once()
        assert len(result.items) == 1

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_strips_version_from_url(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([{
            "id": "http://arxiv.org/abs/2301.99999v3",
            "title": "Test Paper",
            "summary": "Abstract text.",
            "published": "2023-01-01T00:00:00Z",
            "authors": [],
            "tags": [],
        }])
        result = SearchArxiv().execute(query="test")
        assert result.items[0]["url"] == "https://arxiv.org/abs/2301.99999"

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_sort_by_recency_sets_sortby_param(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        SearchArxiv().execute(query="attention", sort_by="recency")
        call_url = mock_parse.call_args[0][0]
        assert "sortBy=submittedDate" in call_url
        assert "sortOrder=descending" in call_url

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_sort_by_relevance_is_default(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        SearchArxiv().execute(query="attention")
        call_url = mock_parse.call_args[0][0]
        assert "sortBy=relevance" in call_url

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_category_filter_included_in_query(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        SearchArxiv().execute(query="attention", category="cs.LG")
        call_url = mock_parse.call_args[0][0]
        assert "cat:cs.LG" in call_url

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_multiword_query_ands_each_term(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        SearchArxiv().execute(query="reward hacking")
        call_url = mock_parse.call_args[0][0]
        assert "all:reward" in call_url
        assert "all:hacking" in call_url
        assert "AND" in call_url

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_single_word_query_uses_all_prefix(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        SearchArxiv().execute(query="transformers")
        call_url = mock_parse.call_args[0][0]
        assert "all:transformers" in call_url

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_days_filter_included_in_query(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = _make_fake_feed([])
        SearchArxiv().execute(query="attention", days=7)
        call_url = mock_parse.call_args[0][0]
        assert "submittedDate" in call_url

    @patch("patronus.tools.arxiv.feedparser.parse")
    def test_journal_ref_included_when_present(self, mock_parse: MagicMock) -> None:
        entry_with_journal = {**FAKE_ENTRIES[0], "journal_ref": "NeurIPS 2023"}

        class FakeEntryWithJournal:
            def __init__(self) -> None:
                self.id = entry_with_journal["id"]
                self.title = entry_with_journal["title"]
                self.summary = entry_with_journal["summary"]
                self.published = entry_with_journal["published"]
                self.authors = [type("A", (), {"name": n})() for n in entry_with_journal["authors"]]
                self.tags = [{"term": t} for t in entry_with_journal["tags"]]
                self.arxiv_journal_ref = entry_with_journal["journal_ref"]

        class FakeFeedWithJournal:
            entries = [FakeEntryWithJournal()]

        mock_parse.return_value = FakeFeedWithJournal()
        result = SearchArxiv().execute(query="attention")
        assert result.items[0].get("journal_ref") == "NeurIPS 2023"

    def test_tool_metadata(self) -> None:
        tool = SearchArxiv()
        assert tool.name == "search_arxiv"
        defn = tool.to_definition()
        props = defn["input_schema"]["properties"]
        assert "query" in props
        assert "n" in props
        assert "sort_by" in props
        assert "category" in props
        assert "days" in props
        assert "citation" in tool.description.lower()


def _save_digest_with_item(db: Database, item_id: str, days_ago: int = 0) -> None:
    generated_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.save_digest(
        generated_at=generated_at,
        item_count=1,
        formatted_text=None,
        items=[{"item_id": item_id, "summary": "s", "score": 1.0, "matched_topic": ""}],
    )


class TestSearchSimilarDigestFilter:
    @patch("patronus.tools.local.embed_text")
    def test_excludes_item_digested_recently(self, mock_embed: MagicMock, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        emb = _unit_vec(1.0, 0.0, 0.0)
        fresh_id = db.add_item(url="https://fresh.com", source_type="rss", title="Fresh",
                               text="text", embedding=emb, timestamp="2026-02-15T00:00:00Z")
        recent_digest_id = db.add_item(url="https://recent.com", source_type="rss", title="RecentlyDigested",
                                       text="text", embedding=emb, timestamp="2026-02-15T00:00:00Z")
        _save_digest_with_item(db, recent_digest_id, days_ago=5)

        mock_embed.return_value = emb
        tool = SearchSimilar(config, db)
        result = tool.execute(query="test")

        ids = {item["id"] for item in result.items}
        assert fresh_id in ids
        assert recent_digest_id not in ids
        db.close()

    @patch("patronus.tools.local.embed_text")
    def test_includes_item_digested_outside_lookback(self, mock_embed: MagicMock, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        emb = _unit_vec(1.0, 0.0, 0.0)
        item_id = db.add_item(url="https://old.com", source_type="rss", title="OldDigest",
                              text="text", embedding=emb, timestamp="2026-02-15T00:00:00Z")
        _save_digest_with_item(db, item_id, days_ago=45)

        mock_embed.return_value = emb
        tool = SearchSimilar(config, db)
        result = tool.execute(query="test")

        assert any(item["id"] == item_id for item in result.items)
        db.close()


class TestSearchRecentDigestFilter:
    def test_excludes_item_digested_recently(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        fresh_id = db.add_item(url="https://fresh.com", source_type="rss", title="Fresh", timestamp=ts)
        recent_digest_id = db.add_item(url="https://recent.com", source_type="rss", title="RecentlyDigested", timestamp=ts)
        _save_digest_with_item(db, recent_digest_id, days_ago=5)

        tool = SearchRecent(db)
        result = tool.execute(days=3)

        ids = {item["id"] for item in result.items}
        assert fresh_id in ids
        assert recent_digest_id not in ids
        db.close()


class TestSearchByTopicDigestFilter:
    def test_excludes_item_digested_recently(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()

        fresh_id = db.add_item(url="https://fresh.com", source_type="rss", title="Fresh",
                               timestamp="2026-02-15T00:00:00Z")
        recent_digest_id = db.add_item(url="https://recent.com", source_type="rss", title="RecentlyDigested",
                                       timestamp="2026-02-15T00:00:00Z")
        for iid in [fresh_id, recent_digest_id]:
            with db._session() as session:
                item = session.get(Item, iid)
                item.topic_cluster = "ml"
                session.add(item)
                session.commit()
        _save_digest_with_item(db, recent_digest_id, days_ago=5)

        tool = SearchByTopic(config, db)
        result = tool.execute(topic="ml")

        ids = {item["id"] for item in result.items}
        assert fresh_id in ids
        assert recent_digest_id not in ids
        db.close()


class TestSearchBySourceDigestFilter:
    def test_excludes_item_digested_recently(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")

        fresh_id = db.add_item(url="https://fresh.com", source_type="rss", title="Fresh",
                               timestamp="2026-02-15T00:00:00Z")
        recent_digest_id = db.add_item(url="https://recent.com", source_type="rss", title="RecentlyDigested",
                                       timestamp="2026-02-15T00:00:00Z")
        _save_digest_with_item(db, recent_digest_id, days_ago=5)

        tool = SearchBySource(db)
        result = tool.execute(source_type="rss")

        ids = {item["id"] for item in result.items}
        assert fresh_id in ids
        assert recent_digest_id not in ids
        db.close()


class TestRegisterLocalTools:
    def test_registers_all_four_tools(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        registry = ToolRegistry()
        register_local_tools(registry, config, db)

        assert set(registry.tool_names) == {
            "search_similar", "search_recent", "search_by_topic", "search_by_source",
        }
        db.close()


def _make_notion_config(mirror_path: str = "", database_ids: dict | None = None) -> Config:
    return Config(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        telegram=TelegramConfig(),
        topics={},
        notion=NotionConfig(
            database_ids=database_ids or {
                "journal": "uuid-journal",
                "notes": "uuid-notes",
                "reviews": "uuid-reviews",
            },
            mirror_path=mirror_path,
        ),
    )


def _insert_notion_page(
    mirror: NotionMirror,
    *,
    page_id: str = "page1",
    title: str = "Test Page",
    content: str = "Some content",
    source_db: str = "uuid-journal",
    url: str = "https://notion.so/page1",
    last_edited_at: str = "2026-02-10T00:00:00Z",
) -> None:
    mirror.upsert_page(
        page_id=page_id,
        title=title,
        content=content,
        source_db=source_db,
        url=url,
        created_at="2026-01-01T00:00:00Z",
        last_edited_at=last_edited_at,
    )


class TestSearchNotion:
    def test_tool_metadata(self, tmp_path: Path) -> None:
        config = _make_notion_config(mirror_path=str(tmp_path / "mirror.sqlite3"))
        tool = SearchNotion(config)
        assert tool.name == "search_notion"
        assert "notion" in tool.description.lower()
        schema = tool.to_definition()
        assert "query" in schema["input_schema"]["properties"]
        assert "n" in schema["input_schema"]["properties"]

    def test_missing_mirror_returns_error(self, tmp_path: Path) -> None:
        config = _make_notion_config(mirror_path=str(tmp_path / "nonexistent.sqlite3"))
        tool = SearchNotion(config)
        result = tool.execute(query="machine learning")
        assert "not found" in result.message.lower()
        assert result.items == []

    def test_empty_query_returns_error(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path):
            pass
        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="")
        assert "required" in result.message.lower()
        assert result.items == []

    def test_returns_matching_results(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(
                mirror,
                page_id="p1",
                title="Mechanistic Interpretability",
                content="circuits and features in neural networks",
                source_db="uuid-journal",
                url="https://notion.so/p1",
            )
            _insert_notion_page(
                mirror,
                page_id="p2",
                title="Cooking Pasta",
                content="boil the water add salt",
                source_db="uuid-notes",
                url="https://notion.so/p2",
            )

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="neural networks circuits")

        assert len(result.items) == 1
        item = result.items[0]
        assert item["title"] == "Mechanistic Interpretability"
        assert item["url"] == "https://notion.so/p1"
        assert "snippet" in item

    def test_item_id_prefixed_with_notion(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(mirror, page_id="abc123", content="interpretability research")

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="interpretability")

        assert len(result.items) == 1
        assert result.items[0]["id"] == "notion:abc123"

    def test_source_db_uuid_mapped_to_readable_name(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(
                mirror,
                page_id="p1",
                content="alignment and safety",
                source_db="uuid-notes",
            )

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="alignment")

        assert len(result.items) == 1
        assert "notes" in result.items[0]["source"]

    def test_unknown_source_db_falls_back_to_uuid(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(
                mirror,
                page_id="p1",
                content="some content about philosophy",
                source_db="unknown-uuid-xyz",
            )

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="philosophy")

        assert len(result.items) == 1
        assert "unknown-uuid-xyz" in result.items[0]["source"]

    def test_respects_n_param(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            for i in range(5):
                _insert_notion_page(
                    mirror,
                    page_id=f"p{i}",
                    title=f"Note {i}",
                    content="transformer attention mechanism",
                    source_db="uuid-journal",
                    url=f"https://notion.so/p{i}",
                )

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="transformer attention", n=3)

        assert len(result.items) <= 3

    def test_no_matches_returns_message(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(mirror, page_id="p1", title="Pasta Recipe", content="boil water")

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="zzznomatch")

        assert result.items == []
        assert "no" in result.message.lower()

    def test_result_includes_timestamp(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(
                mirror,
                page_id="p1",
                content="mechanistic interpretability features",
                last_edited_at="2026-02-10T12:00:00Z",
            )

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="interpretability")

        assert len(result.items) == 1
        assert result.items[0]["timestamp"] == "2026-02-10T12:00:00Z"

    def test_to_text_includes_key_fields(self, tmp_path: Path) -> None:
        mirror_path = str(tmp_path / "mirror.sqlite3")
        with NotionMirror(mirror_path) as mirror:
            _insert_notion_page(
                mirror,
                page_id="p1",
                title="AI Safety Notes",
                content="deceptive alignment and corrigibility",
                url="https://notion.so/p1",
                source_db="uuid-journal",
            )

        config = _make_notion_config(mirror_path=mirror_path)
        tool = SearchNotion(config)
        result = tool.execute(query="deceptive alignment")
        text = result.to_text()

        assert "AI Safety Notes" in text
        assert "https://notion.so/p1" in text
        assert "journal" in text


class TestRegisterNotionTools:
    def test_registers_tool_when_mirror_path_set(self, tmp_path: Path) -> None:
        config = _make_notion_config(mirror_path=str(tmp_path / "mirror.sqlite3"))
        registry = ToolRegistry()
        register_notion_tools(registry, config)
        assert "search_notion" in registry.tool_names

    def test_does_not_register_when_mirror_path_empty(self) -> None:
        config = _make_notion_config(mirror_path="")
        registry = ToolRegistry()
        register_notion_tools(registry, config)
        assert "search_notion" not in registry.tool_names

    def test_does_not_register_when_no_notion_config(self) -> None:
        config = _make_config()
        registry = ToolRegistry()
        register_notion_tools(registry, config)
        assert "search_notion" not in registry.tool_names


FAKE_OPENALEX_WORKS = [
    {
        "id": "https://openalex.org/W1111111111",
        "title": "Attention and Consciousness",
        "doi": "https://doi.org/10.1234/fake.001",
        "abstract_inverted_index": {"A": [0], "study": [1], "of": [2], "attention": [3], "and": [4], "consciousness": [5], "in": [6], "the": [7], "human": [8], "brain": [9]},
        "abstract": "A study of attention and consciousness in the human brain.",
        "authorships": [
            {"author": {"display_name": "Jane Doe"}},
            {"author": {"display_name": "John Smith"}},
        ],
        "publication_date": "2023-06-15",
        "cited_by_count": 42,
        "topics": [
            {"display_name": "Consciousness"},
            {"display_name": "Neuroscience"},
        ],
        "primary_location": {"landing_page_url": "https://example.com/paper1"},
    },
    {
        "id": "https://openalex.org/W2222222222",
        "title": "Language and Thought",
        "doi": None,
        "abstract": "An exploration of the relationship between language and thought.",
        "authorships": [
            {"author": {"display_name": "Alice Brown"}},
        ],
        "publication_date": "2022-03-10",
        "cited_by_count": 15,
        "topics": [
            {"display_name": "Linguistics"},
        ],
        "primary_location": {"landing_page_url": "https://example.com/paper2"},
    },
]


def _make_mock_works_chain(results: list) -> MagicMock:
    mock_chain = MagicMock()
    mock_chain.search.return_value = mock_chain
    mock_chain.sort.return_value = mock_chain
    mock_chain.filter.return_value = mock_chain
    mock_chain.get.return_value = results
    return mock_chain


class TestSearchOpenAlex:
    @patch("patronus.tools.openalex.Works")
    def test_returns_results(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain(FAKE_OPENALEX_WORKS)

        tool = SearchOpenAlex(config)
        result = tool.execute(query="consciousness")

        assert len(result.items) == 2
        assert result.items[0]["title"] == "Attention and Consciousness"
        assert result.items[0]["url"] == "https://doi.org/10.1234/fake.001"
        assert "Jane Doe" in result.items[0]["author"]
        assert result.items[0]["item_type"] == "paper"
        assert result.items[0]["source"] == "openalex"
        assert result.items[0]["citation_count"] == 42
        assert "Consciousness" in result.items[0]["topics"]
        assert "consciousness" in result.items[0]["snippet"].lower()

    @patch("patronus.tools.openalex.Works")
    def test_result_id_is_canonical_url(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain(FAKE_OPENALEX_WORKS[:1])
        result = SearchOpenAlex(config).execute(query="consciousness")
        assert result.items[0]["id"] == "https://doi.org/10.1234/fake.001"

    @patch("patronus.tools.openalex.Works")
    def test_does_not_write_to_db(self, MockWorks: MagicMock, tmp_path: Path) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain(FAKE_OPENALEX_WORKS[:1])
        SearchOpenAlex(config).execute(query="consciousness")
        assert db.get_item_by_url("https://doi.org/10.1234/fake.001") is None
        db.close()

    @patch("patronus.tools.openalex.Works")
    def test_falls_back_to_landing_page_url_when_no_doi(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain(FAKE_OPENALEX_WORKS[1:])
        result = SearchOpenAlex(config).execute(query="language")
        assert result.items[0]["url"] == "https://example.com/paper2"

    @patch("patronus.tools.openalex.Works")
    def test_empty_query_returns_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        result = SearchOpenAlex(config).execute(query="")
        assert "required" in result.message.lower()
        MockWorks.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_no_results_returns_message(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain([])
        result = SearchOpenAlex(config).execute(query="zzznoresults")
        assert len(result.items) == 0
        assert "no" in result.message.lower()

    @patch("patronus.tools.openalex.Works")
    def test_api_failure_returns_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.side_effect = OSError("connection refused")
        result = SearchOpenAlex(config).execute(query="consciousness")
        assert len(result.items) == 0
        assert "failed" in result.message.lower()

    @patch("patronus.tools.openalex.Works")
    def test_sort_by_citations_passes_correct_sort(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention", sort_by="citations")
        mock_chain.sort.assert_called_once_with(cited_by_count="desc")

    @patch("patronus.tools.openalex.Works")
    def test_sort_by_recency_passes_correct_sort(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention", sort_by="recency")
        mock_chain.sort.assert_called_once_with(publication_date="desc")

    @patch("patronus.tools.openalex.Works")
    def test_relevance_does_not_call_sort(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention", sort_by="relevance")
        mock_chain.sort.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_from_publication_year_applies_filter(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention", from_publication_year=2022)
        mock_chain.filter.assert_called_once_with(from_publication_date="2022-01-01")

    @patch("patronus.tools.openalex.Works")
    def test_no_from_year_does_not_call_filter(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention")
        mock_chain.filter.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_field_filter_ai(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="transformers", field="ai")
        mock_chain.filter.assert_called_once_with(topics={"subfield": {"id": "1702"}})

    @patch("patronus.tools.openalex.Works")
    def test_field_filter_philosophy(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="consciousness", field="philosophy")
        mock_chain.filter.assert_called_once_with(topics={"subfield": {"id": "1211"}})

    @patch("patronus.tools.openalex.Works")
    def test_no_field_does_not_call_filter(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention")
        mock_chain.filter.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_field_and_year_both_apply_filters(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_chain
        SearchOpenAlex(config).execute(query="attention", field="ai", from_publication_year=2023)
        assert mock_chain.filter.call_count == 2


class TestGetCitingPapers:
    @patch("patronus.tools.openalex.Works")
    def test_returns_citing_papers(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_chain = _make_mock_works_chain(FAKE_OPENALEX_WORKS)
        MockWorks.return_value = mock_chain

        result = GetCitingPapers(config).execute(doi_or_id="W2741809807")

        mock_chain.filter.assert_called_once_with(cites="W2741809807")
        assert len(result.items) == 2
        assert result.items[0]["source"] == "openalex"

    @patch("patronus.tools.openalex.Works")
    def test_result_id_is_canonical_url(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain(FAKE_OPENALEX_WORKS[:1])
        result = GetCitingPapers(config).execute(doi_or_id="W2741809807")
        assert result.items[0]["id"] == "https://doi.org/10.1234/fake.001"

    @patch("patronus.tools.openalex.Works")
    def test_does_not_write_to_db(self, MockWorks: MagicMock, tmp_path: Path) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain(FAKE_OPENALEX_WORKS[:1])
        GetCitingPapers(config).execute(doi_or_id="W2741809807")
        assert db.get_item_by_url("https://doi.org/10.1234/fake.001") is None
        db.close()

    @patch("patronus.tools.openalex.Works")
    def test_resolves_bare_doi_to_openalex_id(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock_lookup = MagicMock()
        mock_lookup.__getitem__ = MagicMock(return_value={"id": "https://openalex.org/W9999999"})
        mock_chain = _make_mock_works_chain([])
        MockWorks.return_value = mock_lookup
        mock_lookup.filter.return_value = mock_chain
        mock_chain.sort.return_value = mock_chain
        mock_chain.get.return_value = []

        GetCitingPapers(config).execute(doi_or_id="10.1234/foo")
        mock_lookup.filter.assert_called_once_with(cites="W9999999")

    @patch("patronus.tools.openalex.Works")
    def test_openalex_id_passed_through_without_lookup(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain([])
        GetCitingPapers(config).execute(doi_or_id="W2741809807")
        MockWorks.return_value.filter.assert_called_once_with(cites="W2741809807")

    @patch("patronus.tools.openalex.Works")
    def test_default_sort_is_recency(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain([])
        GetCitingPapers(config).execute(doi_or_id="W2741809807")
        MockWorks.return_value.sort.assert_called_once_with(publication_date="desc")

    @patch("patronus.tools.openalex.Works")
    def test_sort_by_citations(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain([])
        GetCitingPapers(config).execute(doi_or_id="W2741809807", sort_by="citations")
        MockWorks.return_value.sort.assert_called_once_with(cited_by_count="desc")

    @patch("patronus.tools.openalex.Works")
    def test_empty_id_returns_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        result = GetCitingPapers(config).execute(doi_or_id="")
        assert "required" in result.message.lower()
        MockWorks.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_arbitrary_url_returns_clear_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        result = GetCitingPapers(config).execute(doi_or_id="https://www.anthropic.com/research/measuring-agent-autonomy")
        assert "could not recognise" in result.message.lower()
        assert "OpenAlex" in result.message
        MockWorks.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_arxiv_url_resolved_correctly(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        mock = MagicMock()
        mock.__getitem__.return_value = {"id": "https://openalex.org/W9999999"}
        mock.filter.return_value = _make_mock_works_chain([])
        mock.filter.return_value.sort.return_value = mock.filter.return_value
        MockWorks.return_value = mock

        GetCitingPapers(config).execute(doi_or_id="https://arxiv.org/abs/1706.03762")
        mock.__getitem__.assert_called_once_with("https://doi.org/10.48550/arxiv.1706.03762")

    @patch("patronus.tools.openalex.Works")
    def test_no_results_returns_message(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = _make_mock_works_chain([])
        result = GetCitingPapers(config).execute(doi_or_id="W9999999")
        assert len(result.items) == 0
        assert "no" in result.message.lower()

    @patch("patronus.tools.openalex.Works")
    def test_api_failure_returns_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.side_effect = OSError("connection refused")
        result = GetCitingPapers(config).execute(doi_or_id="W2741809807")
        assert "failed" in result.message.lower()


class TestGetReferencedPapers:
    def _make_mock_works(self, ref_ids: list[str], fetch_results: list) -> MagicMock:
        mock = MagicMock()
        mock.__getitem__.return_value = {"referenced_works": ref_ids}
        mock_chain = _make_mock_works_chain(fetch_results)
        mock.filter.return_value = mock_chain
        return mock

    @patch("patronus.tools.openalex.Works")
    def test_returns_referenced_papers(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        ref_ids = ["https://openalex.org/W1111111111", "https://openalex.org/W2222222222"]
        MockWorks.return_value = self._make_mock_works(ref_ids, FAKE_OPENALEX_WORKS)

        result = GetReferencedPapers(config).execute(doi_or_id="W9999999")

        assert len(result.items) == 2
        assert result.items[0]["source"] == "openalex"
        assert "2 total in bibliography" in result.message

    @patch("patronus.tools.openalex.Works")
    def test_result_id_is_canonical_url(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = self._make_mock_works(
            ["https://openalex.org/W1111111111"], FAKE_OPENALEX_WORKS[:1]
        )
        result = GetReferencedPapers(config).execute(doi_or_id="W9999999")
        assert result.items[0]["id"] == "https://doi.org/10.1234/fake.001"

    @patch("patronus.tools.openalex.Works")
    def test_does_not_write_to_db(self, MockWorks: MagicMock, tmp_path: Path) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = self._make_mock_works(
            ["https://openalex.org/W1111111111"], FAKE_OPENALEX_WORKS[:1]
        )
        GetReferencedPapers(config).execute(doi_or_id="W9999999")
        assert db.get_item_by_url("https://doi.org/10.1234/fake.001") is None
        db.close()

    @patch("patronus.tools.openalex.Works")
    def test_fetches_by_id_list(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        ref_ids = ["https://openalex.org/W1111111111", "https://openalex.org/W2222222222"]
        mock = self._make_mock_works(ref_ids, [])
        MockWorks.return_value = mock

        GetReferencedPapers(config).execute(doi_or_id="W9999999")
        mock.filter.assert_called_once_with(openalex="W1111111111|W2222222222")

    @patch("patronus.tools.openalex.Works")
    def test_default_sort_is_citations(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = self._make_mock_works(["https://openalex.org/W111"], [])
        GetReferencedPapers(config).execute(doi_or_id="W9999999")
        mock_chain = MockWorks.return_value.filter.return_value
        mock_chain.sort.assert_called_once_with(cited_by_count="desc")

    @patch("patronus.tools.openalex.Works")
    def test_sort_by_recency(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = self._make_mock_works(["https://openalex.org/W111"], [])
        GetReferencedPapers(config).execute(doi_or_id="W9999999", sort_by="recency")
        mock_chain = MockWorks.return_value.filter.return_value
        mock_chain.sort.assert_called_once_with(publication_date="desc")

    @patch("patronus.tools.openalex.Works")
    def test_no_references_returns_message(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value = self._make_mock_works([], [])
        result = GetReferencedPapers(config).execute(doi_or_id="W9999999")
        assert len(result.items) == 0
        assert "no references" in result.message.lower()

    @patch("patronus.tools.openalex.Works")
    def test_empty_id_returns_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        result = GetReferencedPapers(config).execute(doi_or_id="")
        assert "required" in result.message.lower()
        MockWorks.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_arbitrary_url_returns_clear_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        result = GetReferencedPapers(config).execute(doi_or_id="https://www.anthropic.com/research/measuring-agent-autonomy")
        assert "could not recognise" in result.message.lower()
        MockWorks.assert_not_called()

    @patch("patronus.tools.openalex.Works")
    def test_lookup_failure_returns_error(self, MockWorks: MagicMock) -> None:
        config = _make_config(openalex_api_key="test-key")
        MockWorks.return_value.__getitem__.side_effect = OSError("not found")
        result = GetReferencedPapers(config).execute(doi_or_id="W9999999")
        assert "failed" in result.message.lower()


class TestRegisterOpenAlexTools:
    def test_registers_tool_when_api_key_set(self) -> None:
        config = _make_config(openalex_api_key="test-key")
        registry = ToolRegistry()
        register_openalex_tools(registry, config)
        assert "search_openalex" in registry.tool_names

    def test_registers_all_tools_when_api_key_set(self) -> None:
        config = _make_config(openalex_api_key="test-key")
        registry = ToolRegistry()
        register_openalex_tools(registry, config)
        assert "get_citing_papers" in registry.tool_names
        assert "get_referenced_papers" in registry.tool_names

    def test_does_not_register_when_no_api_key(self) -> None:
        config = _make_config(openalex_api_key="")
        registry = ToolRegistry()
        register_openalex_tools(registry, config)
        assert "search_openalex" not in registry.tool_names
        assert "get_citing_papers" not in registry.tool_names
