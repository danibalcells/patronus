from __future__ import annotations

from datetime import datetime, timezone
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


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_config(**overrides: object) -> Config:
    return Config(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        summarization=SummarizationConfig(),
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


class TestSearchArxiv:
    def test_returns_not_implemented(self) -> None:
        tool = SearchArxiv()
        result = tool.execute(query="transformer attention")
        assert "not yet implemented" in result.message.lower()
        assert len(result.items) == 0

    def test_tool_metadata(self) -> None:
        tool = SearchArxiv()
        assert tool.name == "search_arxiv"
        defn = tool.to_definition()
        assert "query" in defn["input_schema"]["properties"]


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
