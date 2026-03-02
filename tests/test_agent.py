from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from patronus.agent import (
    SUBMIT_DIGEST_TOOL,
    NewsFilterResult,
    NewsItem,
    _parse_submit_digest,
    build_inventory,
    filter_news,
    identify_angles,
    plan_and_assemble,
    pull_threads,
    scout_research,
    summarize_chatter,
)
from patronus.config import AgentConfig, Config, DigestConfig, EmbeddingConfig, PollingConfig, TelegramConfig
from patronus.context import Context
from patronus.digest import Digest, SectionType
from patronus.llm import LLMResponse, ToolCall
from patronus.tools import ToolRegistry
from patronus.tools.base import ToolResult


def _make_config(agent: AgentConfig | None = None) -> Config:
    return Config(
        digest=DigestConfig(mode="agent"),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        telegram=TelegramConfig(),
        topics={},
        agent=agent or AgentConfig(max_iterations=3, max_tokens=2000),
    )


def _make_context(prose: str = "Reader is interested in ML and philosophy.") -> Context:
    return Context(prose=prose, vectors={})


def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _make_submit_response(section_type: str = "headlines") -> LLMResponse:
    return LLMResponse(
        text="Done",
        tool_calls=[ToolCall(
            id="c_submit",
            name="submit_digest",
            input={"sections": [{
                "type": section_type,
                "title": section_type.replace("_", " ").title(),
                "items": [{"item_id": "item-1", "title": "T", "url": "https://x.com", "summary": "S."}],
            }]},
        )],
        stop_reason="tool_use",
    )


def _make_search_response(tool_name: str = "search_recent") -> LLMResponse:
    return LLMResponse(
        text="Let me search",
        tool_calls=[ToolCall(id="c1", name=tool_name, input={"days": 3})],
        stop_reason="tool_use",
    )


def _make_text_response(text: str = "Here is the output.") -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn")


class TestParseSubmitDigest:
    def test_valid_digest(self) -> None:
        input_data = {
            "sections": [
                {
                    "type": "long_form_pick",
                    "title": "Today's Pick",
                    "items": [
                        {
                            "item_id": "item-1",
                            "title": "Great ML Paper",
                            "url": "https://example.com/paper",
                            "source": "Arxiv",
                            "author": "Alice",
                            "summary": "A groundbreaking paper on attention mechanisms.",
                        }
                    ],
                },
                {
                    "type": "headlines",
                    "title": "Headlines",
                    "items": [
                        {
                            "item_id": "item-2",
                            "title": "OpenAI releases GPT-5",
                            "url": "https://news.com/gpt5",
                            "summary": "The latest model from OpenAI.",
                        },
                        {
                            "item_id": "item-3",
                            "title": "EU AI Act update",
                            "url": "https://news.com/eu",
                            "summary": "New regulations coming into effect.",
                        },
                    ],
                },
            ]
        }

        digest = _parse_submit_digest(input_data)

        assert digest.mode == "agent"
        assert len(digest.sections) == 2
        assert digest.sections[0].type == SectionType.LONG_FORM_PICK
        assert digest.sections[0].title == "Today's Pick"
        assert len(digest.sections[0].items) == 1
        assert digest.sections[0].items[0].title == "Great ML Paper"
        assert digest.sections[0].items[0].author == "Alice"

        assert digest.sections[1].type == SectionType.HEADLINES
        assert len(digest.sections[1].items) == 2
        assert digest.item_count == 3
        assert digest.generated_at != ""

    def test_empty_sections(self) -> None:
        digest = _parse_submit_digest({"sections": []})
        assert digest.item_count == 0
        assert digest.sections == []

    def test_missing_optional_item_fields(self) -> None:
        input_data = {
            "sections": [{
                "type": "research_roundup",
                "title": "Research",
                "items": [{"item_id": "x", "title": "Paper", "url": "https://x.com", "summary": "Good"}],
            }]
        }
        digest = _parse_submit_digest(input_data)
        item = digest.sections[0].items[0]
        assert item.source == ""
        assert item.author == ""

    def test_all_section_types(self) -> None:
        sections = []
        for st in SectionType:
            sections.append({
                "type": st.value,
                "title": st.value,
                "items": [{"item_id": "id", "title": "t", "url": "u", "summary": "s"}],
            })
        digest = _parse_submit_digest({"sections": sections})
        assert len(digest.sections) == len(SectionType)

    def test_invalid_section_type_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_submit_digest({
                "sections": [{"type": "nonexistent_type", "title": "Bad", "items": []}]
            })

    def test_title_falls_back_to_section_type(self) -> None:
        input_data = {
            "sections": [{
                "type": "threads",
                "items": [{"item_id": "x", "title": "t", "url": "u", "summary": "s"}],
            }]
        }
        digest = _parse_submit_digest(input_data)
        assert digest.sections[0].title == "threads"

    def test_new_section_types(self) -> None:
        for section_type in ("whats_new", "research_roundup", "threads"):
            digest = _parse_submit_digest({
                "sections": [{
                    "type": section_type,
                    "title": section_type,
                    "items": [{"item_id": "x", "title": "t", "url": "u", "summary": "s"}],
                }]
            })
            assert len(digest.sections) == 1


class TestSubmitDigestToolSchema:
    def test_has_required_fields(self) -> None:
        assert SUBMIT_DIGEST_TOOL["name"] == "submit_digest"
        schema = SUBMIT_DIGEST_TOOL["input_schema"]
        assert "sections" in schema["properties"]
        assert "sections" in schema["required"]

    def test_section_schema(self) -> None:
        section_schema = SUBMIT_DIGEST_TOOL["input_schema"]["properties"]["sections"]["items"]
        assert "type" in section_schema["properties"]
        assert "title" in section_schema["properties"]
        assert "items" in section_schema["properties"]

    def test_item_schema_required_fields(self) -> None:
        item_schema = (
            SUBMIT_DIGEST_TOOL["input_schema"]["properties"]["sections"]["items"]
            ["properties"]["items"]["items"]
        )
        assert set(item_schema["required"]) == {"item_id", "title", "url", "summary"}

    def test_new_section_types_in_enum(self) -> None:
        type_enum = (
            SUBMIT_DIGEST_TOOL["input_schema"]["properties"]["sections"]["items"]
            ["properties"]["type"]["enum"]
        )
        assert "whats_new" in type_enum
        assert "research_roundup" in type_enum
        assert "threads" in type_enum


class TestBuildInventory:
    def test_no_items_returns_message(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        config = _make_config()
        result, tweet_result, mapping = build_inventory(config, db)
        assert "No new items" in result
        assert mapping == {}

    def test_no_db_returns_three_tuple(self) -> None:
        config = _make_config()
        result, tweet_result, mapping = build_inventory(config, None)
        assert "DB not provided" in result
        assert "DB not provided" in tweet_result
        assert mapping == {}

    def test_items_appear_in_inventory(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        db.add_item(
            url="https://example.com/paper1",
            source_type="rss",
            title="A Great Paper",
            source="Test Feed",
            item_type="paper",
        )
        config = _make_config()
        result, _, mapping = build_inventory(config, db)
        assert "A Great Paper" in result
        assert "https://example.com/paper1" in result

    def test_short_ids_in_inventory(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        db.add_item(url="https://a.com/1", source_type="rss", source="Feed A", title="Item 1")
        db.add_item(url="https://b.com/2", source_type="rss", source="Feed A", title="Item 2")
        config = _make_config()
        result, _, mapping = build_inventory(config, db)
        assert "ID: 1" in result
        assert "ID: 2" in result
        assert len(mapping) == 2
        assert all(len(v) == 32 for v in mapping.values())  # real IDs are 32-char hex

    def test_mapping_covers_all_items(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        real_ids = [
            db.add_item(url=f"https://x.com/{i}", source_type="rss", title=f"Item {i}")
            for i in range(5)
        ]
        config = _make_config()
        _, _, mapping = build_inventory(config, db)
        assert len(mapping) == 5
        assert set(mapping.values()) == set(real_ids)

    def test_inventory_includes_source_grouping(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        db.add_item(url="https://a.com/1", source_type="rss", source="Feed A", title="Item 1")
        db.add_item(url="https://b.com/1", source_type="rss", source="Feed B", title="Item 2")
        config = _make_config()
        result, _, _ = build_inventory(config, db)
        assert "Feed A" in result
        assert "Feed B" in result

    def test_previously_featured_flagged(self, tmp_path: object) -> None:
        from patronus.db import Database
        from datetime import datetime, timezone
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        item_id = db.add_item(
            url="https://example.com/old",
            source_type="rss",
            title="Old Item",
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.save_digest(
            generated_at=now,
            item_count=1,
            formatted_text="",
            items=[{"item_id": item_id, "summary": "", "score": 0.0, "matched_topic": ""}],
        )
        config = _make_config()
        result, _, _ = build_inventory(config, db)
        assert "PREVIOUSLY_FEATURED" in result

    def test_tweet_inventory_contains_only_tweets(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        db.add_item(url="https://example.com/article", source_type="rss", title="Article", item_type="article")
        db.add_item(url="https://twitter.com/status/1", source_type="rss", title="Tweet content", item_type="tweet")
        config = _make_config()
        _, tweet_inventory, _ = build_inventory(config, db)
        assert "Tweet content" in tweet_inventory
        assert "Article" not in tweet_inventory

    def test_no_tweets_returns_placeholder(self, tmp_path: object) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        db.add_item(url="https://example.com/article", source_type="rss", title="Article", item_type="article")
        config = _make_config()
        _, tweet_inventory, _ = build_inventory(config, db)
        assert "No tweets" in tweet_inventory


class TestIdentifyAngles:
    @patch("patronus.agent._steps.complete")
    def test_returns_llm_output(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "Angle 1: The SAE paper connects to reader's current work."
        config = _make_config()
        result = identify_angles(config, "inventory text", "reader context text")
        assert "Angle 1" in result
        mock_complete.assert_called_once()

    @patch("patronus.agent._steps.complete")
    def test_uses_angles_model_if_set(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "angles"
        config = _make_config(agent=AgentConfig(
            model="anthropic/claude-sonnet-4-20250514",
            angles_model="google/gemini-2.5-flash-lite",
        ))
        identify_angles(config, "inv", "ctx")
        call_model = mock_complete.call_args.args[0]
        assert call_model == "google/gemini-2.5-flash-lite"

    @patch("patronus.agent._steps.complete")
    def test_falls_back_to_main_model(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "angles"
        config = _make_config(agent=AgentConfig(model="anthropic/claude-sonnet-4-20250514"))
        identify_angles(config, "inv", "ctx")
        call_model = mock_complete.call_args.args[0]
        assert call_model == "anthropic/claude-sonnet-4-20250514"

    @patch("patronus.agent._steps.complete")
    def test_inventory_in_prompt(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "angles"
        config = _make_config()
        identify_angles(config, "INVENTORY_SENTINEL", "CONTEXT_SENTINEL")
        user_msg = mock_complete.call_args.kwargs["user_message"]
        assert "INVENTORY_SENTINEL" in user_msg
        assert "CONTEXT_SENTINEL" in user_msg


def _make_news_result(items: list[dict] | None = None) -> NewsFilterResult:
    if items is None:
        items = [{"item_id": "1"}]
    return NewsFilterResult(items=[NewsItem(**i) for i in items])


def _make_items_by_short_id(entries: list[tuple[str, str, str, str]] | None = None) -> dict:
    from patronus.db import Item
    if entries is None:
        entries = [("1", "Big News", "https://x.com", "Reuters")]
    result = {}
    for short_id, title, url, source in entries:
        item = Item()
        item.id = f"real-{short_id}"
        item.title = title
        item.url = url
        item.source = source
        item.text = f"Content of {title}."
        item.item_type = "article"
        result[short_id] = item
    return result


class TestFilterNews:
    @patch("patronus.agent._steps.complete_structured")
    def test_returns_formatted_output(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result()
        config = _make_config()
        items = _make_items_by_short_id()
        result = filter_news(config, "inventory", "context", "angles", items)
        assert "Big News" in result
        assert "https://x.com" in result
        assert "Reuters" in result

    @patch("patronus.agent._steps.complete_structured")
    def test_empty_selection_returns_placeholder(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result(items=[])
        config = _make_config()
        result = filter_news(config, "inventory", "context", "angles", {})
        assert "(No news items selected.)" in result

    @patch("patronus.agent._steps.complete_structured")
    def test_uses_news_model_if_set(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result()
        config = _make_config(agent=AgentConfig(
            model="anthropic/claude-sonnet-4-20250514",
            news_model="google/gemini-2.5-flash-lite",
        ))
        filter_news(config, "inv", "ctx", "angles", _make_items_by_short_id())
        assert mock_cs.call_args.args[0] == "google/gemini-2.5-flash-lite"

    @patch("patronus.agent._steps.complete_structured")
    def test_angles_in_prompt(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result()
        config = _make_config()
        filter_news(config, "inv", "ctx", "ANGLES_SENTINEL", _make_items_by_short_id())
        user_msg = mock_cs.call_args.kwargs["user_message"]
        assert "ANGLES_SENTINEL" in user_msg

    @patch("patronus.agent._steps.complete_structured")
    def test_cross_ref_included_in_output(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result(items=[{"item_id": "1", "cross_ref": "research"}])
        config = _make_config()
        result = filter_news(config, "inv", "ctx", "angles", _make_items_by_short_id())
        assert "CROSS_REF: research" in result

    @patch("patronus.agent._steps.complete_structured")
    def test_output_uses_raw_item_content_not_generated_summary(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result(items=[{"item_id": "1"}])
        config = _make_config()
        items = _make_items_by_short_id([("1", "Paper About Attention", "https://arxiv.org/abs/1", "Arxiv")])
        result = filter_news(config, "inv", "ctx", "angles", items)
        assert "Paper About Attention" in result
        assert "SNIPPET" in result

    @patch("patronus.agent._steps.complete_structured")
    def test_unknown_item_id_is_skipped(self, mock_cs: MagicMock) -> None:
        mock_cs.return_value = _make_news_result(items=[{"item_id": "99"}])
        config = _make_config()
        result = filter_news(config, "inv", "ctx", "angles", {})
        assert "(No news items selected.)" in result or "Selected 0" in result or "99" not in result


class TestSummarizeChatter:
    @patch("patronus.agent._steps.complete")
    def test_returns_llm_output(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "TOPIC: AI regulation\nSUMMARY: People are discussing..."
        config = _make_config()
        result = summarize_chatter(config, "tweet inventory", "reader context")
        assert "TOPIC" in result
        mock_complete.assert_called_once()

    @patch("patronus.agent._steps.complete")
    def test_uses_chatter_model_if_set(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "chatter"
        config = _make_config(agent=AgentConfig(
            model="anthropic/claude-sonnet-4-20250514",
            chatter_model="google/gemini-2.5-flash-lite",
        ))
        summarize_chatter(config, "tweets", "ctx")
        call_model = mock_complete.call_args.args[0]
        assert call_model == "google/gemini-2.5-flash-lite"

    @patch("patronus.agent._steps.complete")
    def test_falls_back_to_main_model(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "chatter"
        config = _make_config(agent=AgentConfig(model="anthropic/claude-sonnet-4-20250514"))
        summarize_chatter(config, "tweets", "ctx")
        call_model = mock_complete.call_args.args[0]
        assert call_model == "anthropic/claude-sonnet-4-20250514"

    @patch("patronus.agent._steps.complete")
    def test_tweet_inventory_in_prompt(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "chatter"
        config = _make_config()
        summarize_chatter(config, "TWEET_SENTINEL", "CONTEXT_SENTINEL")
        user_msg = mock_complete.call_args.kwargs["user_message"]
        assert "TWEET_SENTINEL" in user_msg
        assert "CONTEXT_SENTINEL" in user_msg


class TestScoutResearch:
    @patch("patronus.agent._steps.complete_with_tools")
    def test_returns_text_output(self, mock_cwt: MagicMock) -> None:
        mock_cwt.return_value = _make_text_response("Research output: 3 papers found.")
        config = _make_config()
        result = scout_research(config, "context", "angles", _make_registry())
        assert "Research output" in result

    @patch("patronus.agent._steps.complete_with_tools")
    def test_executes_tool_calls(self, mock_cwt: MagicMock) -> None:
        registry = _make_registry()

        class FakeTool:
            name = "search_similar"
            description = "Search"
            input_schema: dict = {}

            def to_definition(self) -> dict:
                return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

            def execute(self, **params: object) -> ToolResult:
                return ToolResult(message="3 papers found.")

        registry._tools["search_similar"] = FakeTool()

        mock_cwt.side_effect = [
            _make_search_response("search_similar"),
            _make_text_response("Here are the results."),
        ]
        config = _make_config()
        result = scout_research(config, "context", "angles", registry)
        assert mock_cwt.call_count == 2
        assert "Here are the results." in result

    @patch("patronus.agent._steps.complete_with_tools")
    def test_caps_at_max_iterations(self, mock_cwt: MagicMock) -> None:
        registry = _make_registry()

        class FakeTool:
            name = "search_similar"
            description = ""
            input_schema: dict = {}

            def to_definition(self) -> dict:
                return {"name": self.name, "description": "", "input_schema": {}}

            def execute(self, **params: object) -> ToolResult:
                return ToolResult(message="results")

        registry._tools["search_similar"] = FakeTool()

        # Always returns tool calls → exhausts iterations, then triggers synthesis call
        mock_cwt.return_value = _make_search_response("search_similar")
        config = _make_config()
        scout_research(config, "ctx", "angles", registry)
        assert mock_cwt.call_count == 3  # _RESEARCH_MAX_ITERATIONS = 2 + 1 synthesis call

    @patch("patronus.agent._steps.complete_with_tools")
    def test_uses_research_model(self, mock_cwt: MagicMock) -> None:
        mock_cwt.return_value = _make_text_response("done")
        config = _make_config(agent=AgentConfig(
            model="anthropic/claude-sonnet-4-20250514",
            research_model="openai/gpt-4o",
        ))
        scout_research(config, "ctx", "angles", _make_registry())
        assert mock_cwt.call_args.args[0] == "openai/gpt-4o"

    @patch("patronus.agent._steps.complete_with_tools")
    def test_does_not_receive_inventory(self, mock_cwt: MagicMock) -> None:
        mock_cwt.return_value = _make_text_response("done")
        config = _make_config()
        scout_research(config, "CONTEXT_SENTINEL", "ANGLES_SENTINEL", _make_registry())
        user_msg = str(mock_cwt.call_args.kwargs["messages"])
        assert "CONTEXT_SENTINEL" in user_msg
        assert "ANGLES_SENTINEL" in user_msg

    @patch("patronus.agent._steps.complete_with_tools")
    def test_final_synthesis_call_when_iterations_exhausted(self, mock_cwt: MagicMock) -> None:
        """When all iterations are used up with tool calls, a final synthesis call is made."""
        registry = _make_registry()

        class FakeTool:
            name = "search_similar"
            description = ""
            input_schema: dict = {}

            def to_definition(self) -> dict:
                return {"name": self.name, "description": "", "input_schema": {}}

            def execute(self, **params: object) -> ToolResult:
                return ToolResult(message="results")

        registry._tools["search_similar"] = FakeTool()

        # Both iterations return tool calls — exhausts the cap. Then the synthesis call returns text.
        mock_cwt.side_effect = [
            _make_search_response("search_similar"),  # iter 1: tool call
            _make_search_response("search_similar"),  # iter 2: tool call (exhausts max)
            _make_text_response("Final curated paper list."),  # synthesis call
        ]
        config = _make_config()
        result = scout_research(config, "ctx", "angles", registry)
        # 2 iterations + 1 synthesis = 3 total calls
        assert mock_cwt.call_count == 3
        assert "Final curated paper list." in result


class TestPullThreads:
    @patch("patronus.agent._steps.complete_with_tools")
    def test_returns_text_output(self, mock_cwt: MagicMock) -> None:
        mock_cwt.return_value = _make_text_response("Thread: the consciousness paper connects to your Notion notes.")
        config = _make_config()
        result = pull_threads(config, "context", "angles", "news", "research", _make_registry())
        assert "consciousness paper" in result

    @patch("patronus.agent._steps.complete_with_tools")
    def test_receives_news_and_research_context(self, mock_cwt: MagicMock) -> None:
        mock_cwt.return_value = _make_text_response("threads")
        config = _make_config()
        pull_threads(config, "ctx", "angles", "NEWS_SENTINEL", "RESEARCH_SENTINEL", _make_registry())
        user_msg = str(mock_cwt.call_args.kwargs["messages"])
        assert "NEWS_SENTINEL" in user_msg
        assert "RESEARCH_SENTINEL" in user_msg

    @patch("patronus.agent._steps.complete_with_tools")
    def test_caps_at_max_iterations(self, mock_cwt: MagicMock) -> None:
        registry = _make_registry()

        class FakeTool:
            name = "search_similar"
            description = ""
            input_schema: dict = {}

            def to_definition(self) -> dict:
                return {"name": self.name, "description": "", "input_schema": {}}

            def execute(self, **params: object) -> ToolResult:
                return ToolResult(message="results")

        registry._tools["search_similar"] = FakeTool()
        # Always returns tool calls → exhausts iterations, then triggers synthesis call
        mock_cwt.return_value = _make_search_response("search_similar")
        config = _make_config()
        pull_threads(config, "ctx", "angles", "news", "research", registry)
        assert mock_cwt.call_count == 4  # _THREADS_MAX_ITERATIONS = 3 + 1 synthesis call


class TestPlanAndAssemble:
    def test_requires_agent_config(self) -> None:
        config = _make_config()
        config.agent = None
        with pytest.raises(ValueError, match="AgentConfig is required"):
            plan_and_assemble(config, _make_context(), _make_registry())

    @patch("patronus.agent.run.compose_digest")
    @patch("patronus.agent.run.pull_threads")
    @patch("patronus.agent.run.scout_research")
    @patch("patronus.agent.run.summarize_chatter")
    @patch("patronus.agent.run.filter_news")
    @patch("patronus.agent.run.identify_angles")
    @patch("patronus.agent.run.build_inventory")
    def test_full_pipeline_called_in_order(
        self,
        mock_inventory: MagicMock,
        mock_angles: MagicMock,
        mock_news: MagicMock,
        mock_chatter: MagicMock,
        mock_research: MagicMock,
        mock_threads: MagicMock,
        mock_compose: MagicMock,
    ) -> None:
        mock_inventory.return_value = ("inventory", "tweet_inventory", {})
        mock_angles.return_value = "angles"
        mock_news.return_value = "news"
        mock_chatter.return_value = "chatter"
        mock_research.return_value = "research"
        mock_threads.return_value = "threads"
        expected_digest = Digest(
            sections=[],
            generated_at="2025-01-01T00:00:00Z",
            mode="agent",
        )
        mock_compose.return_value = expected_digest

        config = _make_config()
        context = _make_context()
        result = plan_and_assemble(config, context, _make_registry())

        mock_inventory.assert_called_once()
        mock_angles.assert_called_once()
        mock_news.assert_called_once()
        mock_chatter.assert_called_once()
        mock_research.assert_called_once()
        mock_threads.assert_called_once()
        mock_compose.assert_called_once()
        assert result is expected_digest

    @patch("patronus.agent.run.compose_digest")
    @patch("patronus.agent.run.pull_threads")
    @patch("patronus.agent.run.scout_research")
    @patch("patronus.agent.run.summarize_chatter")
    @patch("patronus.agent.run.filter_news")
    @patch("patronus.agent.run.identify_angles")
    @patch("patronus.agent.run.build_inventory")
    def test_db_passed_to_inventory(
        self,
        mock_inventory: MagicMock,
        mock_angles: MagicMock,
        mock_news: MagicMock,
        mock_chatter: MagicMock,
        mock_research: MagicMock,
        mock_threads: MagicMock,
        mock_compose: MagicMock,
        tmp_path: object,
    ) -> None:
        from patronus.db import Database
        db = Database(str(tmp_path / "test.db"))  # type: ignore[arg-type]
        mock_inventory.return_value = ("inventory", "tweet_inventory", {})
        mock_angles.return_value = "angles"
        mock_news.return_value = "news"
        mock_chatter.return_value = "chatter"
        mock_research.return_value = "research"
        mock_threads.return_value = "threads"
        mock_compose.return_value = Digest(generated_at="2025-01-01T00:00:00Z", mode="agent")

        config = _make_config()
        plan_and_assemble(config, _make_context(), _make_registry(), db=db)

        call_args = mock_inventory.call_args
        assert call_args.args[1] is db or call_args.kwargs.get("db") is db

    @patch("patronus.agent.run.compose_digest")
    @patch("patronus.agent.run.pull_threads")
    @patch("patronus.agent.run.scout_research")
    @patch("patronus.agent.run.summarize_chatter")
    @patch("patronus.agent.run.filter_news")
    @patch("patronus.agent.run.identify_angles")
    @patch("patronus.agent.run.build_inventory")
    def test_context_prose_passed_to_steps(
        self,
        mock_inventory: MagicMock,
        mock_angles: MagicMock,
        mock_news: MagicMock,
        mock_chatter: MagicMock,
        mock_research: MagicMock,
        mock_threads: MagicMock,
        mock_compose: MagicMock,
    ) -> None:
        mock_inventory.return_value = ("inventory", "tweet_inventory", {})
        mock_angles.return_value = "angles"
        mock_news.return_value = "news"
        mock_chatter.return_value = "chatter"
        mock_research.return_value = "research"
        mock_threads.return_value = "threads"
        mock_compose.return_value = Digest(generated_at="2025-01-01T00:00:00Z", mode="agent")

        config = _make_config()
        context = _make_context(prose="PROSE_SENTINEL")
        plan_and_assemble(config, context, _make_registry())

        angles_call = mock_angles.call_args
        assert "PROSE_SENTINEL" in str(angles_call)

    @patch("patronus.agent.run.compose_digest")
    @patch("patronus.agent.run.pull_threads")
    @patch("patronus.agent.run.scout_research")
    @patch("patronus.agent.run.summarize_chatter")
    @patch("patronus.agent.run.filter_news")
    @patch("patronus.agent.run.identify_angles")
    @patch("patronus.agent.run.build_inventory")
    def test_chatter_output_passed_to_compose(
        self,
        mock_inventory: MagicMock,
        mock_angles: MagicMock,
        mock_news: MagicMock,
        mock_chatter: MagicMock,
        mock_research: MagicMock,
        mock_threads: MagicMock,
        mock_compose: MagicMock,
    ) -> None:
        mock_inventory.return_value = ("inventory", "tweet_inventory", {})
        mock_angles.return_value = "angles"
        mock_news.return_value = "news"
        mock_chatter.return_value = "CHATTER_SENTINEL"
        mock_research.return_value = "research"
        mock_threads.return_value = "threads"
        mock_compose.return_value = Digest(generated_at="2025-01-01T00:00:00Z", mode="agent")

        config = _make_config()
        plan_and_assemble(config, _make_context(), _make_registry())

        compose_call_kwargs = str(mock_compose.call_args)
        assert "CHATTER_SENTINEL" in compose_call_kwargs

    @patch("patronus.agent.run.compose_digest")
    @patch("patronus.agent.run.pull_threads")
    @patch("patronus.agent.run.scout_research")
    @patch("patronus.agent.run.summarize_chatter")
    @patch("patronus.agent.run.filter_news")
    @patch("patronus.agent.run.identify_angles")
    @patch("patronus.agent.run.build_inventory")
    def test_no_db_passes_none_to_inventory(
        self,
        mock_inventory: MagicMock,
        mock_angles: MagicMock,
        mock_news: MagicMock,
        mock_chatter: MagicMock,
        mock_research: MagicMock,
        mock_threads: MagicMock,
        mock_compose: MagicMock,
    ) -> None:
        mock_inventory.return_value = ("(No inventory available — DB not provided.)", "(No tweet inventory available — DB not provided.)", {})
        mock_angles.return_value = "angles"
        mock_news.return_value = "news"
        mock_chatter.return_value = "chatter"
        mock_research.return_value = "research"
        mock_threads.return_value = "threads"
        mock_compose.return_value = Digest(generated_at="2025-01-01T00:00:00Z", mode="agent")

        config = _make_config()
        plan_and_assemble(config, _make_context(), _make_registry(), db=None)

        # build_inventory is always called; it receives None as db
        mock_inventory.assert_called_once()
        call_args = mock_inventory.call_args
        db_arg = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("db")
        assert db_arg is None
