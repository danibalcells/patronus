from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from patronus.agent import _parse_submit_digest, plan_and_assemble, SUBMIT_DIGEST_TOOL
from patronus.config import AgentConfig, Config, DigestConfig, EmbeddingConfig, PollingConfig, SummarizationConfig, TelegramConfig
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
        summarization=SummarizationConfig(),
        telegram=TelegramConfig(),
        topics={},
        agent=agent or AgentConfig(max_iterations=5, max_tokens=2000),
    )


def _make_context(prose: str = "Reader is interested in ML and philosophy.") -> Context:
    return Context(prose=prose, vectors={})


def _make_registry() -> ToolRegistry:
    return ToolRegistry()


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
                "type": "paper_roundup",
                "title": "Papers",
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
                "type": "serendipity",
                "items": [{"item_id": "x", "title": "t", "url": "u", "summary": "s"}],
            }]
        }
        digest = _parse_submit_digest(input_data)
        assert digest.sections[0].title == "serendipity"


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


class TestPlanAndAssemble:
    def test_requires_agent_config(self) -> None:
        config = _make_config()
        config.agent = None
        with pytest.raises(ValueError, match="AgentConfig is required"):
            plan_and_assemble(config, _make_context(), _make_registry())

    @patch("patronus.agent.complete_with_tools")
    def test_submit_on_first_iteration(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = LLMResponse(
            text="Here's the digest",
            tool_calls=[ToolCall(
                id="call_submit",
                name="submit_digest",
                input={
                    "sections": [{
                        "type": "long_form_pick",
                        "title": "Today's Pick",
                        "items": [{
                            "item_id": "item-1",
                            "title": "Great Paper",
                            "url": "https://example.com",
                            "summary": "Excellent work.",
                        }],
                    }]
                },
            )],
            stop_reason="tool_use",
        )

        config = _make_config()
        digest = plan_and_assemble(config, _make_context(), _make_registry())

        assert digest.mode == "agent"
        assert digest.item_count == 1
        assert digest.sections[0].type == SectionType.LONG_FORM_PICK
        mock_complete.assert_called_once()

    @patch("patronus.agent.complete_with_tools")
    def test_multiple_iterations_with_tool_calls(self, mock_complete: MagicMock) -> None:
        registry = ToolRegistry()

        class FakeTool:
            @property
            def name(self) -> str:
                return "search_recent"
            @property
            def description(self) -> str:
                return "Search"
            @property
            def input_schema(self) -> dict:
                return {"type": "object", "properties": {}}
            def to_definition(self) -> dict:
                return {"name": self.name, "description": self.description, "input_schema": self.input_schema}
            def execute(self, **params: object) -> ToolResult:
                return ToolResult(items=[{"title": "Found Item", "id": "f1"}], message="Found 1 item.")

        registry._tools["search_recent"] = FakeTool()

        mock_complete.side_effect = [
            LLMResponse(
                text="Let me search",
                tool_calls=[ToolCall(id="c1", name="search_recent", input={"days": 3})],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Done",
                tool_calls=[ToolCall(
                    id="c_submit",
                    name="submit_digest",
                    input={"sections": [{
                        "type": "headlines",
                        "title": "Headlines",
                        "items": [{"item_id": "f1", "title": "Found Item", "url": "https://x.com", "summary": "News."}],
                    }]},
                )],
                stop_reason="tool_use",
            ),
        ]

        config = _make_config()
        digest = plan_and_assemble(config, _make_context(), registry)

        assert digest.item_count == 1
        assert digest.sections[0].type == SectionType.HEADLINES
        assert mock_complete.call_count == 2

    @patch("patronus.agent.complete_with_tools")
    def test_max_iterations_returns_empty(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = LLMResponse(
            text="Still searching...",
            tool_calls=[ToolCall(id="c1", name="search_recent", input={})],
            stop_reason="tool_use",
        )

        config = _make_config(agent=AgentConfig(max_iterations=2, max_tokens=2000))
        registry = ToolRegistry()

        class DummyTool:
            @property
            def name(self) -> str:
                return "search_recent"
            @property
            def description(self) -> str:
                return ""
            @property
            def input_schema(self) -> dict:
                return {}
            def to_definition(self) -> dict:
                return {"name": self.name, "description": self.description, "input_schema": self.input_schema}
            def execute(self, **params: object) -> ToolResult:
                return ToolResult(message="Results")

        registry._tools["search_recent"] = DummyTool()

        digest = plan_and_assemble(config, _make_context(), registry)

        assert digest.item_count == 0
        assert digest.mode == "agent"
        assert mock_complete.call_count == 2

    @patch("patronus.agent.complete_with_tools")
    def test_agent_ends_without_submit(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = LLMResponse(
            text="I don't have enough to make a digest.",
            tool_calls=[],
            stop_reason="end_turn",
        )

        config = _make_config()
        digest = plan_and_assemble(config, _make_context(), _make_registry())

        assert digest.item_count == 0
        assert digest.mode == "agent"

    @patch("patronus.agent.complete_with_tools")
    def test_invalid_submit_retries(self, mock_complete: MagicMock) -> None:
        mock_complete.side_effect = [
            LLMResponse(
                text="Submitting",
                tool_calls=[ToolCall(
                    id="c1",
                    name="submit_digest",
                    input={"sections": [{"type": "invalid_type", "title": "Bad", "items": []}]},
                )],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="Fixed",
                tool_calls=[ToolCall(
                    id="c2",
                    name="submit_digest",
                    input={"sections": [{
                        "type": "headlines",
                        "title": "Headlines",
                        "items": [{"item_id": "1", "title": "T", "url": "U", "summary": "S"}],
                    }]},
                )],
                stop_reason="tool_use",
            ),
        ]

        config = _make_config()
        digest = plan_and_assemble(config, _make_context(), _make_registry())

        assert digest.item_count == 1
        assert mock_complete.call_count == 2

    @patch("patronus.agent.complete_with_tools")
    def test_system_prompt_and_context_passed(self, mock_complete: MagicMock) -> None:
        mock_complete.return_value = LLMResponse(
            text=None,
            tool_calls=[ToolCall(
                id="c1",
                name="submit_digest",
                input={"sections": []},
            )],
            stop_reason="tool_use",
        )

        config = _make_config()
        context = _make_context(prose="Reader studies consciousness.")
        plan_and_assemble(config, context, _make_registry())

        call_kwargs = mock_complete.call_args
        assert "editorial" in call_kwargs.kwargs["system"].lower()
        messages = call_kwargs.kwargs["messages"]
        assert "Reader studies consciousness." in messages[0]["content"]
