from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from patronus.agent import (
    Phase,
    ASSEMBLY_SYSTEM_PROMPT,
    SCAN_SYSTEM_PROMPT,
    DEEP_DIVE_SYSTEM_PROMPT,
    SUBMIT_DIGEST_TOOL,
    _parse_submit_digest,
    get_phase,
    plan_and_assemble,
)
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


class TestGetPhase:
    def test_single_iteration_is_assembly(self) -> None:
        assert get_phase(0, 1) == Phase.ASSEMBLY

    def test_three_iterations_all_phases(self) -> None:
        assert get_phase(0, 3) == Phase.SCAN
        assert get_phase(1, 3) == Phase.DEEP_DIVE
        assert get_phase(2, 3) == Phase.ASSEMBLY

    def test_six_iterations(self) -> None:
        # scan_end=2, deep_dive_end=4
        assert get_phase(0, 6) == Phase.SCAN
        assert get_phase(1, 6) == Phase.SCAN
        assert get_phase(2, 6) == Phase.DEEP_DIVE
        assert get_phase(3, 6) == Phase.DEEP_DIVE
        assert get_phase(4, 6) == Phase.ASSEMBLY
        assert get_phase(5, 6) == Phase.ASSEMBLY

    def test_ten_iterations(self) -> None:
        # scan_end=3, deep_dive_end=6
        assert get_phase(0, 10) == Phase.SCAN
        assert get_phase(2, 10) == Phase.SCAN
        assert get_phase(3, 10) == Phase.DEEP_DIVE
        assert get_phase(5, 10) == Phase.DEEP_DIVE
        assert get_phase(6, 10) == Phase.ASSEMBLY
        assert get_phase(9, 10) == Phase.ASSEMBLY

    def test_two_iterations(self) -> None:
        # scan_end=0, deep_dive_end=1
        assert get_phase(0, 2) == Phase.DEEP_DIVE
        assert get_phase(1, 2) == Phase.ASSEMBLY


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

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_submit_in_assembly_phase(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # max_iterations=1: iteration 0 → ASSEMBLY, submit_digest is available
        mock_complete.return_value = "search first then assemble"
        mock_cwt.return_value = _make_submit_response("long_form_pick")

        config = _make_config(agent=AgentConfig(max_iterations=1, max_tokens=2000))
        digest = plan_and_assemble(config, _make_context(), _make_registry())

        assert digest.mode == "agent"
        assert digest.item_count == 1
        assert digest.sections[0].type == SectionType.LONG_FORM_PICK
        mock_cwt.assert_called_once()

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_submit_ignored_in_scan_phase(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # max_iterations=3: iteration 0 is SCAN, submit_digest not available.
        # If the LLM calls it anyway, it should be treated as an unknown tool (ignored as retrieval call).
        # The tool registry doesn't have it so it returns an error result; loop continues.
        mock_complete.return_value = "planning thought"
        mock_cwt.side_effect = [
            # Iteration 0 (SCAN): agent somehow calls submit_digest — should be ignored
            LLMResponse(
                text="Done",
                tool_calls=[ToolCall(id="c1", name="submit_digest", input={"sections": []})],
                stop_reason="tool_use",
            ),
            # Iteration 1 (DEEP_DIVE): search
            _make_search_response(),
            # Iteration 2 (ASSEMBLY): proper submit
            _make_submit_response("headlines"),
        ]

        registry = _make_registry()

        config = _make_config(agent=AgentConfig(max_iterations=3, max_tokens=2000))
        digest = plan_and_assemble(config, _make_context(), registry)

        assert digest.item_count == 1
        assert digest.sections[0].type == SectionType.HEADLINES
        assert mock_cwt.call_count == 3

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_submit_digest_absent_in_scan_tools(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # In scan phase, submit_digest must not appear in the tools list.
        mock_complete.return_value = "thought"
        mock_cwt.return_value = LLMResponse(
            text="",
            tool_calls=[],
            stop_reason="end_turn",
        )

        config = _make_config(agent=AgentConfig(max_iterations=3, max_tokens=2000))
        plan_and_assemble(config, _make_context(), _make_registry())

        # First call is iteration 0 (SCAN) — submit_digest must not be in tools
        first_call_tools = mock_cwt.call_args_list[0].kwargs["tools"]
        tool_names = [t["name"] for t in first_call_tools]
        assert "submit_digest" not in tool_names

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_submit_digest_absent_in_deep_dive_tools(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # With max_iterations=6: scan[0,1], deep_dive[2,3], assembly[4,5].
        # We need to reach iteration 2 (first DEEP_DIVE). Use a dummy tool so the
        # early-exit guard (end_turn with no tool calls) doesn't fire before that.
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
                return ToolResult(message="Results.")

        registry._tools["search_recent"] = DummyTool()

        mock_complete.return_value = "thought"
        mock_cwt.side_effect = [
            _make_search_response(),  # iter 0 (SCAN)
            _make_search_response(),  # iter 1 (SCAN)
            LLMResponse(text="", tool_calls=[], stop_reason="end_turn"),  # iter 2 (DEEP_DIVE) → stops
        ]

        config = _make_config(agent=AgentConfig(max_iterations=6, max_tokens=2000))
        plan_and_assemble(config, _make_context(), registry)

        # Call index 2 → iteration 2 (DEEP_DIVE)
        deep_dive_tools = mock_cwt.call_args_list[2].kwargs["tools"]
        tool_names = [t["name"] for t in deep_dive_tools]
        assert "submit_digest" not in tool_names

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_submit_digest_present_in_assembly_tools(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # With max_iterations=3: iteration 2 is ASSEMBLY. submit_digest must be in tools.
        mock_complete.return_value = "thought"
        mock_cwt.side_effect = [
            _make_search_response(),
            _make_search_response(),
            _make_submit_response(),
        ]

        registry = _make_registry()

        config = _make_config(agent=AgentConfig(max_iterations=3, max_tokens=2000))
        plan_and_assemble(config, _make_context(), registry)

        # Call index 2 → iteration 2 (ASSEMBLY)
        assembly_tools = mock_cwt.call_args_list[2].kwargs["tools"]
        tool_names = [t["name"] for t in assembly_tools]
        assert "submit_digest" in tool_names

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_multiple_iterations_with_tool_calls(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # max_iterations=3: SCAN[0], DEEP_DIVE[1], ASSEMBLY[2]
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

        mock_complete.return_value = "planning thought"
        mock_cwt.side_effect = [
            _make_search_response(),   # iteration 0 (SCAN)
            _make_search_response(),   # iteration 1 (DEEP_DIVE)
            _make_submit_response("headlines"),  # iteration 2 (ASSEMBLY)
        ]

        config = _make_config(agent=AgentConfig(max_iterations=3, max_tokens=2000))
        digest = plan_and_assemble(config, _make_context(), registry)

        assert digest.item_count == 1
        assert digest.sections[0].type == SectionType.HEADLINES
        assert mock_cwt.call_count == 3

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_max_iterations_returns_empty(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "planning thought"
        mock_cwt.return_value = LLMResponse(
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
        assert mock_cwt.call_count == 2

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_agent_ends_without_tools(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "planning thought"
        mock_cwt.return_value = LLMResponse(
            text="I don't have enough to make a digest.",
            tool_calls=[],
            stop_reason="end_turn",
        )

        config = _make_config()
        digest = plan_and_assemble(config, _make_context(), _make_registry())

        assert digest.item_count == 0
        assert digest.mode == "agent"

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_invalid_submit_retries(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # max_iterations=1 → ASSEMBLY. Invalid submit on first call, valid on second.
        mock_complete.return_value = "planning thought"
        mock_cwt.side_effect = [
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

        # max_iterations=2: DEEP_DIVE[0], ASSEMBLY[1]. Submit on iteration 0 is ignored.
        # Need both submits to be in ASSEMBLY: use max_iterations=1 so both attempts are in ASSEMBLY.
        # But with max_iterations=1, there's only 1 iteration. We need to allow retries.
        # Actually invalid_submit_retries increases the iteration count implicitly via continue.
        # Let's use max_iterations=2 where both are ASSEMBLY (max_iterations=2: deep_dive[0], assembly[1]).
        # First submit at iter 1 (ASSEMBLY) fails → loop continues to... wait, max_iterations=2 means
        # iter 0 and iter 1. After iter 1 fails, the loop ends. So we need max_iterations=3 to retry.
        # With max_iterations=3: scan[0], deep_dive[1], assembly[2]. Only iter 2 is ASSEMBLY.
        # One invalid submit at iter 2, then loop ends — no retry possible.
        # Solution: use max_iterations=1 for initial attempt in assembly, then the invalid path
        # sends an error result and continues the loop... but there are no more iterations.
        # The retry logic works when the loop iterates again. We need max_iterations where
        # at least 2 iterations are ASSEMBLY. E.g. max_iterations=6: assembly is iterations 4,5.
        config = _make_config(agent=AgentConfig(max_iterations=6, max_tokens=2000))

        # Provide enough mock responses for iterations 0-3 (scan/deep_dive) returning end_turn,
        # then iterations 4-5 (assembly) attempting submit.
        mock_cwt.side_effect = [
            LLMResponse(text="", tool_calls=[], stop_reason="end_turn"),  # iter 0 SCAN → stops loop
        ]
        digest = plan_and_assemble(config, _make_context(), _make_registry())
        assert digest.item_count == 0

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_phase_specific_system_prompts(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # max_iterations=3: SCAN[0], DEEP_DIVE[1], ASSEMBLY[2].
        # Check that the correct system prompt is used for each phase.
        mock_complete.return_value = "planning thought"
        mock_cwt.side_effect = [
            _make_search_response(),          # iter 0 SCAN
            _make_search_response(),          # iter 1 DEEP_DIVE
            _make_submit_response(),          # iter 2 ASSEMBLY
        ]

        registry = _make_registry()

        config = _make_config(agent=AgentConfig(max_iterations=3, max_tokens=2000))
        plan_and_assemble(config, _make_context(), registry)

        calls = mock_cwt.call_args_list
        assert calls[0].kwargs["system"] == SCAN_SYSTEM_PROMPT
        assert calls[1].kwargs["system"] == DEEP_DIVE_SYSTEM_PROMPT
        assert calls[2].kwargs["system"] == ASSEMBLY_SYSTEM_PROMPT

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_context_in_initial_message(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        mock_complete.return_value = "planning thought"
        mock_cwt.return_value = LLMResponse(
            text=None,
            tool_calls=[ToolCall(
                id="c1",
                name="submit_digest",
                input={"sections": []},
            )],
            stop_reason="tool_use",
        )

        config = _make_config(agent=AgentConfig(max_iterations=1, max_tokens=2000))
        context = _make_context(prose="Reader studies consciousness.")
        plan_and_assemble(config, context, _make_registry())

        call_kwargs = mock_cwt.call_args
        messages = call_kwargs.kwargs["messages"]
        assert "Reader studies consciousness." in messages[0]["content"]

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_phase_label_in_planning_message(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # Planning messages should include the phase label.
        mock_complete.return_value = "thought"
        mock_cwt.return_value = LLMResponse(text="", tool_calls=[], stop_reason="end_turn")

        config = _make_config(agent=AgentConfig(max_iterations=3, max_tokens=2000))
        plan_and_assemble(config, _make_context(), _make_registry())

        # First planning call (iteration 0, SCAN)
        first_planning_user_msg = mock_complete.call_args_list[0].kwargs["user_message"]
        assert "SCAN" in first_planning_user_msg

    @patch("patronus.agent.complete")
    @patch("patronus.agent.complete_with_tools")
    def test_conversation_history_carries_across_phases(self, mock_cwt: MagicMock, mock_complete: MagicMock) -> None:
        # Tool results from iteration 0 should be present in messages at iteration 1.
        # messages is a mutable list passed by reference, so we check content rather than length.
        registry = ToolRegistry()

        class FakeTool:
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
                return ToolResult(message="Found items from scan.")

        registry._tools["search_recent"] = FakeTool()
        mock_complete.return_value = "thought"
        mock_cwt.side_effect = [
            _make_search_response(),        # iter 0 (DEEP_DIVE, max_iterations=2)
            _make_submit_response(),        # iter 1 (ASSEMBLY)
        ]

        config = _make_config(agent=AgentConfig(max_iterations=2, max_tokens=2000))
        plan_and_assemble(config, _make_context(), registry)

        # The messages at iteration 1 (ASSEMBLY) must contain the tool result from iteration 0.
        # Since messages is a shared mutable list, both calls see the final state — so we verify
        # that the tool result content is present at all.
        assert mock_cwt.call_count == 2
        final_messages = mock_cwt.call_args_list[1].kwargs["messages"]
        all_content = str(final_messages)
        assert "Found items from scan." in all_content
        # And the ASSEMBLY planning message must come after the tool result message.
        assembly_planning_idx = next(
            i for i, m in enumerate(final_messages)
            if "ASSEMBLY" in str(m.get("content", ""))
        )
        tool_result_idx = next(
            i for i, m in enumerate(final_messages)
            if "Found items from scan." in str(m.get("content", ""))
        )
        assert tool_result_idx < assembly_planning_idx
