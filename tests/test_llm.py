from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from patronus.llm import (
    LLMResponse,
    ToolCall,
    build_assistant_message_from_response,
    build_tool_result_message,
    complete,
    complete_with_tools,
    _convert_messages_to_openai,
    _convert_tools_to_openai,
)


@pytest.fixture(autouse=True)
def _reset_clients() -> None:
    import patronus.llm as mod
    mod._anthropic_client = None
    mod._google_client = None
    mod._openai_client = None


class TestComplete:
    def test_raises_on_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider: fakeprovider"):
            complete("fakeprovider/model", user_message="hi")

    def test_raises_on_missing_slash(self) -> None:
        with pytest.raises(ValueError):
            complete("noslash", user_message="hi")


class TestAnthropicProvider:
    @patch("patronus.llm._complete_anthropic")
    def test_routes_to_anthropic(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = "response"
        result = complete("anthropic/claude-haiku-4-5-20251001", user_message="hello")
        assert result == "response"
        mock_fn.assert_called_once_with(
            "claude-haiku-4-5-20251001",
            system="",
            user_message="hello",
            max_tokens=4096,
        )

    @patch("anthropic.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_creates_client_and_calls_api(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="hello back")]
        )

        result = complete(
            "anthropic/claude-haiku-4-5-20251001",
            system="Be helpful.",
            user_message="Hi",
            max_tokens=100,
        )

        assert result == "hello back"
        mock_cls.assert_called_once_with(api_key="sk-ant-test")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["system"] == "Be helpful."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    @patch("anthropic.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_omits_system_when_empty(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")]
        )

        complete("anthropic/claude-haiku-4-5-20251001", user_message="Hi")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs


class TestGoogleProvider:
    @patch("patronus.llm._complete_google")
    def test_routes_to_google(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = "gemini response"
        result = complete("google/gemini-2.5-flash-lite", user_message="hello")
        assert result == "gemini response"
        mock_fn.assert_called_once_with(
            "gemini-2.5-flash-lite",
            system="",
            user_message="hello",
            max_tokens=4096,
        )

    @patch("google.genai.Client")
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "gk-test"})
    def test_creates_client_and_calls_api(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = SimpleNamespace(text="gemini says hi")

        result = complete(
            "google/gemini-2.5-flash-lite",
            system="Be concise.",
            user_message="Hello",
            max_tokens=200,
        )

        assert result == "gemini says hi"
        mock_cls.assert_called_once_with(api_key="gk-test")
        call_args = mock_client.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.5-flash-lite"
        assert call_args[1]["contents"] == "Hello"


class TestOpenAIProvider:
    @patch("patronus.llm._complete_openai")
    def test_routes_to_openai(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = "openai response"
        result = complete("openai/gpt-4o-mini", user_message="hello")
        assert result == "openai response"

    @patch("openai.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_creates_client_and_calls_api(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="gpt says hi"))]
        )

        result = complete(
            "openai/gpt-4o-mini",
            system="Be helpful.",
            user_message="Hi",
            max_tokens=100,
        )

        assert result == "gpt says hi"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_tokens"] == 100
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Hi"}

    @patch("openai.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_omits_system_when_empty(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        complete("openai/gpt-4o-mini", user_message="Hi")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"


class TestCompleteWithTools:
    @patch("patronus.llm._complete_with_tools_anthropic")
    def test_routes_to_anthropic(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = LLMResponse(text="hi", stop_reason="end_turn")
        result = complete_with_tools(
            "anthropic/claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )
        assert result.text == "hi"
        mock_fn.assert_called_once()

    @patch("patronus.llm._complete_with_tools_openai")
    def test_routes_to_openai(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = LLMResponse(text="hi", stop_reason="end_turn")
        result = complete_with_tools(
            "openai/gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )
        assert result.text == "hi"
        mock_fn.assert_called_once()

    def test_raises_on_unsupported_provider(self) -> None:
        with pytest.raises(ValueError, match="does not support tool use"):
            complete_with_tools(
                "google/gemini-pro",
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
            )

    @patch("anthropic.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_anthropic_tool_use_response(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me search"),
                SimpleNamespace(type="tool_use", id="call_1", name="search_recent", input={"days": 3}),
            ],
            stop_reason="tool_use",
        )

        result = complete_with_tools(
            "anthropic/claude-sonnet-4-20250514",
            system="Be an editor.",
            messages=[{"role": "user", "content": "Find items"}],
            tools=[{"name": "search_recent", "description": "Search", "input_schema": {}}],
            max_tokens=1000,
        )

        assert result.text == "Let me search"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_recent"
        assert result.tool_calls[0].input == {"days": 3}
        assert result.stop_reason == "tool_use"

    @patch("anthropic.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_anthropic_end_turn(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Done")],
            stop_reason="end_turn",
        )

        result = complete_with_tools(
            "anthropic/claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

        assert result.stop_reason == "end_turn"
        assert result.tool_calls == []

    @patch("openai.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_openai_tool_use_response(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="Searching...",
                    tool_calls=[SimpleNamespace(
                        id="call_abc",
                        function=SimpleNamespace(
                            name="search_similar",
                            arguments='{"query": "ML papers"}',
                        ),
                    )],
                ),
                finish_reason="tool_calls",
            )]
        )

        result = complete_with_tools(
            "openai/gpt-4o",
            messages=[{"role": "user", "content": "Find items"}],
            tools=[{"name": "search_similar", "description": "Search", "input_schema": {}}],
        )

        assert result.text == "Searching..."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_similar"
        assert result.tool_calls[0].input == {"query": "ML papers"}
        assert result.stop_reason == "tool_use"

    @patch("openai.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_openai_no_tool_calls(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Done", tool_calls=None),
                finish_reason="stop",
            )]
        )

        result = complete_with_tools(
            "openai/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )

        assert result.text == "Done"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"


class TestBuildToolResultMessage:
    def test_single_tool_result(self) -> None:
        tc = ToolCall(id="call_1", name="search_recent", input={})
        result = build_tool_result_message([tc], {"call_1": "Found 5 items."})

        assert result["role"] == "user"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "call_1"
        assert result["content"][0]["content"] == "Found 5 items."

    def test_multiple_tool_results(self) -> None:
        tc1 = ToolCall(id="c1", name="search_recent", input={})
        tc2 = ToolCall(id="c2", name="search_similar", input={})
        result = build_tool_result_message(
            [tc1, tc2],
            {"c1": "Result 1", "c2": "Result 2"},
        )

        assert len(result["content"]) == 2
        assert result["content"][0]["tool_use_id"] == "c1"
        assert result["content"][1]["tool_use_id"] == "c2"

    def test_missing_result_returns_empty(self) -> None:
        tc = ToolCall(id="c1", name="search", input={})
        result = build_tool_result_message([tc], {})
        assert result["content"][0]["content"] == ""


class TestBuildAssistantMessage:
    def test_text_and_tool_calls(self) -> None:
        response = LLMResponse(
            text="Thinking...",
            tool_calls=[ToolCall(id="c1", name="search_recent", input={"days": 3})],
        )
        msg = build_assistant_message_from_response(response)

        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Thinking..."
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["id"] == "c1"
        assert msg["content"][1]["name"] == "search_recent"

    def test_tool_calls_only(self) -> None:
        response = LLMResponse(
            text=None,
            tool_calls=[ToolCall(id="c1", name="search", input={})],
        )
        msg = build_assistant_message_from_response(response)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_use"

    def test_text_only(self) -> None:
        response = LLMResponse(text="Just text", tool_calls=[])
        msg = build_assistant_message_from_response(response)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"

    def test_empty_response(self) -> None:
        response = LLMResponse(text=None, tool_calls=[])
        msg = build_assistant_message_from_response(response)
        assert msg["content"] == []


class TestConvertMessagesToOpenai:
    def test_simple_user_message(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = _convert_messages_to_openai(messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_system_prepended(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = _convert_messages_to_openai(messages, system="Be helpful.")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "Be helpful."}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_assistant_with_tool_use(self) -> None:
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search"},
                {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "test"}},
            ],
        }]
        result = _convert_messages_to_openai(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me search"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "c1"
        assert result[0]["tool_calls"][0]["type"] == "function"

    def test_user_tool_results(self) -> None:
        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "Result data"},
            ],
        }]
        result = _convert_messages_to_openai(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "c1"
        assert result[0]["content"] == "Result data"

    def test_assistant_tool_use_no_text(self) -> None:
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "search", "input": {}},
            ],
        }]
        result = _convert_messages_to_openai(messages)
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1


class TestConvertToolsToOpenai:
    def test_converts_anthropic_format(self) -> None:
        tools = [{
            "name": "search_recent",
            "description": "Search recent items",
            "input_schema": {
                "type": "object",
                "properties": {"days": {"type": "integer"}},
            },
        }]
        result = _convert_tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search_recent"
        assert result[0]["function"]["description"] == "Search recent items"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_empty_tools(self) -> None:
        assert _convert_tools_to_openai([]) == []

    def test_missing_optional_fields(self) -> None:
        tools = [{"name": "foo"}]
        result = _convert_tools_to_openai(tools)
        assert result[0]["function"]["description"] == ""
        assert result[0]["function"]["parameters"] == {}
