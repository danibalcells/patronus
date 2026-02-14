from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from patronus.llm import complete


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
