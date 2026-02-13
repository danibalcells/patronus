from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from patronus.summarize import summarize_item


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    import patronus.summarize as mod
    mod._client = None


def _make_message_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


class TestSummarizeItem:
    @patch("patronus.summarize._get_client")
    def test_returns_summary_text(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.messages.create.return_value = _make_message_response(
            "This paper presents novel findings in mechanistic interpretability."
        )
        result = summarize_item(
            title="Circuits in Transformers",
            text="We investigate how transformers implement algorithms...",
            interest_description="Technical ML research.",
        )
        assert result == "This paper presents novel findings in mechanistic interpretability."

    @patch("patronus.summarize._get_client")
    def test_passes_model_kwarg(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.messages.create.return_value = _make_message_response("summary")

        summarize_item(
            title="Title",
            text="Text",
            interest_description="desc",
            model="claude-opus-4-20250514",
        )
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @patch("patronus.summarize._get_client")
    def test_includes_interest_description_in_prompt(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.messages.create.return_value = _make_message_response("summary")

        summarize_item(
            title="Article",
            text="Content here",
            interest_description="Philosophy of mind and consciousness.",
        )
        call_kwargs = client.messages.create.call_args[1]
        user_message = call_kwargs["messages"][0]["content"]
        assert "Philosophy of mind and consciousness." in user_message

    @patch("patronus.summarize._get_client")
    def test_truncates_long_text(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.messages.create.return_value = _make_message_response("summary")

        long_text = "x" * 20_000
        summarize_item(
            title="Long Article",
            text=long_text,
            interest_description="desc",
        )
        call_kwargs = client.messages.create.call_args[1]
        user_message = call_kwargs["messages"][0]["content"]
        assert len(user_message) < len(long_text)

    @patch("patronus.summarize._get_client")
    def test_system_prompt_is_set(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.messages.create.return_value = _make_message_response("summary")

        summarize_item(title="T", text="txt", interest_description="d")
        call_kwargs = client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert "2-3 sentence" in call_kwargs["system"]

    @patch("patronus.summarize.anthropic.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_lazy_client_creation(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message_response("s")

        summarize_item(title="T", text="txt", interest_description="d")

        mock_cls.assert_called_once_with(api_key="sk-ant-test")
