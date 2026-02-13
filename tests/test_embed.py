from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from patronus.embed import embed_batch, embed_text


def _make_embedding_response(embeddings: list[list[float]]) -> SimpleNamespace:
    data = [
        SimpleNamespace(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]
    return SimpleNamespace(data=data)


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    import patronus.embed as mod
    mod._client = None


class TestEmbedText:
    @patch("patronus.embed._get_client")
    def test_returns_float32_ndarray(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.embeddings.create.return_value = _make_embedding_response(
            [[0.1, 0.2, 0.3]]
        )
        result = embed_text("hello")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3], atol=1e-6)

    @patch("patronus.embed._get_client")
    def test_passes_model_kwarg(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.embeddings.create.return_value = _make_embedding_response(
            [[0.1]]
        )
        embed_text("hello", model="text-embedding-3-large")
        client.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-large"
        )

    @patch("patronus.embed._get_client")
    def test_default_model(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.embeddings.create.return_value = _make_embedding_response(
            [[0.1]]
        )
        embed_text("hello")
        client.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-small"
        )


class TestEmbedBatch:
    @patch("patronus.embed._get_client")
    def test_returns_list_of_ndarrays(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        client.embeddings.create.return_value = _make_embedding_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        results = embed_batch(["a", "b"])
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.dtype == np.float32 for r in results)

    def test_empty_input_returns_empty(self) -> None:
        assert embed_batch([]) == []

    @patch("patronus.embed._get_client")
    def test_preserves_order_from_response_index(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        mock_get.return_value = client
        data = [
            SimpleNamespace(embedding=[0.9, 0.9], index=1),
            SimpleNamespace(embedding=[0.1, 0.1], index=0),
        ]
        client.embeddings.create.return_value = SimpleNamespace(data=data)

        results = embed_batch(["first", "second"])
        np.testing.assert_allclose(results[0], [0.1, 0.1], atol=1e-6)
        np.testing.assert_allclose(results[1], [0.9, 0.9], atol=1e-6)


class TestLazyClient:
    @patch("patronus.embed.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    def test_creates_client_on_first_call(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _make_embedding_response(
            [[0.1]]
        )

        embed_text("test")

        mock_openai_cls.assert_called_once_with(api_key="sk-test-key")
