from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from patronus.config import Config, DigestConfig, EmbeddingConfig, PollingConfig, SummarizationConfig, TelegramConfig, TopicConfig
from patronus.interests import load_interest_vectors


@pytest.fixture()
def config() -> Config:
    return Config(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(model="text-embedding-3-small"),
        summarization=SummarizationConfig(),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="ML", description="Machine learning research."),
            "phil": TopicConfig(name="Philosophy", description="Philosophy of mind."),
            "spain": TopicConfig(name="Spain", description="Spanish politics and culture."),
        },
    )


class TestLoadInterestVectors:
    @patch("patronus.interests.embed_batch")
    def test_returns_dict_with_correct_keys(
        self, mock_embed: MagicMock, config: Config
    ) -> None:
        mock_embed.return_value = [
            np.ones(1536, dtype=np.float32),
            np.ones(1536, dtype=np.float32) * 2,
            np.ones(1536, dtype=np.float32) * 3,
        ]
        result = load_interest_vectors(config)
        assert set(result.keys()) == {"ml", "phil", "spain"}

    @patch("patronus.interests.embed_batch")
    def test_values_are_ndarrays(self, mock_embed: MagicMock, config: Config) -> None:
        mock_embed.return_value = [
            np.ones(10, dtype=np.float32) for _ in range(3)
        ]
        result = load_interest_vectors(config)
        for v in result.values():
            assert isinstance(v, np.ndarray)

    @patch("patronus.interests.embed_batch")
    def test_passes_descriptions_and_model(
        self, mock_embed: MagicMock, config: Config
    ) -> None:
        mock_embed.return_value = [np.zeros(10) for _ in range(3)]
        load_interest_vectors(config)

        call_args = mock_embed.call_args
        texts = call_args[0][0]
        assert len(texts) == 3
        assert "Machine learning research." in texts
        assert "Philosophy of mind." in texts
        assert "Spanish politics and culture." in texts
        assert call_args[1]["model"] == "text-embedding-3-small"

    @patch("patronus.interests.embed_batch")
    def test_empty_topics(self, mock_embed: MagicMock) -> None:
        cfg = Config(
            digest=DigestConfig(),
            polling=PollingConfig(),
            embedding=EmbeddingConfig(),
            summarization=SummarizationConfig(),
            telegram=TelegramConfig(),
            topics={},
        )
        mock_embed.return_value = []
        result = load_interest_vectors(cfg)
        assert result == {}
