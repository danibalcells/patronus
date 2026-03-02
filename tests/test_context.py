from __future__ import annotations

import numpy as np

from patronus.config import Config, DigestConfig, EmbeddingConfig, NotionConfig, PollingConfig, TelegramConfig
from patronus.context import Context, PersonalizationSource, merge_sources


def _make_config(**overrides: object) -> Config:
    defaults: dict = dict(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        telegram=TelegramConfig(),
        topics={},
    )
    defaults.update(overrides)
    return Config(**defaults)


class FakeSourceA:
    def get_context(self, config: Config) -> str:
        return "Context from source A."

    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        return {"topic_a": np.array([1.0, 0.0, 0.0], dtype=np.float32)}


class FakeSourceB:
    def get_context(self, config: Config) -> str:
        return "Context from source B."

    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        return None


class FailingSource:
    def get_context(self, config: Config) -> str:
        raise RuntimeError("Notion is down")

    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        raise RuntimeError("Notion is down")


class EmptySource:
    def get_context(self, config: Config) -> str:
        return ""

    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        return {}


class TestPersonalizationSourceProtocol:
    def test_fake_source_is_personalization_source(self) -> None:
        assert isinstance(FakeSourceA(), PersonalizationSource)

    def test_failing_source_is_personalization_source(self) -> None:
        assert isinstance(FailingSource(), PersonalizationSource)


class TestContext:
    def test_default_context(self) -> None:
        ctx = Context()
        assert ctx.prose == ""
        assert ctx.vectors == {}

    def test_context_with_values(self) -> None:
        vecs = {"a": np.array([1.0])}
        ctx = Context(prose="hello", vectors=vecs)
        assert ctx.prose == "hello"
        assert "a" in ctx.vectors


class TestMergeSources:
    def test_merges_two_sources(self) -> None:
        config = _make_config()
        result = merge_sources([FakeSourceA(), FakeSourceB()], config)

        assert "Context from source A." in result.prose
        assert "Context from source B." in result.prose
        assert "topic_a" in result.vectors

    def test_skips_failing_source(self) -> None:
        config = _make_config()
        result = merge_sources([FailingSource(), FakeSourceA()], config)

        assert "Context from source A." in result.prose
        assert "topic_a" in result.vectors

    def test_empty_sources(self) -> None:
        config = _make_config()
        result = merge_sources([], config)

        assert result.prose == ""
        assert result.vectors == {}

    def test_skips_empty_context_strings(self) -> None:
        config = _make_config()
        result = merge_sources([EmptySource(), FakeSourceA()], config)

        assert result.prose == "Context from source A."
        assert "\n\n" not in result.prose

    def test_skips_empty_vector_dicts(self) -> None:
        config = _make_config()
        result = merge_sources([EmptySource()], config)

        assert result.vectors == {}

    def test_vectors_from_later_source_override(self) -> None:
        class SourceWithOverlap:
            def get_context(self, config: Config) -> str:
                return ""

            def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
                return {"topic_a": np.array([0.0, 1.0, 0.0], dtype=np.float32)}

        config = _make_config()
        result = merge_sources([FakeSourceA(), SourceWithOverlap()], config)

        np.testing.assert_array_equal(result.vectors["topic_a"], [0.0, 1.0, 0.0])
