from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from patronus.config import (
    AgentConfig,
    Config,
    DigestConfig,
    EmbeddingConfig,
    PollingConfig,
    SummarizationConfig,
    TelegramConfig,
    TopicConfig,
)
from patronus.context import Context
from patronus.db import Database
from patronus.digest import Digest, DigestItem, DigestSection, SectionType
from patronus.pipeline import DigestPipeline, _build_default_sources, _build_tool_registry
from patronus.rank import ScoredItem


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_config(mode: str = "deterministic", **overrides: object) -> Config:
    return Config(
        digest=DigestConfig(mode=mode),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        summarization=SummarizationConfig(),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="Technical AI/ML", description="ML research"),
        },
        agent=AgentConfig(max_iterations=3, max_tokens=2000),
    )


def _make_item_obj():
    from patronus.db import Item
    return Item(
        id="item-1",
        url="https://example.com",
        source_type="rss",
        title="Test Article",
        source="Test Blog",
        text="Content",
        timestamp="2026-02-15T00:00:00Z",
    )


def _make_agent_digest() -> Digest:
    return Digest(
        sections=[
            DigestSection(
                type=SectionType.LONG_FORM_PICK,
                title="Today's Pick",
                items=[DigestItem(
                    item_id="item-1",
                    title="Great Paper",
                    url="https://example.com",
                    summary="Excellent.",
                )],
            )
        ],
        generated_at="2026-02-16T08:00:00Z",
        mode="agent",
    )


class TestBuildDefaultSources:
    def test_always_includes_interests(self, tmp_path: object) -> None:
        config = _make_config()
        db = Database(db_path=str(tmp_path) + "/test.db")
        sources = _build_default_sources(config, db)
        from patronus.interests import InterestsSource
        assert any(isinstance(s, InterestsSource) for s in sources)
        db.close()

    def test_no_notion_without_config(self, tmp_path: object) -> None:
        config = _make_config()
        config.notion = None
        db = Database(db_path=str(tmp_path) + "/test.db")
        sources = _build_default_sources(config, db)
        assert len(sources) == 1
        db.close()


class TestBuildToolRegistry:
    def test_registers_local_and_arxiv(self, tmp_path: object) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config()
        registry = _build_tool_registry(config, db)
        names = set(registry.tool_names)
        assert "search_similar" in names
        assert "search_recent" in names
        assert "search_by_topic" in names
        assert "search_by_source" in names
        assert "search_arxiv" in names
        db.close()


class TestDigestPipelineDeterministic:
    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_run_deterministic(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="deterministic")

        mock_interests.return_value = {"ml": _unit_vec(1.0, 0.0)}
        mock_summarize.return_value = "Summary text."

        emb = _unit_vec(0.9, 0.1)
        db.add_item(url="https://a.com", source_type="rss", title="Article",
                     text="Content", embedding=emb, timestamp="2026-02-15T00:00:00Z")

        pipeline = DigestPipeline(config, db, sources=[], outputs=[])
        digest = pipeline.run()

        assert digest.mode == "deterministic"
        assert digest.item_count == 1

        digests = db.get_latest_digests(1)
        assert len(digests) == 1
        db.close()

    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_generate_deterministic(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="deterministic")

        mock_interests.return_value = {"ml": _unit_vec(1.0, 0.0)}
        mock_summarize.return_value = "Summary."

        emb = _unit_vec(1.0, 0.0)
        db.add_item(url="https://a.com", source_type="rss", title="Article",
                     text="Content", embedding=emb, timestamp="2026-02-15T00:00:00Z")

        pipeline = DigestPipeline(config, db, sources=[], outputs=[])
        digest = pipeline.generate()

        assert digest.mode == "deterministic"
        assert digest.item_count == 1
        db.close()


class TestDigestPipelineAgent:
    @patch.object(DigestPipeline, "_save_digest")
    @patch("patronus.pipeline.plan_and_assemble")
    @patch("patronus.pipeline.merge_sources")
    def test_agent_mode_generates_digest(
        self,
        mock_merge: MagicMock,
        mock_agent: MagicMock,
        mock_save: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="agent")

        mock_merge.return_value = Context(prose="Reader context here.", vectors={})
        mock_agent.return_value = _make_agent_digest()

        pipeline = DigestPipeline(config, db, sources=[], outputs=[])
        digest = pipeline.run()

        assert digest.mode == "agent"
        assert digest.item_count == 1
        mock_agent.assert_called_once()
        mock_save.assert_called_once()
        db.close()

    @patch.object(DigestPipeline, "_save_digest")
    @patch("patronus.pipeline.plan_and_assemble")
    @patch("patronus.pipeline.merge_sources")
    @patch("patronus.pipeline.generate_digest_deterministic")
    def test_agent_empty_falls_back_to_deterministic(
        self,
        mock_det: MagicMock,
        mock_merge: MagicMock,
        mock_agent: MagicMock,
        mock_save: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="agent")

        mock_merge.return_value = Context(prose="Some context.", vectors={})
        mock_agent.return_value = Digest(generated_at="2026-02-16T08:00:00Z", mode="agent")
        mock_det.return_value = Digest(
            items=[DigestItem(summary="Fallback", scored_item=ScoredItem(
                item=_make_item_obj(), score=0.9, matched_topic="ml", raw_similarity=0.9,
            ))],
            generated_at="2026-02-16T08:00:00Z",
            mode="deterministic",
        )

        pipeline = DigestPipeline(config, db, sources=[], outputs=[])
        digest = pipeline.run()

        assert digest.mode == "deterministic"
        mock_det.assert_called_once()
        db.close()

    @patch.object(DigestPipeline, "_save_digest")
    @patch("patronus.pipeline.merge_sources")
    @patch("patronus.pipeline.generate_digest_deterministic")
    def test_no_context_falls_back_to_deterministic(
        self,
        mock_det: MagicMock,
        mock_merge: MagicMock,
        mock_save: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="agent")

        mock_merge.return_value = Context(prose="", vectors={})
        mock_det.return_value = Digest(generated_at="2026-02-16T08:00:00Z", mode="deterministic")

        pipeline = DigestPipeline(config, db, sources=[], outputs=[])
        digest = pipeline.run()

        assert digest.mode == "deterministic"
        mock_det.assert_called_once()
        db.close()


class TestDigestPipelineOutputs:
    @patch.object(DigestPipeline, "_save_digest")
    @patch("patronus.pipeline.plan_and_assemble")
    @patch("patronus.pipeline.merge_sources")
    def test_outputs_dispatched(
        self,
        mock_merge: MagicMock,
        mock_agent: MagicMock,
        mock_save: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="agent")

        mock_merge.return_value = Context(prose="Context.", vectors={})
        mock_agent.return_value = _make_agent_digest()

        output1 = MagicMock()
        output2 = MagicMock()

        pipeline = DigestPipeline(config, db, sources=[], outputs=[output1, output2])
        pipeline.run()

        output1.send.assert_called_once()
        output2.send.assert_called_once()
        db.close()

    @patch.object(DigestPipeline, "_save_digest")
    @patch("patronus.pipeline.plan_and_assemble")
    @patch("patronus.pipeline.merge_sources")
    def test_output_failure_does_not_crash(
        self,
        mock_merge: MagicMock,
        mock_agent: MagicMock,
        mock_save: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="agent")

        mock_merge.return_value = Context(prose="Context.", vectors={})
        mock_agent.return_value = _make_agent_digest()

        failing_output = MagicMock()
        failing_output.send.side_effect = RuntimeError("Telegram API down")
        ok_output = MagicMock()

        pipeline = DigestPipeline(config, db, sources=[], outputs=[failing_output, ok_output])
        digest = pipeline.run()

        assert digest.item_count == 1
        ok_output.send.assert_called_once()
        db.close()

    @patch("patronus.digest.summarize_item")
    @patch("patronus.digest.load_interest_vectors")
    def test_save_digest_records_to_db(
        self,
        mock_interests: MagicMock,
        mock_summarize: MagicMock,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test.db")
        config = _make_config(mode="deterministic")

        mock_interests.return_value = {"ml": _unit_vec(1.0, 0.0)}
        mock_summarize.return_value = "Summary."

        emb = _unit_vec(1.0, 0.0)
        db.add_item(url="https://a.com", source_type="rss", title="Article",
                     text="Content", embedding=emb, timestamp="2026-02-15T00:00:00Z")

        pipeline = DigestPipeline(config, db, sources=[], outputs=[])
        pipeline.run()

        digests = db.get_latest_digests(5)
        assert len(digests) >= 1
        db.close()
