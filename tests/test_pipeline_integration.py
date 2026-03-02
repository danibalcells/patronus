from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from patronus.config import AgentConfig, Config, DigestConfig, EmbeddingConfig, PollingConfig, TelegramConfig, TopicConfig
from patronus.context import Context, PersonalizationSource
from patronus.db import Database
from patronus.pipeline import DigestPipeline


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def integration_config() -> Config:
    return Config(
        digest=DigestConfig(mode="agent", size=10, max_per_topic=3),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(model="text-embedding-3-small"),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="Technical AI/ML", description="Machine learning research"),
            "tech": TopicConfig(name="Tech Strategy", description="Technology industry and product strategy"),
        },
        agent=AgentConfig(
            model="anthropic/claude-sonnet-4-20250514",
            max_iterations=10,
            max_tokens=4096,
        ),
    )


@pytest.fixture
def test_db(tmp_path: object) -> Database:
    db = Database(db_path=str(tmp_path) + "/test_pipeline_integration.db")
    
    now = datetime.now(timezone.utc)
    recent_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    emb_ml = _unit_vec(1.0, 0.0, 0.0)
    emb_tech = _unit_vec(0.0, 1.0, 0.0)
    
    for i in range(5):
        db.add_item(
            url=f"https://arxiv.org/abs/2024.{i:05d}",
            source_type="rss",
            title=f"ML Research Paper {i}: Attention and Transformers",
            author=f"Researcher {i}",
            source="Arxiv",
            text=f"This paper explores transformer architectures and attention mechanisms. Novel contribution {i} demonstrates improved performance on benchmark tasks.",
            embedding=emb_ml,
            timestamp=recent_ts,
            item_type="paper",
        )
    
    for i in range(5):
        db.add_item(
            url=f"https://techblog.com/article-{i}",
            source_type="rss",
            title=f"Tech Article {i}: AI Industry Analysis",
            author=f"Analyst {i}",
            source="Tech Blog",
            text=f"Analysis of AI industry trends and company strategies. Article {i} covers market dynamics and product development.",
            embedding=emb_tech,
            timestamp=recent_ts,
            item_type="article",
        )
    
    yield db
    db.close()


class MockPersonalizationSource(PersonalizationSource):
    def __init__(self, context: str) -> None:
        self._context = context
    
    def get_context(self, config: Config, notion_force_refresh: bool = False) -> str:
        return self._context
    
    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        return None


class MockOutput:
    def __init__(self) -> None:
        self.digests_sent: list = []
    
    def send(self, digest, config: Config) -> None:
        self.digests_sent.append(digest)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration test",
)
class TestPipelineIntegration:
    def test_full_pipeline_agent_mode(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source = MockPersonalizationSource(
            "The reader is a machine learning researcher interested in transformer models and attention mechanisms."
        )
        output = MockOutput()
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source],
            outputs=[output],
        )
        
        # Use cached Notion context by default (fast)
        digest = pipeline.run(notion_force_refresh=False)
        
        assert digest.mode == "agent"
        assert digest.item_count > 0
        assert len(digest.sections) > 0
        
        assert len(output.digests_sent) == 1
        assert output.digests_sent[0] is digest
        
        digests = test_db.get_latest_digests(1)
        assert len(digests) == 1

    def test_pipeline_with_multiple_sources(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source1 = MockPersonalizationSource("The reader works on ML research, specifically interpretability.")
        source2 = MockPersonalizationSource("The reader also follows tech industry news and product strategy.")
        
        output = MockOutput()
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source1, source2],
            outputs=[output],
        )
        
        digest = pipeline.run(notion_force_refresh=False)
        
        assert digest.item_count > 0
        assert len(output.digests_sent) == 1

    def test_pipeline_with_multiple_outputs(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source = MockPersonalizationSource("The reader is interested in ML and tech.")
        
        output1 = MockOutput()
        output2 = MockOutput()
        output3 = MockOutput()
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source],
            outputs=[output1, output2, output3],
        )
        
        digest = pipeline.run(notion_force_refresh=False)
        
        assert len(output1.digests_sent) == 1
        assert len(output2.digests_sent) == 1
        assert len(output3.digests_sent) == 1
        
        assert output1.digests_sent[0] is digest
        assert output2.digests_sent[0] is digest
        assert output3.digests_sent[0] is digest

    def test_pipeline_handles_failing_output(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source = MockPersonalizationSource("The reader is interested in ML research.")
        
        class FailingOutput:
            def send(self, digest, config: Config) -> None:
                raise RuntimeError("Output failure simulation")
        
        ok_output = MockOutput()
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source],
            outputs=[FailingOutput(), ok_output],
        )
        
        digest = pipeline.run(notion_force_refresh=False)
        
        assert digest.item_count >= 0
        assert len(ok_output.digests_sent) == 1

    def test_pipeline_saves_digest_to_database(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source = MockPersonalizationSource("Reader interested in ML and tech.")
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source],
            outputs=[],
        )
        
        before_count = len(test_db.get_latest_digests(100))
        
        digest = pipeline.run(notion_force_refresh=False)
        
        after_count = len(test_db.get_latest_digests(100))
        
        assert after_count == before_count + 1
        
        latest = test_db.get_latest_digests(1)[0]
        assert latest["item_count"] == digest.item_count

    def test_pipeline_fallback_to_deterministic_on_empty_agent_digest(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source = MockPersonalizationSource("Very vague context")
        
        integration_config.agent.max_iterations = 1
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source],
            outputs=[],
        )
        
        digest = pipeline.run()
        
        assert digest is not None

    def test_pipeline_generate_without_outputs(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        source = MockPersonalizationSource("Reader interested in ML research.")
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=[source],
            outputs=[],
        )
        
        digest = pipeline.generate()
        
        assert digest.mode == "agent"
        assert digest.item_count >= 0
        
        digests = test_db.get_latest_digests(1)
        assert len(digests) == 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"),
    reason="API keys not set - skipping integration test",
)
class TestPipelineDeterministicIntegration:
    def test_deterministic_pipeline_with_real_embeddings(
        self,
        tmp_path: object,
    ) -> None:
        config = Config(
            digest=DigestConfig(mode="deterministic", size=3, max_per_topic=2),
            polling=PollingConfig(),
            embedding=EmbeddingConfig(model="text-embedding-3-small"),
            telegram=TelegramConfig(),
            topics={
                "ml": TopicConfig(
                    name="Technical AI/ML",
                    description="Machine learning research, especially transformer models and attention mechanisms",
                ),
            },
            agent=AgentConfig(digest_summary_model="claude-haiku-4-5-20251001"),
        )
        
        db = Database(db_path=str(tmp_path) + "/test_det.db")
        
        now = datetime.now(timezone.utc)
        recent_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        from patronus.embed import embed_text
        
        for i in range(5):
            text = f"Research paper about transformer models and attention mechanisms. Paper {i} explores novel architectures."
            emb = embed_text(text, model=config.embedding.model)
            db.add_item(
                url=f"https://arxiv.org/abs/2024.{i}",
                source_type="rss",
                title=f"Transformer Paper {i}",
                text=text,
                embedding=emb,
                timestamp=recent_ts,
            )
        
        from patronus.interests import InterestsSource
        
        pipeline = DigestPipeline(
            config,
            db,
            sources=[InterestsSource()],
            outputs=[],
        )
        
        digest = pipeline.run(notion_force_refresh=False)
        
        assert digest.mode == "deterministic"
        assert digest.item_count > 0
        assert digest.item_count <= config.digest.size
        
        db.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration test",
)
class TestEndToEndScenarios:
    def test_complete_daily_digest_workflow(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        from patronus.interests import InterestsSource
        from patronus.output.terminal import TerminalOutput
        
        sources = [InterestsSource()]
        
        class CollectingOutput:
            def __init__(self) -> None:
                self.digest = None
            
            def send(self, digest, config: Config) -> None:
                self.digest = digest
        
        collector = CollectingOutput()
        outputs = [collector]
        
        pipeline = DigestPipeline(
            integration_config,
            test_db,
            sources=sources,
            outputs=outputs,
        )
        
        digest = pipeline.run(notion_force_refresh=False)
        
        assert digest is not None
        assert digest.mode == "agent"
        assert digest.item_count > 0
        
        assert collector.digest is digest
        
        for section in digest.sections:
            assert section.title != ""
            for item in section.items:
                assert item.item_id != ""
                assert item.url.startswith("http")
                assert len(item.summary) > 10
        
        digests = test_db.get_latest_digests(1)
        assert len(digests) == 1
        assert digests[0]["item_count"] == digest.item_count

    def test_pipeline_handles_varied_content_types(
        self,
        integration_config: Config,
        tmp_path: object,
    ) -> None:
        db = Database(db_path=str(tmp_path) + "/test_varied.db")
        
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        emb = _unit_vec(1.0, 0.0, 0.0)
        
        db.add_item(
            url="https://arxiv.org/abs/2024.12345",
            source_type="rss",
            title="Research Paper",
            text="Academic paper text",
            embedding=emb,
            timestamp=ts,
            item_type="paper",
        )
        
        db.add_item(
            url="https://blog.com/article",
            source_type="rss",
            title="Tech Article",
            text="Blog article text",
            embedding=emb,
            timestamp=ts,
            item_type="article",
        )
        
        db.add_item(
            url="https://twitter.com/user/status/123",
            source_type="rss",
            title="Tweet",
            text="Tweet text",
            embedding=emb,
            timestamp=ts,
            item_type="tweet",
        )
        
        source = MockPersonalizationSource("Reader interested in ML and tech.")
        
        pipeline = DigestPipeline(
            integration_config,
            db,
            sources=[source],
            outputs=[],
        )
        
        digest = pipeline.run(notion_force_refresh=False)
        
        assert digest.item_count >= 0
        
        db.close()
