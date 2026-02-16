from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from patronus.agent import plan_and_assemble
from patronus.config import AgentConfig, Config, DigestConfig, EmbeddingConfig, PollingConfig, SummarizationConfig, TelegramConfig, TopicConfig
from patronus.context import Context
from patronus.db import Database
from patronus.tools import ToolRegistry
from patronus.tools.local import register_local_tools


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def integration_config() -> Config:
    return Config(
        digest=DigestConfig(mode="agent", size=5, max_per_topic=3),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(model="text-embedding-3-small"),
        summarization=SummarizationConfig(),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="Technical AI/ML", description="Machine learning and artificial intelligence research, especially mechanistic interpretability and training dynamics"),
            "tech": TopicConfig(name="Tech Strategy", description="Technology strategy, product development, and industry analysis"),
        },
        agent=AgentConfig(
            model="anthropic/claude-sonnet-4-20250514",
            max_iterations=10,
            max_tokens=4096,
        ),
    )


@pytest.fixture
def test_db(tmp_path: object) -> Database:
    db = Database(db_path=str(tmp_path) + "/test_integration.db")
    
    now = datetime.now(timezone.utc)
    recent_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    emb_ml = _unit_vec(1.0, 0.1, 0.0)
    emb_tech = _unit_vec(0.1, 1.0, 0.0)
    emb_phil = _unit_vec(0.0, 0.1, 1.0)
    
    db.add_item(
        url="https://arxiv.org/abs/2024.12345",
        source_type="rss",
        title="Attention Mechanisms in Transformer Models",
        author="Jane Smith et al.",
        source="Arxiv RSS",
        text="This paper explores novel attention mechanisms that improve transformer efficiency by 40%. We demonstrate that sparse attention patterns can be learned dynamically during training, leading to better generalization on downstream tasks.",
        embedding=emb_ml,
        timestamp=recent_ts,
        item_type="paper",
    )
    
    db.add_item(
        url="https://arxiv.org/abs/2024.54321",
        source_type="rss",
        title="Mechanistic Interpretability of Language Models",
        author="Alice Johnson",
        source="Arxiv RSS",
        text="We present a comprehensive framework for understanding the internal representations of large language models. Our analysis reveals distinct computational circuits responsible for different linguistic phenomena.",
        embedding=emb_ml,
        timestamp=recent_ts,
        item_type="paper",
    )
    
    db.add_item(
        url="https://techcrunch.com/ai-startup-funding",
        source_type="rss",
        title="AI Startup Secures $500M in Series C Funding",
        author="TechCrunch Staff",
        source="TechCrunch",
        text="Leading AI infrastructure company raises massive round to compete with OpenAI and Anthropic. The funding will be used to scale compute and hire research talent.",
        embedding=emb_tech,
        timestamp=recent_ts,
        item_type="article",
    )
    
    db.add_item(
        url="https://stratechery.com/product-strategy",
        source_type="rss",
        title="The Evolution of Product Strategy in AI-Native Companies",
        author="Ben Thompson",
        source="Stratechery",
        text="AI-native companies are fundamentally different from traditional software businesses. This analysis explores how product development, distribution, and monetization strategies must adapt to the new paradigm.",
        embedding=emb_tech,
        timestamp=recent_ts,
        item_type="article",
    )
    
    db.add_item(
        url="https://blog.example.com/philosophy",
        source_type="manual",
        title="Consciousness and Computation: A Philosophical Exploration",
        author="David Williams",
        source="Philosophy Blog",
        text="This essay examines the relationship between consciousness and computational systems, drawing on both analytic philosophy and cognitive science.",
        embedding=emb_phil,
        timestamp=recent_ts,
        item_type="article",
    )
    
    db.add_item(
        url="https://twitter.com/researcher/status/123",
        source_type="rss",
        title="Tweet: Breakthrough in RL from DeepMind",
        author="@researcher",
        source="Twitter",
        text="Just read the new DeepMind paper on hierarchical RL. The results are stunning - they've achieved human-level performance on StarCraft II with 10x less compute. Thread below...",
        embedding=emb_ml,
        timestamp=recent_ts,
        item_type="tweet",
    )
    
    yield db
    db.close()


@pytest.fixture
def tool_registry(integration_config: Config, test_db: Database) -> ToolRegistry:
    registry = ToolRegistry()
    register_local_tools(registry, integration_config, test_db)
    return registry


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration test",
)
class TestAgentIntegration:
    def test_agent_generates_digest_with_sections(
        self,
        integration_config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        context = Context(
            prose=(
                "The reader is a machine learning researcher currently working on "
                "mechanistic interpretability. They are particularly interested in "
                "understanding how transformers learn attention patterns and how "
                "different components contribute to model behavior. They also follow "
                "AI industry news and product strategy in the tech sector."
            ),
            vectors={},
        )
        
        digest = plan_and_assemble(integration_config, context, tool_registry)
        
        assert digest.mode == "agent"
        assert digest.item_count > 0
        assert len(digest.sections) > 0
        
        assert digest.generated_at is not None
        
        for section in digest.sections:
            assert section.title != ""
            assert len(section.items) > 0
            
            for item in section.items:
                assert item.item_id != ""
                assert item.title != ""
                assert item.url != ""
                assert item.summary != ""

    def test_agent_uses_search_tools(
        self,
        integration_config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        context = Context(
            prose=(
                "The reader wants to catch up on recent ML papers, especially "
                "anything related to attention mechanisms or interpretability."
            ),
            vectors={},
        )
        
        digest = plan_and_assemble(integration_config, context, tool_registry)
        
        assert digest.item_count > 0
        
        found_attention = False
        found_interpretability = False
        
        for section in digest.sections:
            for item in section.items:
                if "attention" in item.title.lower() or "attention" in item.summary.lower():
                    found_attention = True
                if "interpretability" in item.title.lower() or "interpretability" in item.summary.lower():
                    found_interpretability = True
        
        assert found_attention or found_interpretability, "Agent should have found relevant ML papers"

    def test_agent_handles_empty_context(
        self,
        integration_config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        context = Context(prose="General technology and AI news.", vectors={})
        
        digest = plan_and_assemble(integration_config, context, tool_registry)
        
        assert digest.item_count >= 0

    def test_agent_creates_diverse_sections(
        self,
        integration_config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        context = Context(
            prose=(
                "The reader is interested in ML research papers, tech industry news, "
                "and occasionally philosophy. They appreciate both depth and breadth "
                "in their reading."
            ),
            vectors={},
        )
        
        digest = plan_and_assemble(integration_config, context, tool_registry)
        
        section_types = {section.type for section in digest.sections}
        assert len(section_types) >= 1, "Agent should create at least one section type"

    def test_agent_respects_context_personalization(
        self,
        integration_config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        ml_context = Context(
            prose=(
                "The reader is exclusively focused on technical machine learning research. "
                "They are not interested in business news or general tech commentary."
            ),
            vectors={},
        )
        
        digest = plan_and_assemble(integration_config, ml_context, tool_registry)
        
        assert digest.item_count > 0
        
        ml_related = 0
        total = 0
        
        for section in digest.sections:
            for item in section.items:
                total += 1
                item_text = (item.title + " " + item.summary).lower()
                if any(kw in item_text for kw in ["ml", "machine learning", "model", "paper", "research", "transformer", "attention"]):
                    ml_related += 1
        
        if total > 0:
            ml_ratio = ml_related / total
            assert ml_ratio >= 0.5, f"Expected mostly ML content, got {ml_related}/{total} ML-related items"

    def test_agent_writes_section_appropriate_summaries(
        self,
        integration_config: Config,
        tool_registry: ToolRegistry,
    ) -> None:
        context = Context(
            prose="The reader is interested in ML research and tech strategy.",
            vectors={},
        )
        
        digest = plan_and_assemble(integration_config, context, tool_registry)
        
        assert digest.item_count > 0
        
        for section in digest.sections:
            for item in section.items:
                summary_len = len(item.summary.split())
                assert summary_len > 3, f"Summary too short: {item.summary}"
                assert summary_len < 200, f"Summary too long: {item.summary}"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping integration test",
)
class TestToolsIntegration:
    def test_search_similar_finds_relevant_items(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        from patronus.tools.local import SearchSimilar
        
        tool = SearchSimilar(integration_config, test_db)
        result = tool.execute(query="transformer attention mechanisms", n=3)
        
        assert len(result.items) > 0
        assert "attention" in result.items[0]["title"].lower() or "transformer" in result.items[0]["title"].lower()

    def test_search_recent_returns_fresh_items(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        from patronus.tools.local import SearchRecent
        
        tool = SearchRecent(test_db)
        result = tool.execute(days=1, n=10)
        
        assert len(result.items) > 0

    def test_search_by_topic_filters_correctly(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        from patronus.tools.local import SearchByTopic
        
        with test_db._session() as session:
            from patronus.db import Item
            items = list(session.query(Item).filter_by(read=False).all())
            for item in items[:2]:
                item.topic_cluster = "ml"
                session.add(item)
            session.commit()
        
        tool = SearchByTopic(integration_config, test_db)
        result = tool.execute(topic="ml", n=5)
        
        assert len(result.items) >= 1

    def test_search_by_source_filters_correctly(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        from patronus.tools.local import SearchBySource
        
        tool = SearchBySource(test_db)
        result = tool.execute(source_type="rss", n=10)
        
        assert len(result.items) > 0
        for item in result.items:
            assert item.get("id") is not None

    def test_tools_return_valid_structured_data(
        self,
        integration_config: Config,
        test_db: Database,
    ) -> None:
        from patronus.tools.local import SearchRecent
        
        tool = SearchRecent(test_db)
        result = tool.execute(days=1, n=5)
        
        assert result.message != ""
        
        for item in result.items:
            assert "id" in item
            assert "title" in item
            assert "url" in item
            assert "snippet" in item
