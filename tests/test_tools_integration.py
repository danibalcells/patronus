from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from patronus.config import Config, DigestConfig, EmbeddingConfig, PollingConfig, TelegramConfig, TopicConfig
from patronus.db import Database
from patronus.tools import ToolRegistry
from patronus.tools.local import SearchBySource, SearchByTopic, SearchRecent, SearchSimilar, register_local_tools


def _unit_vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def integration_config() -> Config:
    return Config(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(model="text-embedding-3-small"),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(
                name="Technical AI/ML",
                description="Machine learning and AI research focusing on transformers, attention, and model interpretability",
            ),
            "tech": TopicConfig(
                name="Tech Strategy",
                description="Technology industry analysis, product strategy, and startup ecosystems",
            ),
            "philosophy": TopicConfig(
                name="Philosophy",
                description="Philosophy of mind, consciousness, and cognitive science",
            ),
        },
    )


@pytest.fixture
def test_db_with_real_embeddings(tmp_path: object, integration_config: Config) -> Database:
    from patronus.embed import embed_text
    
    db = Database(db_path=str(tmp_path) + "/test_tools_integration.db")
    
    now = datetime.now(timezone.utc)
    recent_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    old_ts = "2024-01-01T00:00:00Z"
    
    papers = [
        {
            "url": "https://arxiv.org/abs/2024.01",
            "title": "Attention Is All You Need: Revisited",
            "text": "This paper revisits the original transformer architecture and explores improvements to the attention mechanism. We demonstrate that sparse attention patterns can reduce computational complexity while maintaining model performance.",
            "topic": "ml",
        },
        {
            "url": "https://arxiv.org/abs/2024.02",
            "title": "Mechanistic Interpretability in Large Language Models",
            "text": "We present a comprehensive framework for understanding the internal representations of large language models through circuit analysis and feature visualization techniques.",
            "topic": "ml",
        },
        {
            "url": "https://techcrunch.com/ai-funding",
            "title": "AI Startups Raise Record $10B in Q1",
            "text": "Investment in artificial intelligence companies reached record levels this quarter, with infrastructure and application layer startups attracting the most capital.",
            "topic": "tech",
        },
        {
            "url": "https://stratechery.com/product-ai",
            "title": "The Changing Dynamics of AI Product Development",
            "text": "AI-native product development requires rethinking traditional software engineering practices. This analysis explores how successful companies are adapting their development processes.",
            "topic": "tech",
        },
        {
            "url": "https://philosophy-blog.com/consciousness",
            "title": "Computational Theories of Consciousness",
            "text": "An examination of various computational approaches to understanding consciousness, from global workspace theory to integrated information theory.",
            "topic": "philosophy",
        },
    ]
    
    for i, paper in enumerate(papers):
        embedding = embed_text(paper["text"], model=integration_config.embedding.model)
        timestamp = recent_ts if i < 4 else old_ts
        
        item_id = db.add_item(
            url=paper["url"],
            source_type="rss" if "arxiv" in paper["url"] or "techcrunch" in paper["url"] else "manual",
            title=paper["title"],
            author=f"Author {i}",
            source="Arxiv" if "arxiv" in paper["url"] else "Tech Blog",
            text=paper["text"],
            embedding=embedding,
            timestamp=timestamp,
            item_type="paper" if "arxiv" in paper["url"] else "article",
        )
        
        with db._session() as session:
            from patronus.db import Item
            item = session.get(Item, item_id)
            if item:
                item.topic_cluster = paper["topic"]
                session.add(item)
                session.commit()
    
    yield db
    db.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration test",
)
class TestSearchSimilarIntegration:
    def test_finds_semantically_similar_items(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchSimilar(integration_config, test_db_with_real_embeddings)
        
        result = tool.execute(query="transformer models and attention mechanisms", n=3)
        
        assert len(result.items) > 0
        
        first_item = result.items[0]
        assert "attention" in first_item["title"].lower() or "transformer" in first_item["title"].lower()

    def test_different_queries_return_different_results(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchSimilar(integration_config, test_db_with_real_embeddings)
        
        ml_result = tool.execute(query="machine learning interpretability", n=2)
        philosophy_result = tool.execute(query="theories of consciousness", n=2)
        
        assert len(ml_result.items) > 0
        assert len(philosophy_result.items) > 0
        
        ml_ids = {item["id"] for item in ml_result.items}
        phil_ids = {item["id"] for item in philosophy_result.items}
        
        assert ml_ids != phil_ids, "Different queries should return different items"

    def test_respects_n_parameter(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchSimilar(integration_config, test_db_with_real_embeddings)
        
        result_2 = tool.execute(query="AI research", n=2)
        result_5 = tool.execute(query="AI research", n=5)
        
        assert len(result_2.items) <= 2
        assert len(result_5.items) <= 5

    def test_returns_snippets_and_metadata(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchSimilar(integration_config, test_db_with_real_embeddings)
        
        result = tool.execute(query="AI", n=1)
        
        assert len(result.items) > 0
        item = result.items[0]
        
        assert "id" in item
        assert "title" in item
        assert "url" in item
        assert "snippet" in item
        assert len(item["snippet"]) > 0
        assert item["url"].startswith("http")


@pytest.mark.integration
class TestSearchRecentIntegration:
    def test_filters_by_recency(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchRecent(test_db_with_real_embeddings)
        
        result_recent = tool.execute(days=1, n=10)
        result_old = tool.execute(days=365, n=10)
        
        assert len(result_recent.items) >= 4
        assert len(result_old.items) >= len(result_recent.items)

    def test_orders_by_timestamp(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchRecent(test_db_with_real_embeddings)
        
        result = tool.execute(days=30, n=10)
        
        if len(result.items) >= 2:
            timestamps = [item.get("timestamp", "") for item in result.items if item.get("timestamp")]
            for i in range(len(timestamps) - 1):
                assert timestamps[i] >= timestamps[i + 1], "Items should be ordered newest first"


@pytest.mark.integration
class TestSearchByTopicIntegration:
    def test_filters_by_topic_cluster(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchByTopic(integration_config, test_db_with_real_embeddings)
        
        ml_result = tool.execute(topic="ml", n=10)
        tech_result = tool.execute(topic="tech", n=10)
        
        assert len(ml_result.items) > 0
        assert len(tech_result.items) > 0
        
        ml_ids = {item["id"] for item in ml_result.items}
        tech_ids = {item["id"] for item in tech_result.items}
        
        assert ml_ids.isdisjoint(tech_ids), "Different topics should return different items"

    def test_description_includes_available_topics(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchByTopic(integration_config, test_db_with_real_embeddings)
        
        description = tool.description
        assert "ml" in description.lower()
        assert "tech" in description.lower()
        assert "philosophy" in description.lower()


@pytest.mark.integration
class TestSearchBySourceIntegration:
    def test_filters_by_source_type(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchBySource(test_db_with_real_embeddings)
        
        rss_result = tool.execute(source_type="rss", n=10)
        manual_result = tool.execute(source_type="manual", n=10)
        
        assert len(rss_result.items) > 0
        
        for item in rss_result.items:
            assert item.get("id") is not None

    def test_filters_by_source_name(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchBySource(test_db_with_real_embeddings)
        
        result = tool.execute(source_name="Arxiv", n=10)
        
        assert len(result.items) > 0
        
        for item in result.items:
            assert "arxiv" in item.get("source", "").lower() or "arxiv" in item.get("url", "").lower()

    def test_returns_all_without_filters(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        tool = SearchBySource(test_db_with_real_embeddings)
        
        result = tool.execute(n=10)
        
        assert len(result.items) > 0


@pytest.mark.integration
class TestToolRegistryIntegration:
    def test_all_tools_registered_and_executable(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        registry = ToolRegistry()
        register_local_tools(registry, integration_config, test_db_with_real_embeddings)
        
        assert "search_similar" in registry.tool_names
        assert "search_recent" in registry.tool_names
        assert "search_by_topic" in registry.tool_names
        assert "search_by_source" in registry.tool_names
        
        definitions = registry.get_definitions()
        assert len(definitions) == 4
        
        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition

    def test_registry_execute_with_real_search(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        registry = ToolRegistry()
        register_local_tools(registry, integration_config, test_db_with_real_embeddings)
        
        result = registry.execute("search_recent", days=7, n=5)
        
        assert result.message != ""
        assert len(result.items) > 0

    def test_tool_results_are_agent_consumable(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        registry = ToolRegistry()
        register_local_tools(registry, integration_config, test_db_with_real_embeddings)
        
        result = registry.execute("search_recent", days=1, n=3)
        
        text_output = result.to_text()
        
        assert len(text_output) > 50
        
        assert "Title:" in text_output or "URL:" in text_output or "ID:" in text_output


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration test",
)
class TestToolCoordination:
    def test_combining_multiple_tools_for_diverse_results(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        similar_tool = SearchSimilar(integration_config, test_db_with_real_embeddings)
        recent_tool = SearchRecent(test_db_with_real_embeddings)
        topic_tool = SearchByTopic(integration_config, test_db_with_real_embeddings)
        
        similar_result = similar_tool.execute(query="AI research", n=2)
        recent_result = recent_tool.execute(days=7, n=2)
        topic_result = topic_tool.execute(topic="ml", n=2)
        
        all_ids = set()
        for result in [similar_result, recent_result, topic_result]:
            all_ids.update(item["id"] for item in result.items)
        
        assert len(all_ids) >= 3, "Multiple tools should surface diverse content"

    def test_tools_work_with_registry_in_sequence(
        self,
        integration_config: Config,
        test_db_with_real_embeddings: Database,
    ) -> None:
        registry = ToolRegistry()
        register_local_tools(registry, integration_config, test_db_with_real_embeddings)
        
        result1 = registry.execute("search_recent", days=1, n=5)
        result2 = registry.execute("search_by_topic", topic="ml", n=3)
        result3 = registry.execute("search_by_source", source_type="rss", n=5)
        
        assert len(result1.items) > 0
        assert len(result2.items) > 0
        assert len(result3.items) > 0
        
        for result in [result1, result2, result3]:
            text = result.to_text()
            assert "Title:" in text or "No results" in text
