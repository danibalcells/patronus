from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlmodel import select

from patronus.config import Config
from patronus.db import Database, Item, deserialize_embedding
from patronus.embed import embed_text
from patronus.tools.base import ITEM_SNIPPET_MAX_CHARS, Tool, ToolResult


def _item_to_dict(item: Item) -> dict:
    snippet = (item.text or "")[:ITEM_SNIPPET_MAX_CHARS]
    return {
        "id": item.id,
        "title": item.title or "",
        "url": item.url,
        "source": item.source or "",
        "author": item.author or "",
        "item_type": item.item_type,
        "timestamp": item.timestamp or "",
        "snippet": snippet,
    }


class SearchSimilar(Tool):
    def __init__(self, config: Config, db: Database) -> None:
        self._config = config
        self._db = db

    @property
    def name(self) -> str:
        return "search_similar"

    @property
    def description(self) -> str:
        return (
            "Search for items semantically similar to a query string. "
            "Use this to find content related to a specific topic, concept, or question. "
            "Returns items ranked by cosine similarity to the query embedding."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing the topic or concept to find similar items for.",
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 10.",
                    "default": 10,
                },
            },
            "required": ["query"],
        }

    def execute(self, **params: object) -> ToolResult:
        query = str(params.get("query", ""))
        n = int(params.get("n", 10))

        if not query:
            return ToolResult(message="Query string is required.")

        query_embedding = embed_text(query, model=self._config.embedding.model)
        over_digested = self._db.get_over_digested_item_ids()
        items = self._db.get_unread_items()

        scored: list[tuple[float, Item]] = []
        for item in items:
            if item.id in over_digested:
                continue
            if item.embedding is None:
                continue
            emb = deserialize_embedding(item.embedding)
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                continue
            sim = float(np.dot(query_embedding, emb) / (norm_q * norm_e))
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n]

        return ToolResult(
            items=[_item_to_dict(item) for _, item in top],
            message=f"Found {len(top)} items similar to '{query}'.",
        )


class SearchRecent(Tool):
    def __init__(self, db: Database) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "search_recent"

    @property
    def description(self) -> str:
        return (
            "Get the most recent items from the content database. "
            "Use this to see what's been ingested recently. "
            "Returns items ordered by timestamp, newest first."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Defaults to 3.",
                    "default": 3,
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 15.",
                    "default": 15,
                },
            },
        }

    def execute(self, **params: object) -> ToolResult:
        days = int(params.get("days", 3))
        n = int(params.get("n", 15))

        now = datetime.now(timezone.utc)
        over_digested = self._db.get_over_digested_item_ids()
        items = self._db.get_unread_items()

        recent: list[Item] = []
        for item in items:
            if item.id in over_digested:
                continue
            if not item.timestamp:
                continue
            try:
                dt = datetime.strptime(item.timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if (now - dt).total_seconds() <= days * 86400:
                    recent.append(item)
            except (ValueError, TypeError):
                continue

        recent.sort(key=lambda x: x.timestamp or "", reverse=True)
        top = recent[:n]

        return ToolResult(
            items=[_item_to_dict(item) for item in top],
            message=f"Found {len(top)} items from the last {days} days.",
        )


class SearchByTopic(Tool):
    def __init__(self, config: Config, db: Database) -> None:
        self._config = config
        self._db = db

    @property
    def name(self) -> str:
        return "search_by_topic"

    @property
    def description(self) -> str:
        topic_names = [f"'{tc.name}' (key: '{key}')" for key, tc in self._config.topics.items()]
        return (
            "Search for items tagged with a specific topic cluster. "
            "Available topics: " + ", ".join(topic_names) + ". "
            "Use the topic key (not the display name) as the argument."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic key to filter by.",
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 10.",
                    "default": 10,
                },
            },
            "required": ["topic"],
        }

    def execute(self, **params: object) -> ToolResult:
        topic = str(params.get("topic", ""))
        n = int(params.get("n", 10))

        if not topic:
            return ToolResult(message="Topic key is required.")

        over_digested = self._db.get_over_digested_item_ids()
        with self._db._session() as session:
            query = select(Item).where(
                Item.read == False,
                Item.topic_cluster == topic,
            ).order_by(Item.timestamp.desc()).limit(n + len(over_digested))
            raw = list(session.exec(query).all())

        items = [item for item in raw if item.id not in over_digested][:n]

        return ToolResult(
            items=[_item_to_dict(item) for item in items],
            message=f"Found {len(items)} items in topic '{topic}'.",
        )


class SearchBySource(Tool):
    def __init__(self, db: Database) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "search_by_source"

    @property
    def description(self) -> str:
        return (
            "Search for items by their source type (e.g., 'rss', 'manual', 'twitter') "
            "or by a specific source/feed name. "
            "Use this to explore what content is available from different channels."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "description": "Filter by source_type field (e.g., 'rss', 'manual', 'twitter').",
                },
                "source_name": {
                    "type": "string",
                    "description": "Filter by source/feed name (partial match).",
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 10.",
                    "default": 10,
                },
            },
        }

    def execute(self, **params: object) -> ToolResult:
        source_type = str(params.get("source_type", "")) if params.get("source_type") else None
        source_name = str(params.get("source_name", "")) if params.get("source_name") else None
        n = int(params.get("n", 10))

        over_digested = self._db.get_over_digested_item_ids()
        with self._db._session() as session:
            query = select(Item).where(Item.read == False)
            if source_type:
                query = query.where(Item.source_type == source_type)
            if source_name:
                query = query.where(Item.source.contains(source_name))
            query = query.order_by(Item.timestamp.desc()).limit(n + len(over_digested))
            raw = list(session.exec(query).all())

        items = [item for item in raw if item.id not in over_digested][:n]

        msg_parts = []
        if source_type:
            msg_parts.append(f"source_type='{source_type}'")
        if source_name:
            msg_parts.append(f"source_name contains '{source_name}'")
        filter_desc = " and ".join(msg_parts) if msg_parts else "no filters"

        return ToolResult(
            items=[_item_to_dict(item) for item in items],
            message=f"Found {len(items)} items with {filter_desc}.",
        )


def register_local_tools(registry: "ToolRegistry", config: Config, db: Database) -> None:
    from patronus.tools import ToolRegistry as _TR
    registry.register(SearchSimilar(config, db))
    registry.register(SearchRecent(db))
    registry.register(SearchByTopic(config, db))
    registry.register(SearchBySource(db))
