from __future__ import annotations

import logging
from pathlib import Path

from patronus.config import Config
from patronus.notion_mirror import NotionMirror
from patronus.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class SearchNotion(Tool):
    def __init__(self, config: Config) -> None:
        self._config = config
        assert config.notion is not None
        self._mirror_path = config.notion.mirror_path
        self._db_name_by_id: dict[str, str] = {
            v: k for k, v in config.notion.database_ids.items()
        }

    @property
    def name(self) -> str:
        return "search_notion"

    @property
    def description(self) -> str:
        return (
            "Search your personal Notion notes, journal entries, book reviews, and work diary "
            "using full-text search (BM25). Use this to surface connections between today's content "
            "and your own past thinking. Returns snippets with links to the original pages. "
            "Only include results in the digest if they're genuinely relevant — not just topically adjacent."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant personal notes, journal entries, or reviews.",
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def execute(self, **params: object) -> ToolResult:
        query = str(params.get("query", ""))
        n = int(params.get("n", 5))

        if not query:
            return ToolResult(message="Query string is required.")

        mirror_path = Path(self._mirror_path)
        if not mirror_path.exists():
            return ToolResult(message="Notion mirror not found. Run sync_notion_mirror.py first.")

        try:
            with NotionMirror(str(mirror_path)) as mirror:
                pages = mirror.search(query, limit=n)
        except Exception:
            logger.exception("Notion mirror search failed for query %r", query)
            return ToolResult(message="Notion mirror search failed with an internal error.")

        if not pages:
            return ToolResult(message=f"No personal notes found matching '{query}'.")

        items = []
        for page in pages:
            db_name = self._db_name_by_id.get(page.source_db, page.source_db)
            items.append({
                "id": f"notion:{page.id}",
                "title": page.title,
                "url": page.url,
                "source": f"your notes / {db_name}",
                "timestamp": page.last_edited_at,
                "snippet": page.content_snippet,
            })

        return ToolResult(
            items=items,
            message=f"Found {len(items)} personal notes matching '{query}'.",
        )


def register_notion_tools(registry: "ToolRegistry", config: Config) -> None:
    from patronus.tools import ToolRegistry as _TR  # noqa: F401
    if config.notion and config.notion.mirror_path:
        registry.register(SearchNotion(config))
