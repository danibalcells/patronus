from __future__ import annotations

from patronus.tools.base import Tool, ToolResult


class SearchArxiv(Tool):
    @property
    def name(self) -> str:
        return "search_arxiv"

    @property
    def description(self) -> str:
        return (
            "Search the Arxiv API for academic papers matching a query. "
            "Returns paper titles, authors, abstracts, and links. "
            "Results are also ingested into the local database for future retrieval. "
            "NOTE: This tool is not yet implemented and will return empty results."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for Arxiv papers.",
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
        return ToolResult(
            items=[],
            message=f"Arxiv search for '{query}' is not yet implemented. Use local search tools instead.",
        )


def register_arxiv_tools(registry: "ToolRegistry") -> None:
    from patronus.tools import ToolRegistry as _TR
    registry.register(SearchArxiv())
