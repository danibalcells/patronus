from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser

from patronus.tools.base import ITEM_SNIPPET_MAX_CHARS, Tool, ToolResult

logger = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
# One retry after a short wait covers "slightly too fast" transient 429s.
# Sustained IP bans from bursting requests won't be resolved by waiting longer
# within a single agent turn, so we fail fast rather than hanging.
_RATE_LIMIT_RETRY_DELAY = 5

_SORT_BY_MAP = {
    "relevance": "relevance",
    "recency": "submittedDate",
}


def _canonical_arxiv_url(raw_id: str) -> str:
    url = raw_id.replace("http://", "https://")
    url = re.sub(r"v\d+$", "", url)
    if "arxiv.org/abs/" not in url:
        arxiv_id = raw_id.split("/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        url = f"https://arxiv.org/abs/{arxiv_id}"
    return url


def _parse_authors(entry: object) -> str:
    authors = getattr(entry, "authors", None) or []
    names = []
    for a in authors:
        name = getattr(a, "name", None) or (a.get("name") if isinstance(a, dict) else None)
        if name:
            names.append(name)
    return ", ".join(names)


def _parse_categories(entry: object) -> list[str]:
    tags = getattr(entry, "tags", None) or []
    return [t.get("term", "") for t in tags if isinstance(t, dict) and t.get("term")]


def _build_search_query(query: str, category: str | None, days: int | None) -> str:
    # For multi-word queries, AND each term explicitly so they all appear in the
    # paper without requiring the exact phrase, while keeping the boolean chain intact.
    terms = query.split()
    if len(terms) > 1:
        query_part = "+AND+".join(f"all:{quote_plus(t)}" for t in terms)
    else:
        query_part = f"all:{quote_plus(query)}"
    parts = [query_part]
    if category:
        parts.append(f"cat:{quote_plus(category)}")
    if days is not None and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        date_from = cutoff.strftime("%Y%m%d0000")
        date_to = datetime.now(timezone.utc).strftime("%Y%m%d2359")
        parts.append(f"submittedDate:[{date_from}+TO+{date_to}]")
    return "+AND+".join(parts)


class SearchArxiv(Tool):
    @property
    def name(self) -> str:
        return "search_arxiv"

    @property
    def description(self) -> str:
        return (
            "Search the Arxiv API for academic papers matching a query. "
            "Returns paper titles, authors, abstracts, and links. "
            "Supports sorting by relevance or recency, filtering by category (e.g. cs.LG, cs.AI, stat.ML) "
            "and restricting to papers submitted within the last N days. "
            "Note: Arxiv does not expose citation counts or download statistics."
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
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "recency"],
                    "description": (
                        "Sort order for results. 'relevance' (default) uses Arxiv's relevance ranking. "
                        "'recency' returns the most recently submitted papers first."
                    ),
                    "default": "relevance",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional Arxiv category code to restrict results to, e.g. 'cs.LG', 'cs.AI', "
                        "'stat.ML', 'cs.CL', 'cs.CV'. Leave empty to search all categories."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "If set, restrict results to papers submitted within the last N days. "
                        "Useful for finding recent work. Leave unset to search all dates."
                    ),
                },
            },
            "required": ["query"],
        }

    def execute(self, **params: object) -> ToolResult:
        query = str(params.get("query", "")).strip()
        n = int(params.get("n", 5))
        sort_by_key = str(params.get("sort_by", "relevance"))
        category_raw = params.get("category")
        category = str(category_raw).strip() or None if isinstance(category_raw, str) else None
        days_raw = params.get("days")
        days = int(days_raw) if days_raw is not None else None

        if not query:
            return ToolResult(message="Query string is required.")

        sort_by = _SORT_BY_MAP.get(sort_by_key, "relevance")
        search_query = _build_search_query(query, category, days)

        api_url = (
            f"{_ARXIV_API_URL}"
            f"?search_query={search_query}"
            f"&start=0"
            f"&max_results={n}"
            f"&sortBy={sort_by}"
            f"&sortOrder=descending"
        )
        logger.info("Querying Arxiv API: %s", api_url)

        try:
            feed = feedparser.parse(api_url)
        except Exception:
            logger.exception("Arxiv API request failed for query %r", query)
            return ToolResult(message=f"Arxiv API request failed for query '{query}'.")

        if getattr(feed, "status", 200) == 429:
            logger.warning("Arxiv rate limit hit, retrying in %ds…", _RATE_LIMIT_RETRY_DELAY)
            time.sleep(_RATE_LIMIT_RETRY_DELAY)
            try:
                feed = feedparser.parse(api_url)
            except Exception:
                logger.exception("Arxiv API request failed on retry for query %r", query)
                return ToolResult(message=f"Arxiv API request failed for query '{query}'.")
            if getattr(feed, "status", 200) == 429:
                return ToolResult(
                    message=(
                        f"Arxiv rate limit exceeded for query '{query}'. "
                        "The API allows ~1 request per 3 seconds; wait a few minutes before retrying."
                    )
                )

        if not feed.entries:
            return ToolResult(message=f"No Arxiv results found for '{query}'.")

        items: list[dict] = []

        for entry in feed.entries:
            url = _canonical_arxiv_url(getattr(entry, "id", ""))
            if not url:
                continue

            title = " ".join((getattr(entry, "title", "") or "").split())
            abstract = " ".join((getattr(entry, "summary", "") or "").split())
            authors = _parse_authors(entry)
            categories = _parse_categories(entry)
            published = getattr(entry, "published", None) or ""
            journal_ref = getattr(entry, "arxiv_journal_ref", None) or ""

            item: dict = {
                "id": url,
                "title": title,
                "url": url,
                "author": authors,
                "source": "arxiv",
                "item_type": "paper",
                "timestamp": published,
                "snippet": abstract[:ITEM_SNIPPET_MAX_CHARS],
                "categories": categories,
            }
            if journal_ref:
                item["journal_ref"] = journal_ref
            items.append(item)

        return ToolResult(
            items=items,
            message=f"Found {len(items)} Arxiv results for '{query}'.",
        )


def register_arxiv_tools(registry: "ToolRegistry") -> None:
    from patronus.tools import ToolRegistry as _TR  # noqa: F401
    registry.register(SearchArxiv())
