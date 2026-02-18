from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyalex
from pyalex import Works

from patronus.embed import embed_text
from patronus.tools.base import ITEM_SNIPPET_MAX_CHARS, Tool, ToolResult

if TYPE_CHECKING:
    from patronus.config import Config
    from patronus.db import Database

logger = logging.getLogger(__name__)

# Maps agent-friendly field names to pyalex filter kwargs.
# Subfield filters are more precise; field-level filters are broader (catch-all).
# Topic-level filters (topics.id) are used when no subfield matches well enough.
_FIELD_FILTERS: dict[str, dict] = {
    # Core interest areas
    "ai":                         {"topics": {"subfield": {"id": "1702"}}},  # Artificial Intelligence
    "computer_science":           {"topics": {"field":    {"id": "17"}}},    # All of Computer Science
    "philosophy":                 {"topics": {"subfield": {"id": "1211"}}},  # Philosophy
    "cognitive_science":          {"topics": {"subfield": {"id": "2805"}}},  # Cognitive Neuroscience
    "psychology":                 {"topics": {"subfield": {"id": "3205"}}},  # Experimental and Cognitive Psychology
    "linguistics":                {"topics": {"subfield": {"id": "1203"}}},  # Language and Linguistics (Humanities)
    "neuroscience":               {"topics": {"field":    {"id": "28"}}},    # All of Neuroscience
    "political_science":          {"topics": {"subfield": {"id": "3320"}}},  # Political Science and International Relations
    "sociology":                  {"topics": {"subfield": {"id": "3312"}}},  # Sociology and Political Science
    # Specifically requested
    "volcanology":                {"topics": {"subfield": {"id": "1907"}}},  # Geology (incl. volcanology)
    "sociolinguistics":           {"topics": {"subfield": {"id": "3310"}}},  # Linguistics and Language (Social Sciences)
    "psycholinguistics":          {"topics": {"id":       "T13701"}},        # Psycholinguistics and Behavioral Studies
    # Adjacent fields
    "history":                    {"topics": {"subfield": {"id": "1202"}}},  # History
    "history_of_science":         {"topics": {"subfield": {"id": "1207"}}},  # History and Philosophy of Science
    "cultural_studies":           {"topics": {"subfield": {"id": "3316"}}},  # Cultural Studies
    "anthropology":               {"topics": {"subfield": {"id": "3314"}}},  # Anthropology
    "economics":                  {"topics": {"subfield": {"id": "2002"}}},  # Economics and Econometrics
    "earth_science":              {"topics": {"field":    {"id": "19"}}},    # Earth and Planetary Sciences (broad)
    "mathematics":                {"topics": {"field":    {"id": "26"}}},    # Mathematics (all subfields)
    "human_computer_interaction": {"topics": {"subfield": {"id": "1709"}}},  # Human-Computer Interaction
    "communication":              {"topics": {"subfield": {"id": "3315"}}},  # Communication
    "decision_science":           {"topics": {"field":    {"id": "18"}}},    # Decision Sciences
}


def _canonical_url(work: dict) -> str:
    doi = work.get("doi")
    if doi:
        return doi
    primary = work.get("primary_location") or {}
    landing = primary.get("landing_page_url")
    if landing:
        return landing
    return work.get("id", "")


def _parse_authors(work: dict) -> str:
    names = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    return ", ".join(names)


def _parse_topics(work: dict) -> list[str]:
    return [t["display_name"] for t in work.get("topics", [])[:5] if t.get("display_name")]


class SearchOpenAlex(Tool):
    def __init__(self, config: Config, db: Database, *, embed: bool = False) -> None:
        self._config = config
        self._db = db
        self._embed = embed

    @property
    def name(self) -> str:
        return "search_openalex"

    @property
    def description(self) -> str:
        return (
            "Search OpenAlex for academic papers across all disciplines (~250M works). "
            "Broader coverage than Arxiv — includes philosophy, linguistics, cognitive science, "
            "social sciences, and all natural sciences. "
            "Returns titles, authors, abstracts, DOIs, citation counts, and topic tags. "
            "Results are ingested into the local database for future retrieval. "
            "Use the 'field' parameter to restrict results to a discipline."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Full-text search query.",
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 5.",
                    "default": 5,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "citations", "recency"],
                    "description": (
                        "'relevance' (default) ranks by relevance to the query. "
                        "'citations' returns the most-cited papers first. "
                        "'recency' returns the most recently published papers first."
                    ),
                    "default": "relevance",
                },
                "from_publication_year": {
                    "type": "integer",
                    "description": "If set, restrict results to papers published on or after this year.",
                },
                "field": {
                    "type": "string",
                    "enum": list(_FIELD_FILTERS.keys()),
                    "description": (
                        "Optional discipline filter. Restricts results to papers in a given field. "
                        "Core: 'ai' (Artificial Intelligence), 'computer_science' (all CS), "
                        "'philosophy', 'cognitive_science' (Cognitive Neuroscience), "
                        "'psychology' (Experimental & Cognitive), 'linguistics' (Language & Linguistics), "
                        "'neuroscience', 'political_science', 'sociology'. "
                        "Language: 'sociolinguistics', 'psycholinguistics'. "
                        "Humanities/Social: 'history', 'history_of_science', 'cultural_studies', "
                        "'anthropology', 'economics', 'communication', 'decision_science'. "
                        "Natural sciences: 'earth_science', 'volcanology' (Geology). "
                        "Other: 'mathematics', 'human_computer_interaction'. "
                        "Leave unset to search all fields."
                    ),
                },
            },
            "required": ["query"],
        }

    def execute(self, **params: object) -> ToolResult:
        query = str(params.get("query", "")).strip()
        n = int(params.get("n", 5))
        sort_by = str(params.get("sort_by", "relevance"))
        from_year_raw = params.get("from_publication_year")
        from_year = int(from_year_raw) if from_year_raw is not None else None
        field_raw = params.get("field")
        field = str(field_raw) if field_raw is not None else None

        if not query:
            return ToolResult(message="Query string is required.")

        pyalex.config.api_key = self._config.openalex_api_key

        try:
            q = Works().search(query)
            if sort_by == "citations":
                q = q.sort(cited_by_count="desc")
            elif sort_by == "recency":
                q = q.sort(publication_date="desc")
            if from_year is not None:
                q = q.filter(from_publication_date=f"{from_year}-01-01")
            if field is not None:
                field_filter = _FIELD_FILTERS.get(field)
                if field_filter:
                    q = q.filter(**field_filter)
                else:
                    logger.warning("Unknown field filter %r, ignoring", field)
            works = q.get(per_page=n)
        except Exception:
            logger.exception("OpenAlex API request failed for query %r", query)
            return ToolResult(message=f"OpenAlex API request failed for query '{query}'.")

        if not works:
            return ToolResult(message=f"No OpenAlex results found for '{query}'.")

        items: list[dict] = []
        ingested = 0

        for work in works:
            url = _canonical_url(work)
            if not url:
                continue

            title = (work.get("title") or "").strip()
            abstract = (work["abstract"] or "").strip() if work.get("abstract_inverted_index") else ""
            authors = _parse_authors(work)
            topics = _parse_topics(work)
            published = work.get("publication_date") or ""
            cited_by_count = work.get("cited_by_count") or 0

            existing = self._db.get_item_by_url(url)
            if existing is not None:
                logger.debug("OpenAlex paper already in DB, skipping ingest: %s", url)
                item_id = existing.id
            else:
                embedding = None
                if self._embed and abstract:
                    try:
                        embedding = embed_text(abstract, model=self._config.embedding.model)
                    except Exception:
                        logger.exception("Embedding failed for OpenAlex paper: %s", url)

                try:
                    item_id = self._db.add_item(
                        url=url,
                        source_type="openalex_search",
                        item_type="paper",
                        title=title,
                        author=authors,
                        text=abstract,
                        embedding=embedding,
                        timestamp=published,
                    )
                    ingested += 1
                    logger.info("Ingested OpenAlex paper: %s (id=%s)", url, item_id)
                except Exception:
                    logger.exception("Failed to ingest OpenAlex paper: %s", url)
                    item_id = ""

            items.append({
                "id": item_id,
                "title": title,
                "url": url,
                "author": authors,
                "source": "openalex",
                "item_type": "paper",
                "timestamp": published,
                "citation_count": cited_by_count,
                "snippet": abstract[:ITEM_SNIPPET_MAX_CHARS],
                "topics": topics,
            })

        return ToolResult(
            items=items,
            message=f"Found {len(items)} OpenAlex results for '{query}' ({ingested} newly ingested).",
        )


def register_openalex_tools(
    registry: "ToolRegistry",
    config: "Config",
    db: "Database",
    *,
    embed: bool = False,
) -> None:
    from patronus.tools import ToolRegistry as _TR  # noqa: F401
    if config.openalex_api_key:
        registry.register(SearchOpenAlex(config, db, embed=embed))
    else:
        logger.warning("OPENALEX_API_KEY not set — search_openalex tool disabled")
