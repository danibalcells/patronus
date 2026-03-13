from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyalex
from pyalex import Works

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


_ARXIV_URL_PREFIX = "https://arxiv.org/abs/"


def _normalize_to_lookup_key(raw: str) -> str | None:
    """Return a key suitable for Works()[key], or None if the format is unrecognised.

    Accepted inputs:
      - OpenAlex ID:   W2741809807  or  https://openalex.org/W2741809807
      - Bare DOI:      10.48550/arxiv.1706.03762
      - DOI URL:       https://doi.org/10.48550/arxiv.1706.03762
      - arXiv URL:     https://arxiv.org/abs/1706.03762  (converted to DOI)
    """
    if raw.startswith("W") and raw[1:].isdigit():
        return raw
    if raw.startswith("https://openalex.org/W"):
        return raw.split("/")[-1]
    if raw.startswith("10."):
        return f"https://doi.org/{raw}"
    if raw.startswith("https://doi.org/"):
        return raw
    if raw.startswith(_ARXIV_URL_PREFIX):
        arxiv_id = raw[len(_ARXIV_URL_PREFIX):]
        return f"https://doi.org/10.48550/arxiv.{arxiv_id}"
    return None


def _resolve_openalex_id(raw: str) -> str | None:
    """Return an OpenAlex Work ID (W...) for a recognised DOI or ID input.

    The `cites` filter only accepts OpenAlex IDs, so DOIs must be resolved
    via a single-entity lookup first. Returns None if the format is unrecognised.
    """
    lookup_key = _normalize_to_lookup_key(raw)
    if lookup_key is None:
        return None
    if lookup_key.startswith("W") and lookup_key[1:].isdigit():
        return lookup_key
    try:
        work = Works()[lookup_key]
        openalex_url = work.get("id", "")
        return openalex_url.split("/")[-1]  # e.g. "W2626778328"
    except Exception:
        logger.warning("Could not resolve %r to an OpenAlex ID", raw)
        return None


_ID_FORMAT_HELP = (
    "Accepted formats: OpenAlex ID ('W2741809807'), "
    "bare DOI ('10.48550/arxiv.1706.03762'), "
    "DOI URL ('https://doi.org/10.48550/arxiv.1706.03762'), "
    "or arXiv URL ('https://arxiv.org/abs/1706.03762'). "
    "Arbitrary web URLs (e.g. blog posts) are not supported — only works indexed by OpenAlex."
)


def _unrecognised_id_message(raw: str) -> str:
    return (
        f"Could not recognise '{raw}' as a valid work identifier. "
        + _ID_FORMAT_HELP
    )


def _build_item_dict(work: dict) -> dict:
    url = _canonical_url(work)
    return {
        "id": url,
        "title": (work.get("title") or "").strip(),
        "url": url,
        "author": _parse_authors(work),
        "source": "openalex",
        "item_type": "paper",
        "timestamp": work.get("publication_date") or "",
        "citation_count": work.get("cited_by_count") or 0,
        "snippet": ((work["abstract"] or "") if work.get("abstract_inverted_index") else "")[:ITEM_SNIPPET_MAX_CHARS],
        "topics": _parse_topics(work),
    }


def _filter_and_report(
    items: list[dict],
    recently_digested: set[str],
    query_label: str,
) -> tuple[list[dict], str]:
    """Remove already-featured papers and build a transparent skip notice."""
    filtered: list[dict] = []
    skipped: list[str] = []
    for item in items:
        url = item.get("url") or item.get("id", "")
        if url and url in recently_digested:
            title = item.get("title") or url
            skipped.append(f'"{title}" ({url})')
        else:
            filtered.append(item)
    skip_note = ""
    if skipped:
        skip_note = (
            f" {len(skipped)} result(s) were skipped — already featured in a recent digest: "
            + "; ".join(skipped)
            + ". Consider adjusting your query to find different work."
        )
    return filtered, skip_note


class SearchOpenAlex(Tool):
    def __init__(self, config: "Config", db: "Database | None" = None) -> None:
        self._config = config
        self._db = db

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

        recently_digested = self._db.get_recently_digested_item_ids() if self._db else set()
        raw = [_build_item_dict(work) for work in works if _canonical_url(work)]
        items, skip_note = _filter_and_report(raw, recently_digested, query)
        return ToolResult(
            items=items,
            message=f"Found {len(items)} new OpenAlex results for '{query}'.{skip_note}",
        )


class GetCitingPapers(Tool):
    def __init__(self, config: "Config", db: "Database | None" = None) -> None:
        self._config = config
        self._db = db

    @property
    def name(self) -> str:
        return "get_citing_papers"

    @property
    def description(self) -> str:
        return (
            "Find papers that cite a given work, identified by DOI or OpenAlex ID. "
            "Use this to discover recent research building on a landmark paper you already know about. "
            "Accepts DOI in any form (e.g. '10.48550/arxiv.2405.15943', 'https://doi.org/...') "
            "or an OpenAlex work ID (e.g. 'W2741809807')."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "doi_or_id": {
                    "type": "string",
                    "description": "DOI or OpenAlex work ID of the paper whose citing works you want. " + _ID_FORMAT_HELP,
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of citing papers to return. Defaults to 10.",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["citations", "recency"],
                    "description": (
                        "'recency' (default) returns the most recently published citing papers first. "
                        "'citations' returns the most-cited citing papers first."
                    ),
                    "default": "recency",
                },
                "from_publication_year": {
                    "type": "integer",
                    "description": "If set, restrict citing papers to those published on or after this year.",
                },
            },
            "required": ["doi_or_id"],
        }

    def execute(self, **params: object) -> ToolResult:
        raw_id = str(params.get("doi_or_id", "")).strip()
        n = int(params.get("n", 10))
        sort_by = str(params.get("sort_by", "recency"))
        from_year_raw = params.get("from_publication_year")
        from_year = int(from_year_raw) if from_year_raw is not None else None

        if not raw_id:
            return ToolResult(message="doi_or_id is required.")

        pyalex.config.api_key = self._config.openalex_api_key
        work_id = _resolve_openalex_id(raw_id)
        if work_id is None:
            return ToolResult(message=_unrecognised_id_message(raw_id))

        try:
            q = Works().filter(cites=work_id)
            if sort_by == "citations":
                q = q.sort(cited_by_count="desc")
            else:
                q = q.sort(publication_date="desc")
            if from_year is not None:
                q = q.filter(from_publication_date=f"{from_year}-01-01")
            works = q.get(per_page=n)
        except Exception:
            logger.exception("OpenAlex citing-papers request failed for %r", work_id)
            return ToolResult(message=f"OpenAlex request failed for '{raw_id}'.")

        if not works:
            return ToolResult(message=f"No citing papers found for '{raw_id}'.")

        recently_digested = self._db.get_recently_digested_item_ids() if self._db else set()
        raw = [_build_item_dict(work) for work in works if _canonical_url(work)]
        items, skip_note = _filter_and_report(raw, recently_digested, raw_id)
        return ToolResult(
            items=items,
            message=f"Found {len(items)} papers citing '{raw_id}'.{skip_note}",
        )


class GetReferencedPapers(Tool):
    def __init__(self, config: "Config", db: "Database | None" = None) -> None:
        self._config = config
        self._db = db

    @property
    def name(self) -> str:
        return "get_referenced_papers"

    @property
    def description(self) -> str:
        return (
            "Fetch the papers cited by a given work — its bibliography. "
            "Use this to trace the intellectual lineage of a paper or find foundational "
            "works in a field. Accepts the same identifier formats as get_citing_papers. "
            "Results are sorted by citation count by default (most influential references first)."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "doi_or_id": {
                    "type": "string",
                    "description": "DOI or OpenAlex work ID of the paper whose references you want. " + _ID_FORMAT_HELP,
                },
                "n": {
                    "type": "integer",
                    "description": "Maximum number of referenced papers to return. Defaults to 10.",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["citations", "recency"],
                    "description": (
                        "'citations' (default) returns the most-cited references first — useful for "
                        "identifying landmark papers. "
                        "'recency' returns the most recently published references first."
                    ),
                    "default": "citations",
                },
            },
            "required": ["doi_or_id"],
        }

    def execute(self, **params: object) -> ToolResult:
        raw_id = str(params.get("doi_or_id", "")).strip()
        n = int(params.get("n", 10))
        sort_by = str(params.get("sort_by", "citations"))

        if not raw_id:
            return ToolResult(message="doi_or_id is required.")

        pyalex.config.api_key = self._config.openalex_api_key
        work_id = _resolve_openalex_id(raw_id)
        if work_id is None:
            return ToolResult(message=_unrecognised_id_message(raw_id))

        try:
            source = Works()[work_id]
            ref_ids = [r.split("/")[-1] for r in (source.get("referenced_works") or [])]
        except Exception:
            logger.exception("Failed to fetch work %r", work_id)
            return ToolResult(message=f"Failed to fetch work '{raw_id}'.")

        if not ref_ids:
            return ToolResult(message=f"No references found for '{raw_id}'.")

        # Batch-fetch up to 100 IDs with server-side sort, then trim to n.
        # The API supports up to 100 IDs in a single OR filter.
        chunk = ref_ids[:100]
        try:
            q = Works().filter(openalex="|".join(chunk))
            if sort_by == "citations":
                q = q.sort(cited_by_count="desc")
            else:
                q = q.sort(publication_date="desc")
            works = q.get(per_page=n)
        except Exception:
            logger.exception("Failed to fetch referenced works for %r", work_id)
            return ToolResult(message=f"Failed to fetch references for '{raw_id}'.")

        recently_digested = self._db.get_recently_digested_item_ids() if self._db else set()
        raw = [_build_item_dict(work) for work in works if _canonical_url(work)]
        items, skip_note = _filter_and_report(raw, recently_digested, raw_id)
        total_refs = len(ref_ids)
        return ToolResult(
            items=items,
            message=f"Found {len(items)} references for '{raw_id}' ({total_refs} total in bibliography).{skip_note}",
        )


def register_openalex_tools(
    registry: "ToolRegistry",
    config: "Config",
    db: "Database | None" = None,
) -> None:
    from patronus.tools import ToolRegistry as _TR  # noqa: F401
    if config.openalex_api_key:
        registry.register(SearchOpenAlex(config, db=db))
        registry.register(GetCitingPapers(config, db=db))
        registry.register(GetReferencedPapers(config, db=db))
    else:
        logger.warning("OPENALEX_API_KEY not set — OpenAlex tools disabled")
