from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from notion_client import Client as NotionClient
from notion_client.errors import HTTPResponseError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    RetryCallState,
)

from patronus.config import Config
from patronus.db import Database
from patronus.llm import complete
from patronus.observability import iteration_span, llm_generation

logger = logging.getLogger(__name__)

_MAX_RETRY_ATTEMPTS = 5
_RETRY_AFTER_DEFAULT = 30


def _is_rate_limited(exc: BaseException) -> bool:
    return isinstance(exc, HTTPResponseError) and exc.status == 429


def _before_retry_log(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    if isinstance(exc, HTTPResponseError) and exc.status == 429:
        retry_after = exc.headers.get("Retry-After")
        logger.warning(
            "Notion rate limit hit (attempt %d/%d). Waiting %.1fs (Retry-After: %s)",
            retry_state.attempt_number,
            _MAX_RETRY_ATTEMPTS,
            wait,
            retry_after or "not set",
        )
    else:
        logger.warning(
            "Retrying Notion API call after error (attempt %d/%d): %s",
            retry_state.attempt_number,
            _MAX_RETRY_ATTEMPTS,
            exc,
        )


def _retry_after_wait(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, HTTPResponseError) and exc.status == 429:
        raw = exc.headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    return 0.0


def _notion_wait(retry_state: RetryCallState) -> float:
    explicit = _retry_after_wait(retry_state)
    if explicit > 0:
        return explicit
    return wait_exponential(multiplier=1, min=2, max=60)(retry_state)


notion_retry = retry(
    retry=retry_if_exception(_is_rate_limited),
    wait=_notion_wait,
    stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
    before_sleep=_before_retry_log,
    reraise=True,
)

_SUMMARY_MAX_TOKENS = 5000

_SUMMARY_SYSTEM_PROMPT = """\
You are analyzing a user's recent Notion workspace to build a personalization context \
for a reading and research recommendation agent. The agent will use your summary to \
decide what content to surface — so it needs to understand what this person is thinking \
about, working on, struggling with, and curious about RIGHT NOW.

## Source databases

The entries are tagged with their source database. Understanding what each one \
represents is critical for interpreting them correctly.

First-person writing (the user's own words and thinking):
- journal — Daily personal reflections. Mood, life events, what's on the mind. May \
contain poems, fragments, emotional processing.
- work_diary — Day-by-day log of professional work. What was built, bugs hit, design \
decisions, where the user got stuck, small breakthroughs. Rich in specific technical detail.
- notes — The user's own ideas, concepts, and frameworks. Original thinking.
- big_questions — Open questions the user is actively grappling with.
- reviews — Synthesis and evaluation of things consumed (books, papers, talks, courses). \
The user's own analysis and takeaways.

Third-party content (highlighted by the user, written by others):
- library — Readwise highlights synced to Notion. Passages the user found noteworthy \
from books, articles, and papers. Highlights don't imply endorsement — the user often \
highlights to summarize or extract key points from an argument, even when they disagree. \
Treat these as "things the user is engaging with intellectually," not "things the user \
believes."

## Important: created vs last edited

Entries are filtered by last_edited_time, not created_time. This means the input may \
include old notes or journal entries that the user has revisited or updated recently. \
A note created months ago but edited this week likely reflects a CURRENT concern — the \
user went back to it for a reason. Pay attention to the gap between created and \
last_edited dates as a signal of revisitation.

## What to extract

Focus on SPECIFICITY and GRANULARITY over breadth. The agent needs the precise shape \
of this person's current intellectual life, not a high-level category list.

Active questions and problems — What specific intellectual and technical questions is \
the user wrestling with? Not "interested in AI safety" but "trying to figure out whether \
mechanistic interpretability methods can detect deceptive alignment in practice." Include \
sticking points from the work diary, open questions from notes and big_questions, and \
recurring themes from the journal.

Current projects and their state — What is the user actively building or writing? Be \
specific about where they are: what's working, what's blocked, what decisions are open. \
The work diary is especially rich here.

Research and ideas being engaged with — What technical research, arguments, and ideas \
from third-party content (library highlights, reviews) is the user engaging with? This \
should be DISTINCT from the user's own projects. Name authors when available — they help \
the recommendation agent find related work. But lead with the intellectual substance: \
"engaging with Author X's argument that Y because Z" rather than just "reading Paper W." \
Include the user's stance where visible: are they building on it, questioning it, or \
just mapping the landscape?

Life context and patterns — Look for aggregate signals that individual entries might not \
state explicitly. Multiple job application entries → actively job searching. Repeated \
mentions of a city → possibly relocating. Frequent references to a person → important \
relationship. These contextual signals dramatically affect what content is relevant.

Peripheral interests and dormant curiosities — Things that appear in the user's writing \
but are NOT connected to their main lines of work: a book about consciousness highlighted \
last week, a poem about language, a library highlight from a neuroscience paper, a \
journal mention of an unrelated topic. List these explicitly and specifically — they are \
the raw material for serendipitous recommendations. The point is to surface things that \
a query-driven agent would never think to search for. Even one-off mentions count here.

Recurring emotional undertones and themes — What keeps coming back in the journal? What \
is the user frustrated about, excited about, uncertain about? What are poems about? What \
mood or preoccupation runs through the recent entries? These inform editorial tone and \
reveal what might resonate beyond the intellectual content.

Connections across domains — Where do the user's different interests meet or create \
tension? These intersection points are the highest-signal for recommendations.

## Guidelines
- Write in second person ("you're working on...", "you've been stuck on...").
- Organize by theme, not by source database. But let the source inform your \
interpretation: a work diary sticking point is different from a journal reflection.
- Be specific and concrete. Name specific problems, ideas, and questions. Avoid generic \
category labels that could apply to anyone in the field.
- SYNTHESIZE across entries — find the pattern, don't restate each entry.
- Detect aggregate patterns: if there are several entries that each mention a different \
job application, say "you're actively job searching" rather than listing each one.
- Focus on what the user is DOING and THINKING, not the tools or models they mention \
in passing. Tools are incidental; the intellectual work is what matters.
- Prioritize recent and recurring over one-off mentions.
- If you notice something surprising or that doesn't fit the obvious patterns — include \
it. Outliers are often the most valuable signal for recommendations.

## Priority tiers

Structure your output so that each theme or interest is tagged with one of these tiers:

**PRIMARY** — current active technical work and main intellectual preoccupations. This is \
what the reader is doing right now: active projects, open research questions being actively \
pursued, core ideas under development. Use this tier for at most 2-3 things.

**SECONDARY** — active interests that inform the reader's thinking but aren't the primary \
work. These are things the reader cares about and engages with regularly, but they are not \
the central thrust of current work (e.g., job search dynamics, AI policy, industry shifts \
adjacent to their field).

**PERIPHERAL** — dormant curiosities, creative practices, and side interests. These appear \
in the reader's writing but are not connected to their main lines of work. Only surface when \
there is a specific, concrete connection to something in the feed. Do not search proactively \
for these.

After the main context, add this note verbatim:
"NOTE FOR DOWNSTREAM AGENTS: Allocate most attention to PRIMARY areas. Use SECONDARY areas \
to contextualize news and industry developments. PERIPHERAL interests should not drive \
research or long_form_pick selections — only surface them when there is a direct, concrete \
hook in the content."\
"""


@dataclass
class NotionEntry:
    title: str
    source_db: str
    content: str
    last_edited: str
    created: str = ""
    author: str = ""


def fetch_notion_entries(
    client: NotionClient,
    config: Config,
    lookback_days: int,
    data_source_id_cache: dict[str, str] | None = None,
) -> list[NotionEntry]:
    if config.notion is None:
        return []
    if data_source_id_cache is None:
        data_source_id_cache = {}
    entries: list[NotionEntry] = []
    for db_name, db_id in config.notion.database_ids.items():
        try:
            db_entries = query_notion_database(
                client, config, db_id, db_name, lookback_days, data_source_id_cache
            )
            entries.extend(db_entries)
        except Exception:
            logger.warning("Failed to query Notion database %s", db_name, exc_info=True)
    return entries


def resolve_data_source_id(
    client: NotionClient,
    db_id: str,
    cache: dict[str, str],
) -> str:
    if db_id not in cache:
        response = notion_retry(client.databases.retrieve)(database_id=db_id)
        data_sources = response.get("data_sources", [])
        if not data_sources:
            raise ValueError(f"No data sources found for database {db_id}")
        cache[db_id] = data_sources[0]["id"]
    return cache[db_id]


def query_notion_database(
    client: NotionClient,
    config: Config,
    db_id: str,
    db_name: str,
    lookback_days: int,
    data_source_id_cache: dict[str, str] | None = None,
) -> list[NotionEntry]:
    if data_source_id_cache is None:
        data_source_id_cache = {}
    ds_id = resolve_data_source_id(client, db_id, data_source_id_cache)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    results: list[NotionEntry] = []
    cursor: str | None = None

    while True:
        kwargs: dict = {
            "filter": {
                "timestamp": "last_edited_time",
                "last_edited_time": {"after": cutoff},
            },
            "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        response = notion_retry(client.data_sources.query)(data_source_id=ds_id, **kwargs)

        for page in response["results"]:
            title = _extract_page_title(page)
            try:
                content = extract_page_content(client, page["id"])
            except Exception:
                logger.warning("Failed to extract content from page %s", page["id"], exc_info=True)
                content = ""

            notion_cfg = config.notion
            assert notion_cfg is not None
            truncated = content[-notion_cfg.max_chars_per_entry:]
            author = _extract_page_author(page) if db_name == "library" else ""
            results.append(NotionEntry(
                title=title,
                source_db=db_name,
                content=truncated,
                last_edited=page.get("last_edited_time", ""),
                created=page.get("created_time", ""),
                author=author,
            ))

        if not response.get("has_more", False):
            break
        cursor = response.get("next_cursor")

    return results


def extract_page_content(client: NotionClient, page_id: str) -> str:
    blocks = _fetch_all_blocks(client, page_id)
    return _blocks_to_text(client, blocks)


class NotionSource:
    def __init__(
        self,
        notion_client: NotionClient | None = None,
        db: Database | None = None,
    ) -> None:
        self._client = notion_client
        self._db = db
        self._data_source_ids: dict[str, str] = {}

    def _get_client(self, config: Config) -> NotionClient:
        if self._client is None:
            self._client = NotionClient(auth=config.notion_token)
        return self._client

    def _get_cached_context(self, config: Config, allow_stale: bool = False) -> str:
        if self._db is None or config.notion is None:
            return ""

        snapshot = self._db.get_latest_context_snapshot("notion")
        if snapshot is None or not snapshot.content:
            return ""

        if allow_stale:
            return snapshot.content

        try:
            generated_at = datetime.fromisoformat(snapshot.generated_at.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
            if age_hours <= config.notion.cache_ttl_hours:
                return snapshot.content
        except (ValueError, AttributeError):
            logger.warning("Invalid timestamp in cached context snapshot", exc_info=True)

        return ""

    def _get_entries_from_mirror(self, config: Config) -> list[NotionEntry] | None:
        if config.notion is None or not config.notion.mirror_path:
            logger.debug("No mirror_path configured, falling back to Notion API")
            return None
        from patronus.notion_mirror import NotionMirror, trim_content
        try:
            mirror = NotionMirror(config.notion.mirror_path)
        except Exception:
            logger.warning("Failed to open Notion mirror at %s, falling back to API", config.notion.mirror_path, exc_info=True)
            return None

        with mirror:
            page_count = mirror.page_count()
            latest_sync = mirror.latest_synced_at()
            logger.info(
                "Notion mirror: %s (%d pages total, last synced: %s)",
                config.notion.mirror_path,
                page_count,
                latest_sync or "never",
            )

            if mirror.is_stale(config.notion.cache_ttl_hours):
                logger.warning(
                    "Notion mirror is stale (last synced: %s, threshold: %dh) — "
                    "run scripts/sync_notion_mirror.py to refresh; using stale data",
                    latest_sync or "never",
                    config.notion.cache_ttl_hours,
                )

            now = datetime.now(timezone.utc)
            notion_cfg = config.notion

            first_person_dbs = [db for db in notion_cfg.database_ids if db != "library"]

            since_fp = now - timedelta(days=notion_cfg.lookback_days)
            fp_pages = mirror.get_recent(source_dbs=first_person_dbs, since=since_fp)
            logger.info(
                "Mirror query first-person (lookback %dd, dbs=%s): %d pages",
                notion_cfg.lookback_days,
                first_person_dbs,
                len(fp_pages),
            )

            if len(fp_pages) < notion_cfg.min_entries_threshold:
                since_fallback = now - timedelta(days=notion_cfg.fallback_lookback_days)
                fp_pages = mirror.get_recent(source_dbs=first_person_dbs, since=since_fallback)
                logger.info(
                    "Mirror query first-person (fallback lookback %dd): %d pages",
                    notion_cfg.fallback_lookback_days,
                    len(fp_pages),
                )

            since_lib = now - timedelta(days=notion_cfg.library_lookback_days)
            lib_pages = mirror.get_recent(source_dbs=["library"], since=since_lib)
            logger.info(
                "Mirror query library (lookback %dd): %d pages",
                notion_cfg.library_lookback_days,
                len(lib_pages),
            )

            all_pages = sorted(fp_pages + lib_pages, key=lambda p: p.last_edited_at, reverse=True)

            if len(all_pages) < notion_cfg.min_entries_threshold:
                logger.warning(
                    "Mirror has too few entries (%d < threshold %d) — falling back to Notion API",
                    len(all_pages),
                    notion_cfg.min_entries_threshold,
                )
                return []

            snippet_cfg = notion_cfg.snippet
            exclude_prefixes = notion_cfg.exclude_library_prefixes
            entries: list[NotionEntry] = []
            skipped = 0

            for p in all_pages:
                if p.source_db == "library" and exclude_prefixes:
                    if any(p.title.startswith(prefix) for prefix in exclude_prefixes):
                        skipped += 1
                        continue

                if p.source_db == "library":
                    trimmed = trim_content(p.content, snippet_cfg.library_head_chars, snippet_cfg.library_tail_chars)
                else:
                    trimmed = trim_content(p.content, snippet_cfg.first_person_head_chars, snippet_cfg.first_person_tail_chars)

                entries.append(NotionEntry(
                    title=p.title,
                    source_db=p.source_db,
                    content=trimmed,
                    last_edited=p.last_edited_at,
                    created=p.created_at,
                ))

            logger.info(
                "Using mirror as context source (%d entries, %d excluded by prefix filter)",
                len(entries),
                skipped,
            )
            return entries

    def get_context(self, config: Config, force_refresh: bool = False) -> str:
        if config.notion is None:
            return ""

        if not force_refresh:
            cached = self._get_cached_context(config)
            if cached:
                logger.info("Using cached Notion context (%d chars)", len(cached))
                return cached

        with iteration_span("notion-context-build", input={"force_refresh": force_refresh}) as span:
            mirror_entries = self._get_entries_from_mirror(config)
            if mirror_entries is not None:
                entries = mirror_entries
                source_label = "mirror"
            else:
                logger.info("Fetching Notion context via live API")
                entries = fetch_notion_entries(
                    self._get_client(config), config, config.notion.lookback_days, self._data_source_ids
                )
                if len(entries) < config.notion.min_entries_threshold:
                    entries = fetch_notion_entries(
                        self._get_client(config), config, config.notion.fallback_lookback_days, self._data_source_ids
                    )
                source_label = "API"

            if len(entries) < config.notion.min_entries_threshold:
                logger.warning("Insufficient Notion entries (%d < %d), returning empty context",
                             len(entries), config.notion.min_entries_threshold)
                span.update(output={"result": "insufficient_entries", "count": len(entries)})
                return ""

            logger.info("Building Notion context from %s (%d entries)", source_label, len(entries))
            span.update(input={"force_refresh": force_refresh, "source": source_label, "entry_count": len(entries)})

            try:
                summary = self._summarize_entries(entries, config)
            except Exception:
                logger.warning("Failed to generate fresh Notion summary", exc_info=True)
                stale_cached = self._get_cached_context(config, allow_stale=True)
                if stale_cached:
                    logger.info("Using stale cached Notion context as fallback (%d chars)", len(stale_cached))
                    span.update(output={"result": "stale_cache_fallback", "chars": len(stale_cached)})
                    return stale_cached
                span.update(output={"result": "error_no_fallback"})
                return ""

            if self._db is not None and summary:
                self._db.save_context_snapshot("notion", summary)
                logger.info("Saved fresh Notion context to cache (%d chars)", len(summary))

            span.update(output={"result": "fresh_summary", "chars": len(summary)})
            return summary

    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None:
        return None

    def _fetch_all_entries(self, config: Config, lookback_days: int) -> list[NotionEntry]:
        return fetch_notion_entries(
            self._get_client(config), config, lookback_days, self._data_source_ids
        )

    def _summarize_entries(self, entries: list[NotionEntry], config: Config) -> str:
        prompt_parts: list[str] = []
        for entry in entries:
            header = f"## [{entry.source_db}] {entry.title}"
            if entry.author:
                header += f" (by {entry.author})"
            meta = f"Created: {entry.created} | Last edited: {entry.last_edited}"
            prompt_parts.append(
                f"{header}\n"
                f"{meta}\n\n"
                f"{entry.content}"
            )

        max_input_chars = config.notion.max_input_chars if config.notion else 250_000
        full_text = "\n\n---\n\n".join(prompt_parts)
        full_text = full_text[:max_input_chars]

        model = config.agent.notion_context_model if config.agent else ""
        logger.info(
            "Summarizer input: %d entries, %d chars (%d chars after truncation), model: %s",
            len(entries), sum(len(p) for p in prompt_parts), len(full_text), model,
        )

        with llm_generation(
            "notion-context-summarize",
            model,
            {"system": _SUMMARY_SYSTEM_PROMPT, "user": full_text},
        ) as obs:
            result = complete(
                model,
                system=_SUMMARY_SYSTEM_PROMPT,
                user_message=full_text,
                max_tokens=_SUMMARY_MAX_TOKENS,
            )
            obs.update(output=result[:500] if result else "")
            return result


def _fetch_all_blocks(client: NotionClient, block_id: str) -> list[dict]:
    all_blocks: list[dict] = []
    cursor: str | None = None

    while True:
        kwargs: dict = {"block_id": block_id}
        if cursor:
            kwargs["start_cursor"] = cursor

        response = notion_retry(client.blocks.children.list)(**kwargs)
        all_blocks.extend(response["results"])

        if not response.get("has_more", False):
            break
        cursor = response.get("next_cursor")

    return all_blocks


def _blocks_to_text(client: NotionClient, blocks: list[dict], depth: int = 0) -> str:
    lines: list[str] = []

    for block in blocks:
        block_type = block.get("type", "")
        data = block.get(block_type, {})

        if block_type == "synced_block":
            synced_from = data.get("synced_from")
            if synced_from and synced_from.get("block_id"):
                try:
                    child_blocks = _fetch_all_blocks(client, synced_from["block_id"])
                    child_text = _blocks_to_text(client, child_blocks, depth)
                    if child_text:
                        lines.append(child_text)
                except Exception:
                    logger.warning(
                        "Failed to fetch synced block %s", synced_from["block_id"], exc_info=True
                    )
                continue

        line = _block_to_line(block_type, data, depth)
        if line is not None:
            lines.append(line)
        elif block_type not in ("column_list", "column", "table", "synced_block", ""):
            logger.debug("Skipping block type %r (no text extraction)", block_type)

        if block.get("has_children"):
            try:
                child_blocks = _fetch_all_blocks(client, block["id"])
                child_text = _blocks_to_text(client, child_blocks, depth + 1)
                if child_text:
                    lines.append(child_text)
            except Exception as exc:
                if block_type == "unsupported":
                    logger.debug(
                        "Cannot fetch children of unsupported block %s: %s",
                        block.get("id", "?"),
                        exc,
                    )
                else:
                    logger.warning(
                        "Failed to fetch children of block %s (type=%r)",
                        block.get("id", "?"),
                        block_type,
                        exc_info=True,
                    )

    return "\n".join(lines)


def _block_to_line(block_type: str, data: dict, depth: int) -> Optional[str]:
    indent = "  " * depth

    if block_type == "paragraph":
        text = _rich_text_to_str(data.get("rich_text", []))
        return f"{indent}{text}" if text else None

    if block_type == "heading_1":
        return f"{indent}# {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "heading_2":
        return f"{indent}## {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "heading_3":
        return f"{indent}### {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "bulleted_list_item":
        return f"{indent}- {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "numbered_list_item":
        return f"{indent}1. {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "to_do":
        checked = data.get("checked", False)
        marker = "[x]" if checked else "[ ]"
        return f"{indent}{marker} {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "toggle":
        return f"{indent}▸ {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "quote":
        return f"{indent}> {_rich_text_to_str(data.get('rich_text', []))}"

    if block_type == "callout":
        icon = data.get("icon", {})
        emoji = icon.get("emoji", "") if icon else ""
        text = _rich_text_to_str(data.get("rich_text", []))
        prefix = f"{emoji} " if emoji else ""
        return f"{indent}{prefix}{text}" if text else None

    if block_type == "code":
        text = _rich_text_to_str(data.get("rich_text", []))
        lang = data.get("language", "")
        return f"{indent}```{lang}\n{text}\n{indent}```"

    if block_type == "equation":
        expr = data.get("expression", "")
        return f"{indent}{expr}" if expr else None

    if block_type == "divider":
        return f"{indent}---"

    if block_type == "table_row":
        cells = data.get("cells", [])
        cell_texts = [_rich_text_to_str(cell) for cell in cells]
        return f"{indent}{' | '.join(cell_texts)}"

    if block_type == "bookmark":
        url = data.get("url", "")
        caption = _rich_text_to_str(data.get("caption", []))
        if caption:
            return f"{indent}{caption}: {url}"
        return f"{indent}{url}" if url else None

    if block_type == "link_preview":
        url = data.get("url", "")
        return f"{indent}{url}" if url else None

    if block_type in ("column_list", "column", "table", "synced_block"):
        return None

    return None


def _rich_text_to_str(rich_text: list[dict]) -> str:
    return "".join(item.get("plain_text", "") for item in rich_text)


def _extract_page_title(page: dict) -> str:
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_items = prop.get("title", [])
            return "".join(item.get("plain_text", "") for item in title_items)
    return "Untitled"


def _extract_page_author(page: dict) -> str:
    props = page.get("properties", {})
    for name, prop in props.items():
        if name.lower() == "author":
            if prop.get("type") == "rich_text":
                return _rich_text_to_str(prop.get("rich_text", []))
            if prop.get("type") == "select":
                sel = prop.get("select")
                return sel.get("name", "") if sel else ""
    return ""
