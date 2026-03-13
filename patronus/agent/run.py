from __future__ import annotations

import logging
from datetime import datetime, timezone

from patronus.agent._compose import compose_digest
from patronus.agent._inventory import build_inventory
from patronus.agent._prompts import ANGLES_SYSTEM_PROMPT
from patronus.agent._steps import filter_news, identify_angles, pull_threads, scout_research, summarize_chatter
from patronus.config import Config
from patronus.context import Context
from patronus.db import Database, Item
from patronus.digest import Digest
from patronus.llm import estimate_tokens, get_context_limit
from patronus.observability import agent_run, iteration_span
from patronus.tools import ToolRegistry

logger = logging.getLogger(__name__)

_SHORT_SNIPPET_CHARS = 150
_ANGLES_MAX_TOKENS = 2048
_RECENTLY_FEATURED_LOOKBACK_DAYS = 14


def _build_recently_featured_block(db: Database | None) -> str:
    """Build a human-readable list of papers featured in research sections recently.

    Returned as a bullet-list string ready to embed in the scout_research prompt.
    Returns an empty string when the DB is unavailable or no papers are found.
    """
    if db is None:
        return ""
    try:
        papers = db.get_recent_research_papers(lookback_days=_RECENTLY_FEATURED_LOOKBACK_DAYS)
    except Exception:
        logger.warning("Could not fetch recently featured papers for research scout", exc_info=True)
        return ""
    if not papers:
        return ""
    lines = [
        "The following papers have appeared in recent digests. Do not include them again. "
        "If a search returns one of these, adjust your query to find different work instead.\n"
    ]
    for p in papers:
        title = p.get("title") or ""
        item_id = p.get("item_id", "")
        date = p.get("digest_date", "")
        if title:
            lines.append(f"- \"{title}\" — {item_id} (featured {date})")
        else:
            lines.append(f"- {item_id} (featured {date})")
    return "\n".join(lines)


def _angles_prompt_tokens(inventory: str, reader_context: str) -> int:
    return estimate_tokens(ANGLES_SYSTEM_PROMPT + inventory + reader_context)


def _build_angles_inventory(
    config: Config,
    db: Database | None,
    full_inventory: str,
    reader_context: str,
    short_id_map: dict[str, str],
    context_limit: int,
) -> str:
    budget = context_limit - _ANGLES_MAX_TOKENS
    full_tokens = _angles_prompt_tokens(full_inventory, reader_context)
    logger.info(
        "Angles prompt estimate: ~%d tokens (budget: %d, limit: %d)",
        full_tokens, budget, context_limit,
    )
    if full_tokens <= budget:
        logger.info("Angles prompt fits — using full inventory (%d-char snippets)", 300)
        return full_inventory

    logger.warning(
        "Angles prompt too large (~%d tokens > %d budget) — rebuilding with %d-char snippets",
        full_tokens, budget, _SHORT_SNIPPET_CHARS,
    )
    trimmed, _, _ = build_inventory(config, db, snippet_chars=_SHORT_SNIPPET_CHARS)
    trimmed_tokens = _angles_prompt_tokens(trimmed, reader_context)
    logger.info("Trimmed inventory prompt estimate: ~%d tokens", trimmed_tokens)
    if trimmed_tokens <= budget:
        logger.info("Trimmed inventory fits — proceeding with %d-char snippets", _SHORT_SNIPPET_CHARS)
        return trimmed

    agent_cfg = config.agent
    current_lookback = agent_cfg.inventory_lookback_days if agent_cfg else 2
    if current_lookback > 1:
        logger.warning(
            "Trimmed inventory still too large (~%d tokens > %d budget) — rebuilding with "
            "1-day lookback and %d-char snippets",
            trimmed_tokens, budget, _SHORT_SNIPPET_CHARS,
        )
        reduced, _, _ = build_inventory(config, db, snippet_chars=_SHORT_SNIPPET_CHARS, lookback_days_override=1)
        reduced_tokens = _angles_prompt_tokens(reduced, reader_context)
        logger.info(
            "Reduced inventory prompt estimate: ~%d tokens (1-day lookback, %d-char snippets)",
            reduced_tokens, _SHORT_SNIPPET_CHARS,
        )
        return reduced

    logger.warning(
        "Cannot reduce inventory further (already at 1-day lookback) — proceeding with "
        "trimmed snippets (~%d tokens)",
        trimmed_tokens,
    )
    return trimmed


def plan_and_assemble(
    config: Config,
    context: Context,
    tool_registry: ToolRegistry,
    db: Database | None = None,
) -> Digest:
    agent_config = config.agent
    if agent_config is None:
        raise ValueError("AgentConfig is required for agent mode.")

    today_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    reader_context = context.prose

    with agent_run(
        "agent-digest-run",
        {
            "model": agent_config.model,
            "reader_context": reader_context,
            "today": today_str,
        },
    ) as run_obs:
        # Step 0: Build inventory
        with iteration_span("step0-inventory", input={"lookback_days": agent_config.inventory_lookback_days}) as span:
            if db is None:
                logger.warning("No DB provided to plan_and_assemble — inventory will be empty")
            inventory, tweet_inventory, short_id_map = build_inventory(config, db)
            logger.info("Inventory built: %d chars, %d items", len(inventory), len(short_id_map))
            span.update(output={"chars": len(inventory), "items": len(short_id_map), "preview": inventory[:300]})

        items_by_short_id: dict[str, Item] = {}
        if db is not None:
            for short_id, real_id in short_id_map.items():
                item = db.get_item(real_id)
                if item is not None:
                    items_by_short_id[short_id] = item

        # Step 2: Identify angles (with adaptive inventory to stay within context limit)
        angles_model = agent_config.angles_model or agent_config.model
        angles_inventory = _build_angles_inventory(
            config, db, inventory, reader_context, short_id_map, get_context_limit(angles_model)
        )
        with iteration_span("step2-angles", input={"inventory_chars": len(angles_inventory), "context_chars": len(reader_context)}) as span:
            angles = identify_angles(config, angles_inventory, reader_context)
            span.update(output={"angles": angles})

        # Step 3a: Filter news
        with iteration_span("step3a-news", input={"angles_chars": len(angles)}) as span:
            news_output = filter_news(config, inventory, reader_context, angles, items_by_short_id)
            span.update(output={"news": news_output})

        # Step 3a': Summarize chatter
        with iteration_span("step3a-chatter", input={"tweet_chars": len(tweet_inventory)}) as span:
            chatter_output = summarize_chatter(config, tweet_inventory, reader_context)
            span.update(output={"chatter": chatter_output})

        # Step 3b: Scout research
        recently_featured_papers = _build_recently_featured_block(db)
        with iteration_span("step3b-research", input={"angles_chars": len(angles), "recently_featured_count": recently_featured_papers.count("\n- ")}) as span:
            research_output = scout_research(config, reader_context, angles, tool_registry, recently_featured=recently_featured_papers)
            span.update(output={"research": research_output})

        # Step 3c: Pull threads
        with iteration_span("step3c-threads", input={"angles_chars": len(angles)}) as span:
            threads_output = pull_threads(config, reader_context, angles, news_output, research_output, tool_registry)
            span.update(output={"threads": threads_output})

        # Step 4: Compose
        with iteration_span("step4-compose", input={
            "news_chars": len(news_output),
            "chatter_chars": len(chatter_output),
            "research_chars": len(research_output),
            "threads_chars": len(threads_output),
        }) as span:
            digest = compose_digest(config, reader_context, angles, news_output, chatter_output, research_output, threads_output)
            # Remap short inventory IDs back to real DB item IDs
            if short_id_map:
                for section in digest.sections:
                    for item in section.items:
                        if item.item_id in short_id_map:
                            item.item_id = short_id_map[item.item_id]

            # Enrich published_date from DB for items that originated there
            if db is not None:
                for section in digest.sections:
                    for item in section.items:
                        if item.published_date:
                            continue
                        db_item = db.get_item(item.item_id)
                        if db_item and db_item.timestamp:
                            item.published_date = db_item.timestamp[:10]
            span.update(output={
                "sections": len(digest.sections),
                "items": digest.item_count,
                "section_types": [s.type.value for s in digest.sections],
            })

        run_obs.update(output={
            "sections": [
                {
                    "type": s.type.value,
                    "title": s.title,
                    "items": [
                        {"title": item.title, "url": item.url, "summary": item.summary}
                        for item in s.items
                    ],
                }
                for s in digest.sections
            ],
        })

    return digest
