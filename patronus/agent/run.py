from __future__ import annotations

import logging
from datetime import datetime, timezone

from patronus.agent._compose import compose_digest
from patronus.agent._inventory import build_inventory
from patronus.agent._steps import filter_news, identify_angles, pull_threads, scout_research, summarize_chatter
from patronus.config import Config
from patronus.context import Context
from patronus.db import Database, Item
from patronus.digest import Digest
from patronus.observability import agent_run, iteration_span
from patronus.tools import ToolRegistry

logger = logging.getLogger(__name__)


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

        # Step 2: Identify angles
        with iteration_span("step2-angles", input={"inventory_chars": len(inventory), "context_chars": len(reader_context)}) as span:
            angles = identify_angles(config, inventory, reader_context)
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
        with iteration_span("step3b-research", input={"angles_chars": len(angles)}) as span:
            research_output = scout_research(config, reader_context, angles, tool_registry)
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
