from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from patronus.config import Config
from patronus.db import Database, Item

logger = logging.getLogger(__name__)

_FALLBACK_LOOKBACK_DAYS = 3


def _format_item(item: Item, short_id: str, previously_featured: bool, cluster_note: str, snippet_chars: int = 300) -> str:
    parts = [
        f"ID: {short_id}",
        f"Title: {item.title or '(no title)'}",
        f"URL: {item.url}",
        f"Source: {item.source or '?'}",
        f"Type: {item.item_type}",
        f"Date: {item.timestamp or item.ingested_at or '?'}",
    ]
    if item.author:
        parts.append(f"Author: {item.author}")
    if previously_featured:
        parts.append("Flag: PREVIOUSLY_FEATURED")
    if cluster_note:
        parts.append(f"Cluster: {cluster_note}")
    if item.text and snippet_chars > 0:
        snippet = item.text[:snippet_chars].replace("\n", " ").strip()
        if len(item.text) > snippet_chars:
            snippet += "…"
        parts.append(f"Snippet: {snippet}")
    return "\n".join(f"  {p}" for p in parts)


def build_inventory(
    config: Config,
    db: Database | None,
    snippet_chars: int = 300,
    lookback_days_override: int | None = None,
) -> tuple[str, str, dict[str, str]]:
    if db is None:
        return "(No inventory available — DB not provided.)", "(No tweet inventory available — DB not provided.)", {}
    agent_cfg = config.agent
    lookback_days = lookback_days_override if lookback_days_override is not None else (
        agent_cfg.inventory_lookback_days if agent_cfg else _FALLBACK_LOOKBACK_DAYS
    )

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Inventory: using %d-day lookback: %s", lookback_days, cutoff)

    items = db.get_items_since(cutoff)
    logger.info("Inventory: %d items since %s", len(items), cutoff)

    if not items:
        return "No new items since last digest.", "(No tweets in inventory.)", {}

    recently_digested = db.get_recently_digested_item_ids(lookback_days=7)

    url_to_items: dict[str, list[Item]] = {}
    for item in items:
        url_to_items.setdefault(item.url, []).append(item)

    source_item_clusters: dict[str, list[Item]] = {}
    for item in items:
        if item.source_item_id:
            source_item_clusters.setdefault(item.source_item_id, []).append(item)

    by_source: dict[str, list[Item]] = {}
    for item in items:
        key = item.source or item.source_type or "unknown"
        by_source.setdefault(key, []).append(item)

    # Assign short sequential IDs for the inventory so models can reference items easily
    short_id_map: dict[str, str] = {}  # short_id -> real DB id
    real_to_short: dict[str, str] = {}  # real DB id -> short_id
    counter = 1
    for source_items in sorted(by_source.items(), key=lambda x: x[0]):
        for item in source_items[1]:
            short_id = str(counter)
            short_id_map[short_id] = item.id
            real_to_short[item.id] = short_id
            counter += 1

    sections: list[str] = []
    sections.append(f"Content inventory — {len(items)} items since {cutoff}\n")

    for source_name, source_items in sorted(by_source.items()):
        sections.append(f"## {source_name} ({len(source_items)} items)")
        for item in source_items:
            previously_featured = item.id in recently_digested

            cluster_note = ""
            refs = source_item_clusters.get(item.id, [])
            if refs:
                cluster_note = f"referenced by {len(refs)} item(s)"

            sections.append(_format_item(item, real_to_short[item.id], previously_featured, cluster_note, snippet_chars))
            sections.append("")

    digest_history_counts: dict[str, int] = {}
    for item in items:
        history = json.loads(item.digest_history or "[]")
        if history:
            digest_history_counts[item.id] = len(history)

    summary_parts = [
        f"Total: {len(items)} items",
        f"Sources: {len(by_source)}",
        f"Previously featured: {sum(1 for i in items if i.id in recently_digested)}",
    ]
    sections.append("---\nInventory summary: " + " | ".join(summary_parts))

    main_inventory = "\n".join(sections)

    tweet_items = [item for item in items if item.item_type == "tweet"]
    if tweet_items:
        tweet_sections: list[str] = [f"Tweet inventory — {len(tweet_items)} tweets since {cutoff}\n"]
        tweet_by_source: dict[str, list[Item]] = {}
        for item in tweet_items:
            key = item.source or item.source_type or "unknown"
            tweet_by_source.setdefault(key, []).append(item)
        for source_name, source_items in sorted(tweet_by_source.items()):
            tweet_sections.append(f"## {source_name} ({len(source_items)} items)")
            for item in source_items:
                previously_featured = item.id in recently_digested
                cluster_note = ""
                refs = source_item_clusters.get(item.id, [])
                if refs:
                    cluster_note = f"referenced by {len(refs)} item(s)"
                tweet_sections.append(_format_item(item, real_to_short[item.id], previously_featured, cluster_note, snippet_chars))
                tweet_sections.append("")
        tweet_inventory = "\n".join(tweet_sections)
    else:
        tweet_inventory = "(No tweets in inventory.)"

    return main_inventory, tweet_inventory, short_id_map
