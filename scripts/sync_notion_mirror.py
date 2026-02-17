import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from tqdm import tqdm

from patronus import setup_logging
from patronus.config import load_config
from patronus.notion import (
    extract_page_content,
    notion_retry,
    resolve_data_source_id,
)
from patronus.notion_mirror import NotionMirror, open_mirror

logger = logging.getLogger(__name__)

_DEFAULT_WORKERS = 5


@dataclass
class _FetchedPage:
    page_id: str
    title: str
    content: str
    source_db: str
    url: str
    created_at: str
    last_edited_at: str
    embedding: Optional[np.ndarray]


def sync_pages(
    mirror: NotionMirror,
    config,
    page_ids: list[str],
    *,
    embed: bool = False,
) -> int:
    if config.notion is None:
        logger.error("No Notion config found — set notion section in config.yaml")
        return 0

    if not config.notion_token:
        logger.error("NOTION_TOKEN env var not set")
        return 0

    from notion_client import Client as NotionClient
    from patronus.notion import notion_retry
    client = NotionClient(auth=config.notion_token)

    embed_fn = None
    if embed:
        from patronus.embed import embed_text
        embed_fn = embed_text

    db_id_to_name: dict[str, str] = {
        v.replace("-", ""): k for k, v in config.notion.database_ids.items()
    }

    synced = 0
    for raw_id in page_ids:
        page_id = raw_id.replace("-", "")
        formatted = f"{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}"
        logger.info("Fetching page %s", page_id)
        try:
            page = notion_retry(client.pages.retrieve)(page_id=formatted)
        except Exception:
            logger.warning("Failed to retrieve page %s", page_id, exc_info=True)
            continue

        parent = page.get("parent", {})
        parent_db_id = parent.get("database_id", "").replace("-", "")
        db_name = db_id_to_name.get(parent_db_id, "unknown")

        try:
            fetched = _fetch_single_page(client, page, db_name, embed_fn)
        except Exception:
            logger.warning("Failed to fetch page %s", page_id, exc_info=True)
            continue

        mirror.upsert_page(
            page_id=fetched.page_id,
            title=fetched.title,
            content=fetched.content,
            source_db=fetched.source_db,
            url=fetched.url,
            created_at=fetched.created_at,
            last_edited_at=fetched.last_edited_at,
            embedding=fetched.embedding,
        )
        logger.info("Synced page %s (db=%s, content=%d chars)", page_id, db_name, len(fetched.content))
        synced += 1

    return synced


def sync(
    mirror: NotionMirror,
    config,
    *,
    full: bool = False,
    embed: bool = False,
    workers: int = _DEFAULT_WORKERS,
) -> dict[str, int]:
    if config.notion is None:
        logger.error("No Notion config found — set notion section in config.yaml")
        return {}

    if not config.notion_token:
        logger.error("NOTION_TOKEN env var not set")
        return {}

    from notion_client import Client as NotionClient
    client = NotionClient(auth=config.notion_token)
    data_source_id_cache: dict[str, str] = {}

    embed_fn = None
    if embed:
        from patronus.embed import embed_text
        embed_fn = embed_text

    counts: dict[str, int] = {}

    for db_name, db_id in config.notion.database_ids.items():
        if full:
            lookback_days = 36500
        else:
            last_synced = mirror.get_last_synced_at(db_name)
            if last_synced:
                try:
                    cutoff_dt = datetime.fromisoformat(last_synced.replace("Z", "+00:00"))
                    lookback_days = int((datetime.now(timezone.utc) - cutoff_dt).total_seconds() / 86400) + 1
                except (ValueError, AttributeError):
                    logger.warning("Invalid last_synced_at for %s, doing full sync for this DB", db_name)
                    lookback_days = 36500
            else:
                lookback_days = 36500

        logger.info("Syncing %s (lookback_days=%d, full=%s)", db_name, lookback_days, full)

        try:
            pages_raw = _query_pages_raw(client, config, db_id, lookback_days, data_source_id_cache, db_name=db_name)
        except Exception:
            logger.warning("Failed to query Notion database %s", db_name, exc_info=True)
            counts[db_name] = 0
            continue

        fetched = _fetch_pages_parallel(client, pages_raw, db_name, embed_fn, workers=workers)

        synced = 0
        for page in fetched:
            mirror.upsert_page(
                page_id=page.page_id,
                title=page.title,
                content=page.content,
                source_db=page.source_db,
                url=page.url,
                created_at=page.created_at,
                last_edited_at=page.last_edited_at,
                embedding=page.embedding,
            )
            synced += 1

        mirror.set_last_synced_at(db_name, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        logger.info("Synced %d pages from %s", synced, db_name)
        counts[db_name] = synced

    return counts


def _fetch_single_page(
    client,
    page: dict,
    db_name: str,
    embed_fn,
) -> _FetchedPage:
    page_id: str = page["id"].replace("-", "")
    title = _extract_title(page)
    url = page.get("url", f"https://notion.so/{page_id}")
    created_at = page.get("created_time", "")
    last_edited_at = page.get("last_edited_time", "")

    try:
        content = extract_page_content(client, page["id"])
    except Exception:
        logger.warning("Failed to extract content from page %s", page["id"], exc_info=True)
        content = ""

    embedding = None
    if embed_fn is not None and content:
        try:
            embedding = embed_fn(content)
        except Exception:
            logger.warning("Failed to embed page %s", page_id, exc_info=True)

    return _FetchedPage(
        page_id=page_id,
        title=title,
        content=content,
        source_db=db_name,
        url=url,
        created_at=created_at,
        last_edited_at=last_edited_at,
        embedding=embedding,
    )


def _fetch_pages_parallel(
    client,
    pages_raw: list[dict],
    db_name: str,
    embed_fn,
    workers: int = _DEFAULT_WORKERS,
) -> list[_FetchedPage]:
    if not pages_raw:
        return []

    results: list[_FetchedPage] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_single_page, client, page, db_name, embed_fn): page
            for page in pages_raw
        }
        with tqdm(total=len(futures), desc=f"  {db_name}", unit="page", leave=True) as bar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    page = futures[future]
                    logger.warning("Unhandled error fetching page %s", page.get("id", "?"), exc_info=True)
                bar.update(1)

    return results


def _query_pages_raw(
    client,
    config,
    db_id: str,
    lookback_days: int,
    data_source_id_cache: dict[str, str],
    db_name: str = "",
) -> list[dict]:
    ds_id = resolve_data_source_id(client, db_id, data_source_id_cache)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    results: list[dict] = []
    cursor: str | None = None
    label = f"  {db_name} (index)" if db_name else "  indexing"

    with tqdm(desc=label, unit="page", leave=False) as bar:
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
            batch = response["results"]
            results.extend(batch)
            bar.update(len(batch))

            if not response.get("has_more", False):
                break
            cursor = response.get("next_cursor")

    return results


def _extract_title(page: dict) -> str:
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_items = prop.get("title", [])
            return "".join(item.get("plain_text", "") for item in title_items)
    return "Untitled"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Notion databases to a local SQLite mirror with FTS5")
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="Path to mirror SQLite DB (default: notion.mirror_path from config, or notion_mirror.sqlite3)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full resync (ignore last sync timestamp, refetch all pages)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate and store embeddings for page content",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Number of parallel threads for fetching page content (default: {_DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        metavar="PAGE_ID",
        default=None,
        help="Re-fetch and upsert specific page IDs (bypasses normal DB query; useful for targeted resync)",
    )
    args = parser.parse_args()

    setup_logging()

    config = load_config()

    mirror_path = args.mirror
    if mirror_path is None:
        if config.notion and config.notion.mirror_path:
            mirror_path = config.notion.mirror_path
        else:
            mirror_path = "notion_mirror.sqlite3"

    logger.info("Opening Notion mirror at %s", mirror_path)
    with open_mirror(mirror_path) as mirror:
        if args.pages:
            synced = sync_pages(mirror, config, args.pages, embed=args.embed)
            logger.info("Sync complete: %d page(s) resynced", synced)
        else:
            counts = sync(mirror, config, full=args.full, embed=args.embed, workers=args.workers)
            total = sum(counts.values())
            logger.info("Sync complete: %d pages total across %d databases", total, len(counts))
            for db_name, count in counts.items():
                logger.info("  %s: %d pages", db_name, count)


if __name__ == "__main__":
    main()
