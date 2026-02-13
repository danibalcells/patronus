from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import mktime
from typing import Optional

import feedparser
import trafilatura

from patronus.db import Database, Feed
from patronus.embed import embed_batch, embed_text

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 25_000
DEFAULT_WORKERS = 8


def _entry_author(entry: object) -> Optional[str]:
    if hasattr(entry, "author") and getattr(entry, "author"):
        return getattr(entry, "author")
    if hasattr(entry, "authors") and getattr(entry, "authors"):
        authors = getattr(entry, "authors")
        try:
            first = authors[0]
            if isinstance(first, dict):
                return first.get("name") or first.get("email")
            return getattr(first, "name", None)
        except Exception:
            return None
    if hasattr(entry, "author_detail") and getattr(entry, "author_detail"):
        detail = getattr(entry, "author_detail")
        try:
            return getattr(detail, "name", None) or getattr(detail, "email", None)
        except Exception:
            return None
    return None


def _parse_timestamp(entry: object) -> Optional[str]:
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                dt = datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                continue
    return None


def _extract_full_text(url: str) -> Optional[str]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False
            )
            if text:
                return text[:MAX_TEXT_CHARS]
    except Exception:
        logger.debug("trafilatura extraction failed for %s", url, exc_info=True)
    return None


def _display_author(
    author: Optional[str], feed_title: Optional[str]
) -> Optional[str]:
    if author and feed_title and author.lower() != feed_title.lower():
        return f"{author} - {feed_title}"
    return author or feed_title


def _parse_single_feed(feed: Feed) -> Optional[object]:
    try:
        return feedparser.parse(
            feed.url,
            etag=feed.etag or None,
            modified=feed.last_modified or None,
        )
    except Exception:
        logger.exception("Failed to parse feed %s", feed.url)
        return None


def poll_feeds(
    db: Database,
    *,
    limit: Optional[int] = None,
    feed_limit: Optional[int] = None,
    skip_embed: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> list[str]:
    feeds = db.get_active_feeds()
    if feed_limit is not None:
        feeds = feeds[:feed_limit]
    if not feeds:
        logger.info("No active feeds to poll")
        return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parsed_feeds = list(pool.map(_parse_single_feed, feeds))

    raw_entries: list[dict] = []

    for feed, parsed in zip(feeds, parsed_feeds):
        if parsed is None:
            continue

        if getattr(parsed, "status", None) == 304:
            logger.debug("Feed unchanged (304): %s", feed.url)
            db.update_feed_poll(
                feed.id, etag=feed.etag, last_modified=feed.last_modified
            )
            continue

        feed_title = getattr(parsed.feed, "title", None) or feed.name
        new_etag = getattr(parsed, "etag", None)
        new_modified = getattr(parsed, "modified", None)

        feed_new = 0
        feed_skipped = 0

        for entry in parsed.entries:
            if limit is not None and feed_new >= limit:
                break

            link = getattr(entry, "link", None)
            if not link:
                continue

            if db.get_item_by_url(link) is not None:
                feed_skipped += 1
                continue

            summary = getattr(entry, "summary", None) or getattr(
                entry, "description", None
            )

            raw_entries.append(
                {
                    "url": link,
                    "title": getattr(entry, "title", None),
                    "author": _entry_author(entry),
                    "timestamp": _parse_timestamp(entry),
                    "summary": summary,
                    "feed_title": feed_title,
                }
            )
            feed_new += 1

        db.update_feed_poll(feed.id, etag=new_etag, last_modified=new_modified)
        logger.info(
            "Feed %s: %d new, %d skipped",
            feed.name or feed.url,
            feed_new,
            feed_skipped,
        )

    urls_to_fetch = [e["url"] for e in raw_entries]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        full_texts = list(pool.map(_extract_full_text, urls_to_fetch))

    pending: list[dict] = []
    for entry_data, full_text in zip(raw_entries, full_texts):
        text = full_text
        if not text and entry_data["summary"]:
            text = entry_data["summary"][:MAX_TEXT_CHARS]

        pending.append(
            {
                "url": entry_data["url"],
                "source_type": "rss",
                "title": entry_data["title"],
                "author": _display_author(
                    entry_data["author"], entry_data["feed_title"]
                ),
                "source": entry_data["feed_title"],
                "text": text,
                "timestamp": entry_data["timestamp"],
            }
        )

    new_item_ids: list[str] = []
    embeddings: list[Optional[bytes]] = [None] * len(pending)

    if not skip_embed:
        embeddable_indices = [i for i, p in enumerate(pending) if p["text"]]
        if embeddable_indices:
            try:
                texts = [pending[i]["text"] for i in embeddable_indices]
                results = embed_batch(texts)
                for idx, emb in zip(embeddable_indices, results):
                    embeddings[idx] = emb
            except Exception:
                logger.exception(
                    "Batch embedding failed; storing items without embeddings"
                )

    for item_data, embedding in zip(pending, embeddings):
        try:
            item_id = db.add_item(
                url=item_data["url"],
                source_type=item_data["source_type"],
                title=item_data["title"],
                author=item_data["author"],
                source=item_data["source"],
                text=item_data["text"],
                embedding=embedding,
                timestamp=item_data["timestamp"],
            )
            new_item_ids.append(item_id)
        except Exception:
            logger.exception("Failed to store item %s", item_data["url"])

    logger.info("Poll complete: %d new items ingested", len(new_item_ids))
    return new_item_ids


def ingest_url(db: Database, url: str) -> Optional[str]:
    if db.get_item_by_url(url) is not None:
        logger.info("URL already exists: %s", url)
        return None

    title: Optional[str] = None
    author: Optional[str] = None
    text: Optional[str] = None

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            result = trafilatura.bare_extraction(
                downloaded, include_comments=False, include_tables=False
            )
            if result:
                text = (result.get("text") or "")[:MAX_TEXT_CHARS] or None
                title = result.get("title")
                author = result.get("author")
    except Exception:
        logger.exception("Content extraction failed for %s", url)

    if not text:
        logger.warning("Could not extract text from %s", url)

    embedding = None
    if text:
        try:
            embedding = embed_text(text)
        except Exception:
            logger.exception("Embedding failed for %s", url)

    try:
        item_id = db.add_item(
            url=url,
            source_type="manual",
            title=title,
            author=author,
            text=text,
            embedding=embedding,
        )
        logger.info("Ingested manual URL: %s (id=%s)", url, item_id)
        return item_id
    except Exception:
        logger.exception("Failed to store manual URL: %s", url)
        return None
