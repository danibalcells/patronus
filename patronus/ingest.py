from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from time import mktime
from typing import Optional
from urllib.parse import urlparse

import feedparser
import trafilatura

from patronus.db import Database, Feed
from patronus.embed import embed_batch, embed_text

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 25_000
DEFAULT_WORKERS = 8

ITEM_TYPE_PATTERNS: dict[str, list[str]] = {
    "tweet": ["x.com", "twitter.com"],
    "paper": ["arxiv.org", "openreview.net"],
}

LINK_EXTRACT_DOMAINS: set[str] = {
    "arxiv.org",
    "openreview.net",
    "anthropic.com",
    "deepmind.google",
    "openai.com",
}

LINK_EXTRACT_DOMAIN_SUFFIXES: list[str] = [
    ".substack.com",
]


class _TweetHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._parts.append("\n")
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._links.append(href)

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    @property
    def text(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", unescape(raw)).strip()

    @property
    def links(self) -> list[str]:
        return list(self._links)


def _parse_tweet_html(html: str) -> tuple[str, list[str]]:
    parser = _TweetHTMLParser()
    parser.feed(html)
    return parser.text, parser.links


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href and href.startswith("http"):
                self._links.append(href)

    @property
    def links(self) -> list[str]:
        return list(self._links)


def _extract_links_from_html(html: str) -> list[str]:
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        logger.debug("HTML link extraction failed", exc_info=True)
    return parser.links


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").removeprefix("www.")
    if hostname == "arxiv.org" and parsed.path.startswith("/pdf/"):
        return url.replace("/pdf/", "/abs/", 1)
    return url


def _is_allowed_link(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").removeprefix("www.")
    if hostname in LINK_EXTRACT_DOMAINS:
        return True
    return any(hostname.endswith(suffix) for suffix in LINK_EXTRACT_DOMAIN_SUFFIXES)


def _filter_allowed_links(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalized = _normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        if _is_allowed_link(normalized):
            result.append(normalized)
    return result


def classify_item_type(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    hostname = hostname.removeprefix("www.")
    for item_type, domains in ITEM_TYPE_PATTERNS.items():
        if hostname in domains:
            return item_type
    return "article"


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


def _extract_tweet_content(entry_data: dict) -> tuple[Optional[str], list[str]]:
    summary = entry_data.get("summary")
    if not summary:
        return None, []
    text, links = _parse_tweet_html(summary)
    return (text[:MAX_TEXT_CHARS] if text else None), links


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


def ingest_linked_items(
    db: Database,
    parent_item_id: str,
    urls: list[str],
    *,
    skip_embed: bool = False,
) -> list[str]:
    new_ids: list[str] = []
    for url in urls:
        try:
            if db.get_item_by_url(url) is not None:
                logger.debug("Linked URL already exists, skipping: %s", url)
                continue

            text = _extract_full_text(url)

            embedding = None
            if text and not skip_embed:
                try:
                    embedding = embed_text(text)
                except Exception:
                    logger.exception("Embedding failed for linked URL: %s", url)

            item_id = db.add_item(
                url=url,
                source_type="link_extraction",
                item_type=classify_item_type(url),
                text=text,
                embedding=embedding,
                source_item_id=parent_item_id,
            )
            logger.info(
                "Ingested linked item: %s (id=%s, parent=%s)",
                url,
                item_id,
                parent_item_id,
            )
            new_ids.append(item_id)
        except Exception:
            logger.exception("Failed to ingest linked URL: %s", url)
    return new_ids


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

    item_types = [classify_item_type(e["url"]) for e in raw_entries]

    non_tweet_indices = [i for i, t in enumerate(item_types) if t != "tweet"]
    fetched_texts: dict[int, Optional[str]] = {}
    if non_tweet_indices:
        urls_to_fetch = [raw_entries[i]["url"] for i in non_tweet_indices]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_extract_full_text, urls_to_fetch))
        for idx, full_text in zip(non_tweet_indices, results):
            fetched_texts[idx] = full_text

    pending: list[dict] = []
    for i, entry_data in enumerate(raw_entries):
        item_type = item_types[i]

        if item_type == "tweet":
            text, raw_links = _extract_tweet_content(entry_data)
        else:
            text = fetched_texts.get(i)
            if not text and entry_data["summary"]:
                text = entry_data["summary"][:MAX_TEXT_CHARS]
            raw_links = _extract_links_from_html(entry_data.get("summary") or "")

        pending.append(
            {
                "url": entry_data["url"],
                "source_type": "rss",
                "item_type": item_type,
                "title": entry_data["title"],
                "author": _display_author(
                    entry_data["author"], entry_data["feed_title"]
                ),
                "source": entry_data["feed_title"],
                "text": text,
                "timestamp": entry_data["timestamp"],
                "_links": _filter_allowed_links(raw_links),
            }
        )

    new_item_ids: list[str] = []
    embeddings: list[Optional[bytes]] = [None] * len(pending)

    if not skip_embed:
        embeddable_indices = [i for i, p in enumerate(pending) if p["text"]]
        if embeddable_indices:
            logger.info(
                "Embedding %d items:", len(embeddable_indices)
            )
            for i in embeddable_indices:
                logger.info(
                    "  → %s (%s)",
                    pending[i]["title"] or "untitled",
                    pending[i]["url"],
                )
            try:
                texts = [pending[i]["text"] for i in embeddable_indices]
                results = embed_batch(texts)
                for idx, emb in zip(embeddable_indices, results):
                    embeddings[idx] = emb
                logger.info("Embedding complete")
            except Exception:
                logger.exception(
                    "Batch embedding failed; storing items without embeddings"
                )

    parent_links: list[tuple[str, list[str]]] = []
    for item_data, embedding in zip(pending, embeddings):
        try:
            item_id = db.add_item(
                url=item_data["url"],
                source_type=item_data["source_type"],
                item_type=item_data["item_type"],
                title=item_data["title"],
                author=item_data["author"],
                source=item_data["source"],
                text=item_data["text"],
                embedding=embedding,
                timestamp=item_data["timestamp"],
            )
            new_item_ids.append(item_id)
            if item_data.get("_links"):
                parent_links.append((item_id, item_data["_links"]))
        except Exception:
            logger.exception("Failed to store item %s", item_data["url"])

    for parent_id, links in parent_links:
        linked_ids = ingest_linked_items(
            db, parent_id, links, skip_embed=skip_embed
        )
        new_item_ids.extend(linked_ids)

    logger.info("Poll complete: %d new items ingested", len(new_item_ids))
    return new_item_ids


def ingest_url(db: Database, url: str) -> Optional[str]:
    if db.get_item_by_url(url) is not None:
        logger.info("URL already exists: %s", url)
        return None

    title: Optional[str] = None
    author: Optional[str] = None
    text: Optional[str] = None
    downloaded: Optional[str] = None

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
            item_type=classify_item_type(url),
            title=title,
            author=author,
            text=text,
            embedding=embedding,
        )
        logger.info("Ingested manual URL: %s (id=%s)", url, item_id)

        if downloaded:
            links = _filter_allowed_links(_extract_links_from_html(downloaded))
            if links:
                ingest_linked_items(db, item_id, links)

        return item_id
    except Exception:
        logger.exception("Failed to store manual URL: %s", url)
        return None
