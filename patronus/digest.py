from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from patronus.config import Config
from patronus.db import Database
from patronus.interests import load_interest_vectors
from patronus.rank import ScoredItem, rank_unread, select_digest
from patronus.summarize import summarize_item

logger = logging.getLogger(__name__)


@dataclass
class DigestItem:
    scored_item: ScoredItem
    summary: str


@dataclass
class Digest:
    items: list[DigestItem] = field(default_factory=list)
    generated_at: str = ""

    @property
    def item_count(self) -> int:
        return len(self.items)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply_repeat_penalty(
    scored_items: list[ScoredItem],
    repeat_penalty: float,
) -> list[ScoredItem]:
    for scored in scored_items:
        history = json.loads(scored.item.digest_history or "[]")
        times_digested = len(history)
        if times_digested > 0:
            scored.score *= repeat_penalty ** times_digested
    scored_items.sort(key=lambda s: s.score, reverse=True)
    return scored_items


def generate_digest(config: Config, db: Database, *, skip_penalty: bool = False) -> Digest:
    interest_vectors = load_interest_vectors(config)
    unread = db.get_unread_items()
    logger.info("Generating digest from %d unread items", len(unread))

    generated_at = _now_utc()

    if not unread:
        return Digest(items=[], generated_at=generated_at)

    scored = rank_unread(unread, interest_vectors)
    if not skip_penalty:
        scored = _apply_repeat_penalty(scored, config.digest.repeat_penalty)
    selected = select_digest(
        scored,
        size=config.digest.size,
        max_per_topic=config.digest.max_per_topic,
    )

    digest_items: list[DigestItem] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for i, scored_item in enumerate(selected, 1):
        logger.info("Summarizing item %d/%d: %s", i, len(selected), scored_item.item.title or scored_item.item.url)
        topic_key = scored_item.matched_topic
        topic_config = config.topics.get(topic_key)
        interest_text = topic_config.description if topic_config else ""

        summary = ""
        try:
            summary = summarize_item(
                title=scored_item.item.title or "",
                text=scored_item.item.text or "",
                interest_description=interest_text,
                model=config.summarization.model,
            )
        except Exception:
            logger.exception("Failed to summarize item %s", scored_item.item.id)

        digest_items.append(DigestItem(scored_item=scored_item, summary=summary))
        db.update_digest_history(scored_item.item.id, today)

    digest = Digest(items=digest_items, generated_at=generated_at)

    formatted = format_telegram(digest, config)
    db.save_digest(
        generated_at=generated_at,
        item_count=digest.item_count,
        formatted_text=formatted,
        items=[
            {
                "item_id": di.scored_item.item.id,
                "summary": di.summary,
                "score": di.scored_item.score,
                "matched_topic": di.scored_item.matched_topic,
            }
            for di in digest_items
        ],
    )

    logger.info("Digest generated with %d items", digest.item_count)
    return digest


def _escape_markdown_v2(text: str) -> str:
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", text)


def _escape_url(url: str) -> str:
    return url.replace("\\", "\\\\").replace(")", "\\)")


def format_telegram(digest: Digest, config: Config) -> str:
    if not digest.items:
        return "No items for today\\'s digest\\."

    groups: dict[str, list[DigestItem]] = {}
    for di in digest.items:
        topic = di.scored_item.matched_topic
        groups.setdefault(topic, []).append(di)

    parts: list[str] = []
    date_str = digest.generated_at[:10] if digest.generated_at else ""
    parts.append(f"*Daily Digest* — {_escape_markdown_v2(date_str)}")

    for topic_key, items in groups.items():
        topic_config = config.topics.get(topic_key)
        topic_name = topic_config.name if topic_config else topic_key
        parts.append(f"\n*{_escape_markdown_v2(topic_name)}*")

        for di in items:
            title = di.scored_item.item.title or "Untitled"
            source = di.scored_item.item.source or ""
            url = di.scored_item.item.url
            summary = di.summary

            line = f"[{_escape_markdown_v2(title)}]({_escape_url(url)})"
            if source:
                line += f" — _{_escape_markdown_v2(source)}_"
            if summary:
                line += f"\n{_escape_markdown_v2(summary)}"

            parts.append(line)

    return "\n\n".join(parts)
