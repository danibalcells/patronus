from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from patronus.config import Config
from patronus.db import Database
from patronus.interests import load_interest_vectors
from patronus.rank import ScoredItem, rank_unread, select_digest
from patronus.summarize import summarize_item

logger = logging.getLogger(__name__)


class SectionType(str, Enum):
    LONG_FORM_PICK = "long_form_pick"
    PAPER_ROUNDUP = "paper_roundup"
    HEADLINES = "headlines"
    SERENDIPITY = "serendipity"
    CHATTER = "chatter"
    FROM_NOTES = "from_notes"


@dataclass
class DigestItem:
    scored_item: ScoredItem | None = None
    summary: str = ""
    item_id: str = ""
    title: str = ""
    url: str = ""
    source: str = ""
    author: str = ""
    item_type: str = "article"
    published_date: str = ""


@dataclass
class DigestSection:
    type: SectionType
    title: str
    items: list[DigestItem] = field(default_factory=list)


@dataclass
class Digest:
    items: list[DigestItem] = field(default_factory=list)
    sections: list[DigestSection] = field(default_factory=list)
    generated_at: str = ""
    mode: str = "deterministic"

    @property
    def item_count(self) -> int:
        if self.sections:
            return sum(len(s.items) for s in self.sections)
        return len(self.items)

    @property
    def all_items(self) -> list[DigestItem]:
        if self.sections:
            result: list[DigestItem] = []
            for s in self.sections:
                result.extend(s.items)
            return result
        return self.items


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


def generate_digest_deterministic(config: Config, db: Database, *, skip_penalty: bool = False) -> Digest:
    interest_vectors = load_interest_vectors(config)
    unread = db.get_unread_items()
    logger.info("Generating deterministic digest from %d unread items", len(unread))

    generated_at = _now_utc()

    if not unread:
        return Digest(items=[], generated_at=generated_at, mode="deterministic")

    scored = rank_unread(unread, interest_vectors)
    if not skip_penalty:
        scored = _apply_repeat_penalty(scored, config.digest.repeat_penalty)
    per_topic_max = {key: tc.max_items for key, tc in config.topics.items()}
    selected = select_digest(
        scored,
        size=config.digest.size,
        max_per_topic=per_topic_max,
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

        digest_items.append(DigestItem(
            scored_item=scored_item,
            summary=summary,
            item_id=scored_item.item.id,
            title=scored_item.item.title or "",
            url=scored_item.item.url,
            source=scored_item.item.source or "",
            author=scored_item.item.author or "",
            item_type=scored_item.item.item_type,
        ))
        db.update_digest_history(scored_item.item.id, today)

    digest = Digest(items=digest_items, generated_at=generated_at, mode="deterministic")

    db.save_digest(
        generated_at=generated_at,
        item_count=digest.item_count,
        formatted_text="",
        items=[
            {
                "item_id": di.item_id or (di.scored_item.item.id if di.scored_item else ""),
                "summary": di.summary,
                "score": di.scored_item.score if di.scored_item else 0.0,
                "matched_topic": di.scored_item.matched_topic if di.scored_item else "",
            }
            for di in digest_items
        ],
    )

    logger.info("Deterministic digest generated with %d items", digest.item_count)
    return digest


def generate_digest(config: Config, db: Database, *, skip_penalty: bool = False) -> Digest:
    if config.digest.mode == "agent":
        from patronus.pipeline import DigestPipeline
        pipeline = DigestPipeline(config, db)
        return pipeline.generate()
    return generate_digest_deterministic(config, db, skip_penalty=skip_penalty)
