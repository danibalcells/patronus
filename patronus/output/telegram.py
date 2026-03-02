from __future__ import annotations

import asyncio
import logging
import re

import telegram as tg

from patronus.config import Config
from patronus.digest import Digest, DigestItem, DigestSection, SectionType

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def _escape_markdown_v2(text: str) -> str:
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", text)


def _escape_url(url: str) -> str:
    return url.replace("\\", "\\\\").replace(")", "\\)")


def _format_item_line(item: DigestItem) -> str:
    title = _clean_html(item.title or "Untitled")
    url = item.url
    source = item.source or ""
    date = item.published_date[:7] if item.published_date else ""  # YYYY-MM

    line = f"[{_escape_markdown_v2(title)}]({_escape_url(url)})"
    meta_parts = []
    if source:
        meta_parts.append(f"_{_escape_markdown_v2(source)}_")
    if date:
        meta_parts.append(_escape_markdown_v2(date))
    if meta_parts:
        line += " — " + ", ".join(meta_parts)
    if item.summary:
        line += f"\n{_escape_markdown_v2(item.summary)}"
    return line


def _section_emoji(section_type: SectionType) -> str:
    return {
        SectionType.LONG_FORM_PICK: "\u2b50",
        SectionType.PAPER_ROUNDUP: "\U0001f4d1",
        SectionType.HEADLINES: "\U0001f4f0",
        SectionType.WHATS_NEW: "\U0001f4f0",
        SectionType.RESEARCH_ROUNDUP: "\U0001f4d1",
        SectionType.THREADS: "\U0001f9f5",
        SectionType.SERENDIPITY: "\U0001f52e",
        SectionType.CHATTER: "\U0001f4ac",
        SectionType.FROM_NOTES: "\U0001f4d3",
    }.get(section_type, "")


def _format_section_agent(section: DigestSection) -> str:
    emoji = _section_emoji(section.type)
    header = f"*{emoji} {_escape_markdown_v2(section.title)}*" if emoji else f"*{_escape_markdown_v2(section.title)}*"
    lines = [header]
    for item in section.items:
        lines.append(_format_item_line(item))
    return "\n\n".join(lines)


def _format_topic_section_deterministic(topic_name: str, items: list[DigestItem]) -> str:
    lines: list[str] = [f"\n*{_escape_markdown_v2(topic_name)}*"]
    for di in items:
        title = _clean_html(di.title or (di.scored_item.item.title if di.scored_item else "Untitled"))
        source = di.source or (di.scored_item.item.source if di.scored_item else "") or ""
        url = di.url or (di.scored_item.item.url if di.scored_item else "")

        line = f"[{_escape_markdown_v2(title)}]({_escape_url(url)})"
        if source:
            line += f" — _{_escape_markdown_v2(source)}_"
        if di.summary:
            line += f"\n{_escape_markdown_v2(di.summary)}"
        lines.append(line)
    return "\n\n".join(lines)


def format_telegram_sections(digest: Digest, config: Config) -> list[str]:
    if digest.mode == "agent" and digest.sections:
        return _format_agent_sections(digest)
    return _format_deterministic_sections(digest, config)


def _format_agent_sections(digest: Digest) -> list[str]:
    if not digest.sections:
        return ["No items for today\\'s digest\\."]

    date_str = digest.generated_at[:10] if digest.generated_at else ""
    header = f"*Daily Digest* — {_escape_markdown_v2(date_str)}"

    sections: list[str] = [header]
    for section in digest.sections:
        if section.items:
            sections.append(_format_section_agent(section))

    return sections


def _format_deterministic_sections(digest: Digest, config: Config) -> list[str]:
    if not digest.items:
        return ["No items for today\\'s digest\\."]

    groups: dict[str, list[DigestItem]] = {}
    for di in digest.items:
        topic = di.scored_item.matched_topic if di.scored_item else "other"
        groups.setdefault(topic, []).append(di)

    date_str = digest.generated_at[:10] if digest.generated_at else ""
    header = f"*Daily Digest* — {_escape_markdown_v2(date_str)}"

    sections: list[str] = [header]

    for topic_key in config.topics:
        if topic_key not in groups:
            continue
        topic_name = config.topics[topic_key].name
        sections.append(_format_topic_section_deterministic(topic_name, groups[topic_key]))

    for topic_key, items in groups.items():
        if topic_key in config.topics:
            continue
        sections.append(_format_topic_section_deterministic(topic_key, items))

    return sections


def format_telegram(digest: Digest, config: Config) -> str:
    return "\n\n".join(format_telegram_sections(digest, config))


def _pack_sections(sections: list[str], limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    separator = "\n\n"
    messages: list[str] = []
    current = ""
    for section in sections:
        candidate = f"{current}{separator}{section}" if current else section
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                messages.append(current)
            current = section
    if current:
        messages.append(current)
    return messages


async def _send_markdown(bot: tg.Bot, chat_id: str, text: str) -> None:
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="MarkdownV2",
        disable_web_page_preview=True,
    )


async def _send_sections(bot: tg.Bot, chat_id: str, sections: list[str]) -> None:
    for message in _pack_sections(sections):
        await _send_markdown(bot, chat_id, message)


class TelegramOutput:
    def send(self, digest: Digest, config: Config) -> None:
        if not config.telegram_bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set, skipping Telegram output")
            return

        sections = format_telegram_sections(digest, config)
        bot = tg.Bot(token=config.telegram_bot_token)

        async def _send() -> None:
            async with bot:
                await _send_sections(bot, config.telegram.chat_id, sections)

        asyncio.run(_send())


def send_message(config: Config, text: str) -> None:
    bot = tg.Bot(token=config.telegram_bot_token)

    async def _send() -> None:
        async with bot:
            await _send_markdown(bot, config.telegram.chat_id, text)

    asyncio.run(_send())
