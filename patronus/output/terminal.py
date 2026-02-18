from __future__ import annotations

import logging

from patronus.config import Config
from patronus.digest import Digest, DigestItem, DigestSection, SectionType

logger = logging.getLogger(__name__)


_SECTION_EMOJI = {
    SectionType.LONG_FORM_PICK: "\u2b50",
    SectionType.PAPER_ROUNDUP: "\U0001f4d1",
    SectionType.HEADLINES: "\U0001f4f0",
    SectionType.SERENDIPITY: "\U0001f52e",
    SectionType.CHATTER: "\U0001f4ac",
    SectionType.FROM_NOTES: "\U0001f4d3",
}

_SEPARATOR = "\u2500" * 60


def _format_item(item: DigestItem, indent: str = "  ") -> str:
    lines = []
    title = item.title or "Untitled"
    lines.append(f"{indent}{title}")
    if item.url:
        lines.append(f"{indent}  {item.url}")
    source_parts = []
    if item.source:
        source_parts.append(item.source)
    if item.author:
        source_parts.append(item.author)
    if source_parts:
        lines.append(f"{indent}  [{' / '.join(source_parts)}]")
    if item.summary:
        lines.append(f"{indent}  {item.summary}")
    return "\n".join(lines)


def _format_section(section: DigestSection) -> str:
    emoji = _SECTION_EMOJI.get(section.type, "")
    header = f"{emoji} {section.title}" if emoji else section.title

    lines = [header, "-" * len(header.encode("ascii", errors="replace"))]
    for item in section.items:
        lines.append(_format_item(item))
        lines.append("")
    return "\n".join(lines)


def _format_deterministic_item(item: DigestItem) -> str:
    lines = []
    title = item.title or (item.scored_item.item.title if item.scored_item else "Untitled")
    url = item.url or (item.scored_item.item.url if item.scored_item else "")
    source = item.source or (item.scored_item.item.source if item.scored_item else "")
    topic = item.scored_item.matched_topic if item.scored_item else ""
    score = item.scored_item.score if item.scored_item else 0.0

    lines.append(f"  {title}")
    if url:
        lines.append(f"    {url}")
    if source:
        lines.append(f"    [{source}]")
    if topic:
        lines.append(f"    Topic: {topic} (score: {score:.3f})")
    if item.summary:
        lines.append(f"    {item.summary}")
    return "\n".join(lines)


def format_digest(digest: Digest) -> str:
    lines: list[str] = []

    date_str = digest.generated_at[:10] if digest.generated_at else "unknown"
    lines.append(f"\n{_SEPARATOR}")
    lines.append(f"  DAILY DIGEST — {date_str}")
    lines.append(f"  Mode: {digest.mode} | Items: {digest.item_count}")
    lines.append(_SEPARATOR)

    if digest.mode == "agent" and digest.sections:
        for section in digest.sections:
            lines.append("")
            lines.append(_format_section(section))
    elif digest.items:
        lines.append("")
        for item in digest.items:
            lines.append(_format_deterministic_item(item))
            lines.append("")
    else:
        lines.append("\n  No items in today's digest.")

    lines.append(_SEPARATOR)
    lines.append("")

    return "\n".join(lines)


class TerminalOutput:
    def send(self, digest: Digest, config: Config) -> None:
        print(format_digest(digest))
