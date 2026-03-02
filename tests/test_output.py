from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from patronus.config import (
    Config,
    DigestConfig,
    EmbeddingConfig,
    PollingConfig,
    TelegramConfig,
    TopicConfig,
)
from patronus.digest import Digest, DigestItem, DigestSection, SectionType
from patronus.output.telegram import (
    _escape_markdown_v2,
    _escape_url,
    _format_item_line,
    _format_section_agent,
    _pack_sections,
    _section_emoji,
    format_telegram,
    format_telegram_sections,
)
from patronus.output.terminal import TerminalOutput, _format_item, _format_section
from patronus.rank import ScoredItem


def _make_config() -> Config:
    return Config(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),
        telegram=TelegramConfig(),
        topics={
            "ml": TopicConfig(name="Technical AI/ML", description="ML research"),
        },
    )


def _make_agent_digest() -> Digest:
    return Digest(
        sections=[
            DigestSection(
                type=SectionType.LONG_FORM_PICK,
                title="Today's Pick",
                items=[DigestItem(
                    item_id="item-1",
                    title="Attention Is All You Need (Still)",
                    url="https://arxiv.org/abs/1234",
                    source="Arxiv",
                    author="Vaswani et al.",
                    summary="A retrospective on the transformer architecture.",
                )],
            ),
            DigestSection(
                type=SectionType.HEADLINES,
                title="Headlines",
                items=[
                    DigestItem(
                        item_id="item-2",
                        title="GPT-5 Released",
                        url="https://news.com/gpt5",
                        summary="OpenAI's latest model.",
                    ),
                    DigestItem(
                        item_id="item-3",
                        title="EU AI Act Update",
                        url="https://eu.com/ai-act",
                        source="EU News",
                        summary="New regulations.",
                    ),
                ],
            ),
        ],
        generated_at="2026-02-16T08:00:00Z",
        mode="agent",
    )


class TestEscapeUrl:
    def test_escapes_parentheses(self) -> None:
        assert _escape_url("https://x.com/path(1)") == "https://x.com/path(1\\)"

    def test_no_change_for_normal_url(self) -> None:
        assert _escape_url("https://example.com/page") == "https://example.com/page"


class TestSectionEmoji:
    def test_all_types_have_emojis(self) -> None:
        for st in SectionType:
            assert _section_emoji(st) != ""


class TestFormatItemLine:
    def test_basic_item(self) -> None:
        item = DigestItem(title="Paper Title", url="https://x.com", summary="Good paper.")
        line = _format_item_line(item)
        assert "Paper Title" in line
        assert "https://x.com" in line
        assert "Good paper" in line

    def test_item_with_source(self) -> None:
        item = DigestItem(title="Paper", url="https://x.com", source="Arxiv", summary="Summary.")
        line = _format_item_line(item)
        assert "Arxiv" in line

    def test_item_without_title(self) -> None:
        item = DigestItem(url="https://x.com", summary="Summary.")
        line = _format_item_line(item)
        assert "Untitled" in line

    def test_escapes_markdown_in_title(self) -> None:
        item = DigestItem(title="Score: 1.5", url="https://x.com", summary="s")
        line = _format_item_line(item)
        assert "1\\.5" in line


class TestFormatSectionAgent:
    def test_section_has_header_and_items(self) -> None:
        section = DigestSection(
            type=SectionType.PAPER_ROUNDUP,
            title="Paper Roundup",
            items=[
                DigestItem(title="Paper A", url="https://a.com", summary="Summary A."),
                DigestItem(title="Paper B", url="https://b.com", summary="Summary B."),
            ],
        )
        text = _format_section_agent(section)
        assert "Paper Roundup" in text
        assert "Paper A" in text
        assert "Paper B" in text


class TestPackSections:
    def test_fits_in_one_message(self) -> None:
        sections = ["Short text", "Another short text"]
        messages = _pack_sections(sections, limit=100)
        assert len(messages) == 1

    def test_splits_long_sections(self) -> None:
        sections = ["A" * 50, "B" * 50]
        messages = _pack_sections(sections, limit=60)
        assert len(messages) == 2
        assert "A" in messages[0]
        assert "B" in messages[1]

    def test_empty_sections(self) -> None:
        assert _pack_sections([]) == []

    def test_single_oversized_section(self) -> None:
        messages = _pack_sections(["A" * 200], limit=100)
        assert len(messages) == 1


class TestFormatTelegramSections:
    def test_agent_mode(self) -> None:
        config = _make_config()
        digest = _make_agent_digest()
        sections = format_telegram_sections(digest, config)

        assert len(sections) >= 2
        full = "\n\n".join(sections)
        assert "Daily Digest" in full
        assert "Today's Pick" in full or "Attention" in full

    def test_deterministic_mode(self) -> None:
        from patronus.db import Item
        import numpy as np

        item = Item(id="1", url="https://x.com", source_type="rss", title="Article",
                     source="Blog", text="text", timestamp="2026-02-15T00:00:00Z")
        scored = ScoredItem(item=item, score=0.9, matched_topic="ml", raw_similarity=0.9)

        config = _make_config()
        digest = Digest(
            items=[DigestItem(scored_item=scored, summary="Summary.")],
            generated_at="2026-02-16T08:00:00Z",
            mode="deterministic",
        )
        sections = format_telegram_sections(digest, config)
        full = "\n\n".join(sections)
        assert "Technical AI/ML" in full
        assert "Article" in full

    def test_empty_agent_digest(self) -> None:
        config = _make_config()
        digest = Digest(sections=[], generated_at="2026-02-16T08:00:00Z", mode="agent")
        sections = format_telegram_sections(digest, config)
        full = "\n\n".join(sections)
        assert "No items" in full

    def test_empty_deterministic_digest(self) -> None:
        config = _make_config()
        digest = Digest(items=[], generated_at="2026-02-16T08:00:00Z", mode="deterministic")
        sections = format_telegram_sections(digest, config)
        full = "\n\n".join(sections)
        assert "No items" in full


class TestFormatTelegram:
    def test_joins_sections(self) -> None:
        config = _make_config()
        digest = _make_agent_digest()
        result = format_telegram(digest, config)
        assert "Daily Digest" in result
        assert isinstance(result, str)


class TestTerminalFormatItem:
    def test_agent_item(self) -> None:
        item = DigestItem(
            title="Great Paper",
            url="https://example.com",
            source="Arxiv",
            author="Alice",
            summary="Excellent work.",
        )
        text = _format_item(item)
        assert "Great Paper" in text
        assert "https://example.com" in text
        assert "Arxiv" in text
        assert "Alice" in text
        assert "Excellent work." in text

    def test_item_without_optionals(self) -> None:
        item = DigestItem(title="Title Only", url="https://x.com", summary="S")
        text = _format_item(item)
        assert "Title Only" in text
        assert "https://x.com" in text

    def test_untitled_item(self) -> None:
        item = DigestItem(url="https://x.com")
        text = _format_item(item)
        assert "Untitled" in text


class TestTerminalFormatSection:
    def test_section_with_items(self) -> None:
        section = DigestSection(
            type=SectionType.SERENDIPITY,
            title="Serendipity",
            items=[
                DigestItem(title="Interesting Read", url="https://x.com", summary="Unexpected."),
            ],
        )
        text = _format_section(section)
        assert "Serendipity" in text
        assert "Interesting Read" in text


class TestTerminalOutput:
    def test_agent_mode(self) -> None:
        config = _make_config()
        digest = _make_agent_digest()
        output = TerminalOutput()

        with patch("builtins.print") as mock_print:
            output.send(digest, config)
            mock_print.assert_called_once()
            printed = mock_print.call_args[0][0]
            assert "DAILY DIGEST" in printed
            assert "agent" in printed
            assert "Items: 3" in printed

    def test_deterministic_mode(self) -> None:
        from patronus.db import Item

        item = Item(id="1", url="https://x.com", source_type="rss", title="Article",
                     source="Blog", text="text", timestamp="2026-02-15T00:00:00Z")
        scored = ScoredItem(item=item, score=0.9, matched_topic="ml", raw_similarity=0.9)

        config = _make_config()
        digest = Digest(
            items=[DigestItem(scored_item=scored, summary="Summary.")],
            generated_at="2026-02-16T08:00:00Z",
            mode="deterministic",
        )
        output = TerminalOutput()

        with patch("builtins.print") as mock_print:
            output.send(digest, config)
            printed = mock_print.call_args[0][0]
            assert "DAILY DIGEST" in printed
            assert "deterministic" in printed
            assert "Article" in printed

    def test_empty_digest(self) -> None:
        config = _make_config()
        digest = Digest(generated_at="2026-02-16T08:00:00Z")
        output = TerminalOutput()

        with patch("builtins.print") as mock_print:
            output.send(digest, config)
            printed = mock_print.call_args[0][0]
            assert "No items" in printed
