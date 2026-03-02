from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from patronus.config import AgentConfig, Config, DigestConfig, EmbeddingConfig, NotionConfig, PollingConfig, TelegramConfig
from patronus.context import PersonalizationSource
from patronus.db import Database
from patronus.notion import (
    NotionEntry,
    NotionSource,
    _block_to_line,
    _blocks_to_text,
    _extract_page_title,
    _rich_text_to_str,
)


def _make_config(**overrides: object) -> Config:
    defaults: dict = dict(
        digest=DigestConfig(),
        polling=PollingConfig(),
        embedding=EmbeddingConfig(),

        telegram=TelegramConfig(),
        topics={},
        agent=AgentConfig(),
        notion=NotionConfig(
            database_ids={
                "journal": "db-journal-id",
                "notes": "db-notes-id",
            },
            lookback_days=14,
            fallback_lookback_days=30,
            min_entries_threshold=3,
            max_chars_per_entry=3000,
        ),
        notion_token="secret_test_token",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_page(page_id: str, title: str, last_edited: str = "2026-02-10T00:00:00Z") -> dict:
    return {
        "id": page_id,
        "last_edited_time": last_edited,
        "properties": {
            "Name": {
                "type": "title",
                "title": [{"plain_text": title}],
            }
        },
    }


def _make_block(block_type: str, rich_text: list[dict] | None = None, **extra: object) -> dict:
    data: dict = {}
    if rich_text is not None:
        data["rich_text"] = rich_text
    data.update(extra)
    return {
        "id": f"block-{block_type}",
        "type": block_type,
        block_type: data,
        "has_children": extra.get("has_children", False),
    }


def _rt(text: str) -> list[dict]:
    return [{"plain_text": text}]


class TestPersonalizationSourceProtocol:
    def test_notion_source_is_personalization_source(self) -> None:
        assert isinstance(NotionSource(), PersonalizationSource)


class TestRichTextToStr:
    def test_simple_text(self) -> None:
        assert _rich_text_to_str([{"plain_text": "hello"}]) == "hello"

    def test_multiple_segments(self) -> None:
        result = _rich_text_to_str([
            {"plain_text": "hello "},
            {"plain_text": "world"},
        ])
        assert result == "hello world"

    def test_empty_list(self) -> None:
        assert _rich_text_to_str([]) == ""

    def test_missing_plain_text(self) -> None:
        assert _rich_text_to_str([{"type": "text"}]) == ""


class TestExtractPageTitle:
    def test_extracts_title(self) -> None:
        page = _make_page("p1", "My Journal Entry")
        assert _extract_page_title(page) == "My Journal Entry"

    def test_returns_untitled_when_no_title(self) -> None:
        page = {"properties": {"Status": {"type": "select"}}}
        assert _extract_page_title(page) == "Untitled"

    def test_returns_untitled_when_no_properties(self) -> None:
        assert _extract_page_title({}) == "Untitled"

    def test_concatenates_title_segments(self) -> None:
        page = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [
                        {"plain_text": "Part 1 "},
                        {"plain_text": "Part 2"},
                    ],
                }
            }
        }
        assert _extract_page_title(page) == "Part 1 Part 2"


class TestBlockToLine:
    def test_paragraph(self) -> None:
        assert _block_to_line("paragraph", {"rich_text": _rt("Hello world")}, 0) == "Hello world"

    def test_paragraph_empty(self) -> None:
        assert _block_to_line("paragraph", {"rich_text": []}, 0) is None

    def test_heading_1(self) -> None:
        assert _block_to_line("heading_1", {"rich_text": _rt("Title")}, 0) == "# Title"

    def test_heading_2(self) -> None:
        assert _block_to_line("heading_2", {"rich_text": _rt("Subtitle")}, 0) == "## Subtitle"

    def test_heading_3(self) -> None:
        assert _block_to_line("heading_3", {"rich_text": _rt("Section")}, 0) == "### Section"

    def test_bulleted_list_item(self) -> None:
        assert _block_to_line("bulleted_list_item", {"rich_text": _rt("item")}, 0) == "- item"

    def test_bulleted_list_item_indented(self) -> None:
        assert _block_to_line("bulleted_list_item", {"rich_text": _rt("nested")}, 2) == "    - nested"

    def test_numbered_list_item(self) -> None:
        assert _block_to_line("numbered_list_item", {"rich_text": _rt("first")}, 0) == "1. first"

    def test_to_do_unchecked(self) -> None:
        result = _block_to_line("to_do", {"rich_text": _rt("task"), "checked": False}, 0)
        assert result == "[ ] task"

    def test_to_do_checked(self) -> None:
        result = _block_to_line("to_do", {"rich_text": _rt("done task"), "checked": True}, 0)
        assert result == "[x] done task"

    def test_toggle(self) -> None:
        assert _block_to_line("toggle", {"rich_text": _rt("Details")}, 0) == "▸ Details"

    def test_quote(self) -> None:
        assert _block_to_line("quote", {"rich_text": _rt("wise words")}, 0) == "> wise words"

    def test_callout_with_emoji(self) -> None:
        data = {"rich_text": _rt("Note"), "icon": {"emoji": "💡"}}
        assert _block_to_line("callout", data, 0) == "💡 Note"

    def test_callout_without_icon(self) -> None:
        data = {"rich_text": _rt("Note"), "icon": None}
        assert _block_to_line("callout", data, 0) == "Note"

    def test_code_block(self) -> None:
        data = {"rich_text": _rt("print('hi')"), "language": "python"}
        result = _block_to_line("code", data, 0)
        assert result == "```python\nprint('hi')\n```"

    def test_equation(self) -> None:
        assert _block_to_line("equation", {"expression": "E=mc^2"}, 0) == "E=mc^2"

    def test_divider(self) -> None:
        assert _block_to_line("divider", {}, 0) == "---"

    def test_table_row(self) -> None:
        data = {"cells": [_rt("A"), _rt("B"), _rt("C")]}
        assert _block_to_line("table_row", data, 0) == "A | B | C"

    def test_bookmark_with_caption(self) -> None:
        data = {"url": "https://example.com", "caption": _rt("Example")}
        assert _block_to_line("bookmark", data, 0) == "Example: https://example.com"

    def test_bookmark_without_caption(self) -> None:
        data = {"url": "https://example.com", "caption": []}
        assert _block_to_line("bookmark", data, 0) == "https://example.com"

    def test_link_preview(self) -> None:
        assert _block_to_line("link_preview", {"url": "https://arxiv.org/123"}, 0) == "https://arxiv.org/123"

    def test_unknown_block_returns_none(self) -> None:
        assert _block_to_line("image", {}, 0) is None

    def test_column_list_returns_none(self) -> None:
        assert _block_to_line("column_list", {}, 0) is None

    def test_indentation(self) -> None:
        assert _block_to_line("paragraph", {"rich_text": _rt("deep")}, 3) == "      deep"


class TestBlocksToText:
    def test_simple_blocks(self) -> None:
        client = MagicMock()
        blocks = [
            _make_block("heading_1", _rt("Title")),
            _make_block("paragraph", _rt("Some text.")),
        ]
        result = _blocks_to_text(client, blocks)
        assert "# Title" in result
        assert "Some text." in result

    def test_recurses_into_children(self) -> None:
        client = MagicMock()
        parent = _make_block("toggle", _rt("Click me"), has_children=True)
        child = _make_block("paragraph", _rt("Hidden text"))

        client.blocks.children.list.return_value = {
            "results": [child],
            "has_more": False,
        }

        result = _blocks_to_text(client, [parent])
        assert "▸ Click me" in result
        assert "Hidden text" in result

    def test_synced_block_reference(self) -> None:
        client = MagicMock()
        synced_ref = {
            "id": "synced-ref-id",
            "type": "synced_block",
            "synced_block": {"synced_from": {"block_id": "original-block-id"}},
            "has_children": False,
        }
        original_child = _make_block("paragraph", _rt("Synced content"))

        client.blocks.children.list.return_value = {
            "results": [original_child],
            "has_more": False,
        }

        result = _blocks_to_text(client, [synced_ref])
        assert "Synced content" in result
        client.blocks.children.list.assert_called_with(block_id="original-block-id")

    def test_synced_block_original(self) -> None:
        client = MagicMock()
        synced_original = {
            "id": "synced-original-id",
            "type": "synced_block",
            "synced_block": {"synced_from": None},
            "has_children": True,
        }
        child = _make_block("paragraph", _rt("Original synced content"))

        client.blocks.children.list.return_value = {
            "results": [child],
            "has_more": False,
        }

        result = _blocks_to_text(client, [synced_original])
        assert "Original synced content" in result

    def test_handles_failed_block_fetch(self) -> None:
        client = MagicMock()
        parent = _make_block("toggle", _rt("Click me"), has_children=True)
        client.blocks.children.list.side_effect = Exception("API error")

        result = _blocks_to_text(client, [parent])
        assert "▸ Click me" in result

    def test_pagination_in_block_children(self) -> None:
        client = MagicMock()
        parent = _make_block("toggle", _rt("Paginated"), has_children=True)

        client.blocks.children.list.side_effect = [
            {
                "results": [_make_block("paragraph", _rt("Page 1"))],
                "has_more": True,
                "next_cursor": "cursor-1",
            },
            {
                "results": [_make_block("paragraph", _rt("Page 2"))],
                "has_more": False,
            },
        ]

        result = _blocks_to_text(client, [parent])
        assert "Page 1" in result
        assert "Page 2" in result

    def test_unsupported_block_no_children_produces_no_output(self) -> None:
        client = MagicMock()
        ai_block = {
            "id": "ai-block-id",
            "type": "unsupported",
            "unsupported": {},
            "has_children": False,
        }
        result = _blocks_to_text(client, [ai_block])
        assert result == ""
        client.blocks.children.list.assert_not_called()

    def test_unsupported_block_with_accessible_children_includes_text(self) -> None:
        client = MagicMock()
        ai_block = {
            "id": "ai-block-id",
            "type": "unsupported",
            "unsupported": {},
            "has_children": True,
        }
        child = _make_block("paragraph", _rt("AI meeting summary text"))
        client.blocks.children.list.return_value = {
            "results": [child],
            "has_more": False,
        }

        result = _blocks_to_text(client, [ai_block])
        assert "AI meeting summary text" in result

    def test_unsupported_block_with_inaccessible_children_does_not_raise(self) -> None:
        client = MagicMock()
        ai_block = {
            "id": "ai-block-id",
            "type": "unsupported",
            "unsupported": {},
            "has_children": True,
        }
        client.blocks.children.list.side_effect = Exception("API does not support this block type")

        result = _blocks_to_text(client, [ai_block])
        assert result == ""

    def test_unsupported_block_children_failure_logs_debug_not_warning(self, caplog) -> None:
        import logging
        client = MagicMock()
        ai_block = {
            "id": "ai-block-id",
            "type": "unsupported",
            "unsupported": {},
            "has_children": True,
        }
        client.blocks.children.list.side_effect = Exception("Block type transcription is not supported via the API.")

        with caplog.at_level(logging.DEBUG, logger="patronus.notion"):
            _blocks_to_text(client, [ai_block])

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warning_records, "Expected no warnings for unsupported block children fetch failure"

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG and "ai-block-id" in r.message]
        assert debug_records, "Expected a DEBUG log for unsupported block children fetch failure"


class TestNotionSourceGetContext:
    def test_returns_empty_when_no_notion_config(self) -> None:
        config = _make_config(notion=None)
        source = NotionSource()
        assert source.get_context(config) == ""

    def test_returns_empty_when_interest_vectors(self) -> None:
        config = _make_config()
        source = NotionSource()
        assert source.get_interest_vectors(config) is None

    @patch("patronus.notion.complete")
    def test_fetches_and_summarizes(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config()

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-id", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry 1"), _make_page("p2", "Entry 2")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Some content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Summary of activity."

        source = NotionSource(notion_client=mock_client)
        result = source.get_context(config)

        assert result == "Summary of activity."
        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args[1]
        assert "Entry 1" in call_kwargs["user_message"]
        assert "Entry 2" in call_kwargs["user_message"]

    @patch("patronus.notion.complete")
    def test_falls_back_to_longer_window(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=3,
                lookback_days=14,
                fallback_lookback_days=30,
            )
        )

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }

        call_count = 0

        def query_side_effect(**kwargs: object) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return {"results": [_make_page("p1", "Only one")], "has_more": False}
            return {
                "results": [
                    _make_page("p1", "One"),
                    _make_page("p2", "Two"),
                    _make_page("p3", "Three"),
                ],
                "has_more": False,
            }

        mock_client.data_sources.query.side_effect = query_side_effect
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Expanded summary."

        source = NotionSource(notion_client=mock_client)
        result = source.get_context(config)

        assert result == "Expanded summary."
        assert mock_client.data_sources.query.call_count == 2

    @patch("patronus.notion.complete")
    def test_returns_empty_when_below_threshold(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=5,
            )
        )

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Lonely entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [],
            "has_more": False,
        }

        source = NotionSource(notion_client=mock_client)
        result = source.get_context(config)

        assert result == ""
        mock_complete.assert_not_called()

    @patch("patronus.notion.complete")
    def test_survives_single_database_failure(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config()

        def retrieve_side_effect(database_id: str) -> dict:
            if database_id == "db-journal-id":
                raise Exception("Notion API error")
            return {"data_sources": [{"id": f"ds-{database_id}", "name": "source"}]}

        mock_client.databases.retrieve.side_effect = retrieve_side_effect
        mock_client.data_sources.query.return_value = {
            "results": [
                _make_page("p1", "Note 1"),
                _make_page("p2", "Note 2"),
                _make_page("p3", "Note 3"),
            ],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Partial summary."

        source = NotionSource(notion_client=mock_client)
        result = source.get_context(config)

        assert result == "Partial summary."

    @patch("patronus.notion.complete")
    def test_saves_to_db_when_provided(self, mock_complete: MagicMock, tmp_path: object) -> None:
        mock_client = MagicMock()
        config = _make_config()

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-id", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [
                _make_page("p1", "E1"),
                _make_page("p2", "E2"),
                _make_page("p3", "E3"),
            ],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Cached summary."

        db = Database(db_path=":memory:")
        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == "Cached summary."
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is not None
        assert snapshot.content == "Cached summary."
        assert snapshot.source_type == "notion"

    @patch("patronus.notion.complete")
    def test_truncates_entry_content(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                max_chars_per_entry=50,
                min_entries_threshold=1,
            )
        )

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry")],
            "has_more": False,
        }

        long_text = "x" * 200
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", [{"plain_text": long_text}])],
            "has_more": False,
        }
        mock_complete.return_value = "Summary."

        source = NotionSource(notion_client=mock_client)
        source.get_context(config)

        call_kwargs = mock_complete.call_args[1]
        assert len(call_kwargs["user_message"]) < 200

    @patch("patronus.notion.complete")
    def test_passes_correct_model(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            agent=AgentConfig(notion_context_model="anthropic/claude-haiku-4-5-20251001"),
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
            )
        )

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Summary."

        source = NotionSource(notion_client=mock_client)
        source.get_context(config)

        assert mock_complete.call_args[0][0] == "anthropic/claude-haiku-4-5-20251001"

    @patch("patronus.notion.complete")
    def test_handles_page_content_extraction_failure(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
            )
        )

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry 1"), _make_page("p2", "Entry 2")],
            "has_more": False,
        }

        block_call_count = 0

        def blocks_side_effect(**kwargs: object) -> dict:
            nonlocal block_call_count
            block_call_count += 1
            if kwargs.get("block_id") == "p1":
                raise Exception("Block fetch failed")
            return {
                "results": [_make_block("paragraph", _rt("Content"))],
                "has_more": False,
            }

        mock_client.blocks.children.list.side_effect = blocks_side_effect
        mock_complete.return_value = "Summary."

        source = NotionSource(notion_client=mock_client)
        result = source.get_context(config)

        assert result == "Summary."


class TestQueryDatabasePagination:
    @patch("patronus.notion.complete")
    def test_paginates_through_results(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
            )
        )

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.side_effect = [
            {
                "results": [_make_page("p1", "Page 1")],
                "has_more": True,
                "next_cursor": "cursor-1",
            },
            {
                "results": [_make_page("p2", "Page 2")],
                "has_more": False,
            },
        ]
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Summary."

        source = NotionSource(notion_client=mock_client)
        source.get_context(config)

        call_kwargs = mock_complete.call_args[1]
        assert "Page 1" in call_kwargs["user_message"]
        assert "Page 2" in call_kwargs["user_message"]

        second_query_kwargs = mock_client.data_sources.query.call_args_list[1][1]
        assert second_query_kwargs["start_cursor"] == "cursor-1"


class TestNotionContextCaching:
    @patch("patronus.notion.complete")
    def test_uses_fresh_cache_when_available(self, mock_complete: MagicMock) -> None:
        from datetime import datetime, timedelta, timezone

        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
                cache_ttl_hours=24,
            )
        )

        db = Database(db_path=":memory:")
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.save_context_snapshot("notion", "Cached context from 12 hours ago")
        
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is not None
        snapshot.generated_at = recent_time
        
        with db._session() as session:
            session.add(snapshot)
            session.commit()

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == "Cached context from 12 hours ago"
        mock_complete.assert_not_called()
        mock_client.data_sources.query.assert_not_called()

    @patch("patronus.notion.complete")
    def test_fetches_fresh_when_cache_stale(self, mock_complete: MagicMock) -> None:
        from datetime import datetime, timedelta, timezone

        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
                cache_ttl_hours=24,
            )
        )

        db = Database(db_path=":memory:")
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.save_context_snapshot("notion", "Old cached context")
        
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is not None
        snapshot.generated_at = stale_time
        
        with db._session() as session:
            session.add(snapshot)
            session.commit()

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Fresh entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Fresh content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Fresh summary from Notion"

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == "Fresh summary from Notion"
        mock_complete.assert_called_once()

    @patch("patronus.notion.complete")
    def test_fetches_fresh_when_no_cache(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
            )
        )

        db = Database(db_path=":memory:")

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Fresh summary"

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == "Fresh summary"
        mock_complete.assert_called_once()

    @patch("patronus.notion.complete")
    def test_force_refresh_bypasses_cache(self, mock_complete: MagicMock) -> None:
        from datetime import datetime, timedelta, timezone

        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
                cache_ttl_hours=24,
            )
        )

        db = Database(db_path=":memory:")
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.save_context_snapshot("notion", "Very fresh cached context")
        
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is not None
        snapshot.generated_at = recent_time
        
        with db._session() as session:
            session.add(snapshot)
            session.commit()

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "New entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("New content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Forced fresh summary"

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config, force_refresh=True)

        assert result == "Forced fresh summary"
        mock_complete.assert_called_once()

    @patch("patronus.notion.complete")
    def test_does_not_cache_empty_summary(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=10,
            )
        )

        db = Database(db_path=":memory:")

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Only one")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [],
            "has_more": False,
        }

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == ""
        mock_complete.assert_not_called()
        
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is None

    @patch("patronus.notion.complete")
    def test_uses_stale_cache_on_llm_failure(self, mock_complete: MagicMock) -> None:
        from datetime import datetime, timedelta, timezone

        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
                cache_ttl_hours=24,
            )
        )

        db = Database(db_path=":memory:")
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.save_context_snapshot("notion", "Stale but valid cached context")
        
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is not None
        snapshot.generated_at = stale_time
        
        with db._session() as session:
            session.add(snapshot)
            session.commit()

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.side_effect = Exception("LLM API down")

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == "Stale but valid cached context"
        mock_complete.assert_called_once()

    @patch("patronus.notion.complete")
    def test_returns_empty_on_llm_failure_with_no_cache(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
            )
        )

        db = Database(db_path=":memory:")

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.side_effect = Exception("LLM API down")

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == ""
        mock_complete.assert_called_once()

    @patch("patronus.notion.complete")
    def test_handles_invalid_cache_timestamp(self, mock_complete: MagicMock) -> None:
        mock_client = MagicMock()
        config = _make_config(
            notion=NotionConfig(
                database_ids={"journal": "db-j"},
                min_entries_threshold=1,
            )
        )

        db = Database(db_path=":memory:")
        db.save_context_snapshot("notion", "Cached with bad timestamp")
        
        snapshot = db.get_latest_context_snapshot("notion")
        assert snapshot is not None
        snapshot.generated_at = "not-a-timestamp"
        
        with db._session() as session:
            session.add(snapshot)
            session.commit()

        mock_client.databases.retrieve.return_value = {
            "data_sources": [{"id": "ds-j", "name": "source"}],
        }
        mock_client.data_sources.query.return_value = {
            "results": [_make_page("p1", "Entry")],
            "has_more": False,
        }
        mock_client.blocks.children.list.return_value = {
            "results": [_make_block("paragraph", _rt("Content"))],
            "has_more": False,
        }
        mock_complete.return_value = "Fresh summary"

        source = NotionSource(notion_client=mock_client, db=db)
        result = source.get_context(config)

        assert result == "Fresh summary"
        mock_complete.assert_called_once()

