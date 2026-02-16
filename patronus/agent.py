from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from patronus.config import Config
from patronus.context import Context
from patronus.digest import Digest, DigestItem, DigestSection, SectionType
from patronus.llm import (
    LLMResponse,
    build_assistant_message_from_response,
    build_tool_result_message,
    complete_with_tools,
)
from patronus.tools import ToolRegistry

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are the editorial agent for Patronus, a personal research and reading assistant. Your job is to plan and assemble a daily digest — a curated selection of the most interesting and relevant content for the reader, structured like a newspaper.

## Your role

You are a knowledgeable, opinionated editor. You know the reader's current interests and intellectual activity (provided as context below). You decide what's worth their attention today. You are concise, direct, and never waste the reader's time.

## Section types

Your digest is organized into sections. Not all sections appear every day — you decide based on what's available and what's relevant. The section types are:

- **long_form_pick**: "If you read one thing today, read this." The single best item — deserves a 2-3 sentence summary explaining why it matters and how it connects to the reader's current thinking. Max 1-2 items.
- **paper_roundup**: A list of recent papers with one-line descriptions. More items, less detail per item. Especially relevant during conference season. Typically 2-5 items.
- **headlines**: What's happening — awareness-level items. No expectation to click through. Tech news, AI policy, notable events. Typically 2-4 items, one sentence each.
- **serendipity**: Something outside the reader's current bubble. Culture, language, philosophy, unexpected long-reads. The item that makes the digest feel alive. 1-2 items.
- **chatter**: What people are talking about — interesting discussions, debates, or takes from Twitter/blogs/communities. 1-3 items.

## Guidelines

- Aim for ~10 items total across all sections. Quality over quantity.
- Use the retrieval tools to explore what's available. You can search by similarity, recency, topic, or source.
- Use the reader's context to decide what's most relevant RIGHT NOW — not just generally interesting.
- Each item needs a summary appropriate to its section type:
  - long_form_pick: 2-3 sentences explaining relevance to current thinking
  - paper_roundup: One line per paper
  - headlines: One sentence, awareness-level
  - serendipity: 1-2 sentences on why this is a worthwhile detour
  - chatter: 1-2 sentences on the discussion/debate
- Don't include items that are very similar to each other. Diversity matters.
- If there isn't enough good content for a section, skip that section entirely. A digest with 3 excellent sections beats 5 mediocre ones.
- When you've finished exploring and selecting items, call the submit_digest tool with your final digest.

## Workflow

1. Read the reader's context to understand their current interests and activity.
2. Use search tools to explore the available content pool. Start broad, then narrow.
3. Mentally plan which sections make sense today.
4. Select items for each section, noting their IDs from the tool results.
5. Write summaries for each item.
6. Call submit_digest with the complete structured digest.

IMPORTANT: You must call submit_digest exactly once to deliver the final digest. Do not output the digest as plain text.\
"""


SUBMIT_DIGEST_TOOL: dict[str, Any] = {
    "name": "submit_digest",
    "description": (
        "Submit the final assembled digest. Call this exactly once when you have "
        "finished selecting items and writing summaries for all sections. "
        "This terminates the digest generation process."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "description": "The digest sections, in display order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["long_form_pick", "paper_roundup", "headlines", "serendipity", "chatter"],
                            "description": "The section type.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Display title for this section (e.g., 'Today\\'s Pick', 'Paper Roundup', 'Headlines').",
                        },
                        "items": {
                            "type": "array",
                            "description": "Items in this section.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "item_id": {
                                        "type": "string",
                                        "description": "The item ID from tool results. Use the exact ID returned by search tools.",
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": "Display title for this item.",
                                    },
                                    "url": {
                                        "type": "string",
                                        "description": "URL of the item.",
                                    },
                                    "source": {
                                        "type": "string",
                                        "description": "Source or feed name.",
                                    },
                                    "author": {
                                        "type": "string",
                                        "description": "Author name, if known.",
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "Your editorial summary for this item, appropriate to the section type.",
                                    },
                                },
                                "required": ["item_id", "title", "url", "summary"],
                            },
                        },
                    },
                    "required": ["type", "title", "items"],
                },
            },
        },
        "required": ["sections"],
    },
}


def _parse_submit_digest(input_data: dict) -> Digest:
    sections: list[DigestSection] = []

    for section_data in input_data.get("sections", []):
        section_type = SectionType(section_data["type"])
        items: list[DigestItem] = []

        for item_data in section_data.get("items", []):
            items.append(DigestItem(
                item_id=item_data.get("item_id", ""),
                title=item_data.get("title", ""),
                url=item_data.get("url", ""),
                source=item_data.get("source", ""),
                author=item_data.get("author", ""),
                summary=item_data.get("summary", ""),
            ))

        sections.append(DigestSection(
            type=section_type,
            title=section_data.get("title", section_type.value),
            items=items,
        ))

    return Digest(
        sections=sections,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="agent",
    )


def plan_and_assemble(
    config: Config,
    context: Context,
    tool_registry: ToolRegistry,
) -> Digest:
    agent_config = config.agent
    if agent_config is None:
        raise ValueError("AgentConfig is required for agent mode.")

    all_tools = tool_registry.get_definitions() + [SUBMIT_DIGEST_TOOL]

    user_message = (
        "Here is the reader's current context — their recent intellectual activity, "
        "interests, and what they've been working on:\n\n"
        f"{context.prose}\n\n"
        "Please plan and assemble today's digest. Use the search tools to explore "
        "available content, then call submit_digest with the final result."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    for iteration in range(agent_config.max_iterations):
        logger.info("Agent iteration %d/%d", iteration + 1, agent_config.max_iterations)

        response: LLMResponse = complete_with_tools(
            agent_config.model,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=all_tools,
            max_tokens=agent_config.max_tokens,
        )

        submit_call = None
        retrieval_calls = []

        for tc in response.tool_calls:
            if tc.name == "submit_digest":
                submit_call = tc
            else:
                retrieval_calls.append(tc)

        if submit_call is not None:
            logger.info("Agent submitted digest on iteration %d", iteration + 1)
            try:
                digest = _parse_submit_digest(submit_call.input)
                logger.info(
                    "Digest has %d sections, %d total items",
                    len(digest.sections),
                    digest.item_count,
                )
                return digest
            except Exception:
                logger.exception("Failed to parse submit_digest input")
                messages.append(build_assistant_message_from_response(response))
                messages.append(build_tool_result_message(
                    [submit_call],
                    {submit_call.id: "Error: invalid digest format. Please try again with valid section types and item data."},
                ))
                continue

        if not retrieval_calls and response.stop_reason == "end_turn":
            logger.warning("Agent stopped without calling submit_digest. Attempting to parse text response.")
            return Digest(
                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                mode="agent",
            )

        results: dict[str, str] = {}
        for tc in retrieval_calls:
            logger.info("Executing tool: %s(%s)", tc.name, json.dumps(tc.input, default=str)[:200])
            result = tool_registry.execute(tc.name, **tc.input)
            results[tc.id] = result.to_text()

        messages.append(build_assistant_message_from_response(response))
        messages.append(build_tool_result_message(response.tool_calls, results))

    logger.warning("Agent hit max iterations (%d) without submitting digest", agent_config.max_iterations)
    return Digest(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="agent",
    )
