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
    complete,
    complete_with_tools,
)
from patronus.observability import agent_run, iteration_span, llm_generation, planning_generation, tool_call
from patronus.output.terminal import format_digest
from patronus.tools import ToolRegistry

logger = logging.getLogger(__name__)


ITERATION_PLANNING_PROMPT = """\
You are the editorial agent for Patronus. You are mid-process assembling a daily digest. Before each round of tool calls, briefly reflect on where you are and what to do next.

## If this is the first iteration (no searches yet)

Start by brainstorming from the reader's context before planning any searches. Ask yourself:
- Given what the reader has been working on and thinking about, what specific topics, papers, events, or discussions would be most valuable to find?
- Are there any named projects, themes, researchers, or ideas in their context that suggest concrete search angles?
- What would make today's digest feel personally relevant rather than generic?

List 3-5 concrete hypotheses — specific things you'd hope to find. Then plan your searches to test them.

## On subsequent iterations

Your reflection should cover:

1. **What you've found so far** — any strong candidates? notable gaps? anything surprising?
2. **What to do this round** — which tool(s) and what query/parameters, and why. Be specific.
3. **Readiness check** — do you have enough to submit a good digest, or do you need more searches?

## Key principles

- **Let structure emerge.** Don't commit to a section lineup before you've searched. The digest shape should follow from what's actually available, not from an upfront plan.
- **Adjust as you go.** Each round reveals more about the content pool. If a section type isn't yielding good content, drop it rather than forcing it.
- **Know when to stop.** If you have strong candidates across a few sections, a tight digest beats a bloated one. Don't search indefinitely.
- **Avoid redundancy.** If you've already searched a topic or tool+query combination, don't repeat it — move on to unexplored angles.

Be concise. Name specific tools and queries. This is internal reasoning only.\
"""


SYSTEM_PROMPT = """\
You are the editorial agent for Patronus, a personal research and reading assistant. Your job is to assemble a daily digest — a curated selection of the most interesting and relevant content for the reader, structured like a newspaper.

## Your role

You are a knowledgeable, opinionated editor. You know the reader's current interests and intellectual activity (provided as context). You decide what's worth their attention today. You are concise, direct, and never waste the reader's time.

## Section types

Your digest is organized into sections. Not all sections appear every day — the structure should emerge from what you find, not from a plan decided upfront. The section types are:

- **long_form_pick**: "If you read one thing today, read this." One standout pick with a 2-3 sentence summary explaining why it matters and how it connects to the reader's current thinking. Include 2 additional items with shorter (1-2 sentence) summaries. Total: 1 featured pick + 2 others.
- **paper_roundup**: A bullet list of recent papers, each with one line. More items are fine — aim for 4-8. Breadth over depth.
- **headlines**: What's happening right now — awareness-level bullets. No expectation to click through. Tech news, AI policy, notable events. Items must be recent (published within the last 7 days). Do not include older content here regardless of how relevant it is. Aim for 4-8 items, one sentence each. More is fine as long as each is genuinely noteworthy.
- **serendipity**: Something outside the reader's current bubble. Culture, language, philosophy, unexpected long-reads. Exactly 2 items with 1-2 sentence summaries each.
- **chatter**: What people are talking about — interesting discussions, debates, or takes from Twitter/blogs/communities. More tweets are fine as long as each fits in 1-2 lines. Aim for 3-6 items.
- **from_notes**: Personal notes, journal entries, or reviews from the reader's Notion that connect to today's content. Use the `search_notion` tool to find these. Only include this section if you found something genuinely worth surfacing — a past note that illuminates a current article, a review that ties into something in the feed. 2-4 items max, each with a 1-2 sentence note on *why* it's relevant right now.

## Guidelines

- Use the retrieval tools to explore what's available. You can search by similarity, recency, topic, or source.
- Use `search_notion` to check whether the reader has written anything relevant to the strongest items you've found. Search with specific concept queries, not broad topic words.
- Use the reader's context to decide what's most relevant RIGHT NOW — not just generally interesting.
- Each item needs a summary appropriate to its section type:
  - long_form_pick: 2-3 sentences for the featured pick; 1-2 sentences for the others
  - paper_roundup: One line per paper
  - headlines: One sentence, awareness-level
  - serendipity: 1-2 sentences on why this is a worthwhile detour
  - chatter: 1-2 lines on the discussion/take
  - from_notes: 1-2 sentences on why this past note is relevant to something in today's feed
- All sections (paper_roundup, headlines, chatter, from_notes) should be presented as bullet lists — one item per bullet.
- Don't include items that are very similar to each other. Diversity matters.
- If there isn't enough good content for a section type, skip it entirely. A digest with 3 excellent sections beats 5 mediocre ones.
- **Recency matters.** Each item has a `Date` field — use it. Old content (more than 7 days) must not appear in `headlines`. If an older article is genuinely worth surfacing, place it in `long_form_pick` or `serendipity` and frame it as a recommendation, not as news.
- When you've finished exploring and have strong candidates, call submit_digest with the final digest.

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
                            "enum": ["long_form_pick", "paper_roundup", "headlines", "serendipity", "chatter", "from_notes"],
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


def _build_iteration_planning_message(
    iteration: int,
    max_iterations: int,
    search_history: list[str],
    context_prose: str,
) -> str:
    history_str = "\n".join(f"  - {s}" for s in search_history) if search_history else "  None yet."
    return (
        f"Reader context:\n{context_prose}\n\n"
        f"---\n\n"
        f"Iteration {iteration + 1} of {max_iterations}.\n\n"
        f"Searches completed so far:\n{history_str}\n\n"
        "Reflect on what you've found and decide what to do in this iteration."
    )


def plan_and_assemble(
    config: Config,
    context: Context,
    tool_registry: ToolRegistry,
) -> Digest:
    agent_config = config.agent
    if agent_config is None:
        raise ValueError("AgentConfig is required for agent mode.")

    planning_model = agent_config.planning_model or agent_config.model
    all_tools = tool_registry.get_definitions() + [SUBMIT_DIGEST_TOOL]

    today_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    initial_user_message = (
        f"Today is {today_str}.\n\n"
        "Here is the reader's current context — their recent intellectual activity, "
        "interests, and what they've been working on:\n\n"
        f"{context.prose}\n\n"
        "Explore the content pool using the available search tools and assemble a digest. "
        "Let the structure emerge from what you find — don't commit to a section lineup "
        "before searching. Call submit_digest when you have strong candidates."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_user_message},
    ]
    search_history: list[str] = []

    with agent_run(
        "agent-digest-run",
        {
            "system_prompt": SYSTEM_PROMPT,
            "iteration_planning_prompt": ITERATION_PLANNING_PROMPT,
            "context_prose": context.prose,
            "model": agent_config.model,
            "planning_model": planning_model,
        },
    ) as run_obs:
        for iteration in range(agent_config.max_iterations):
            logger.info("Agent iteration %d/%d", iteration + 1, agent_config.max_iterations)

            with iteration_span(f"iteration-{iteration + 1}") as iter_obs:
                planning_user_message = _build_iteration_planning_message(
                    iteration, agent_config.max_iterations, search_history, context.prose,
                )
                with planning_generation(
                    "planning",
                    planning_model,
                    [
                        {"role": "system", "content": ITERATION_PLANNING_PROMPT},
                        {"role": "user", "content": planning_user_message},
                    ],
                ) as plan_obs:
                    thought = complete(
                        planning_model,
                        system=ITERATION_PLANNING_PROMPT,
                        user_message=planning_user_message,
                        max_tokens=512,
                    )
                    logger.debug("Iteration %d thought:\n%s", iteration + 1, thought)
                    plan_obs.update(output=thought)

                messages.append({
                    "role": "user",
                    "content": (
                        f"[Reflection, iteration {iteration + 1}/{agent_config.max_iterations}]\n\n"
                        f"{thought}"
                    ),
                })

                with llm_generation(
                    "llm-call",
                    agent_config.model,
                    [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                ) as gen_obs:
                    response: LLMResponse = complete_with_tools(
                        agent_config.model,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                        tools=all_tools,
                        max_tokens=agent_config.max_tokens,
                    )
                    gen_obs.update(output={
                        "text": response.text,
                        "tool_calls": [{"name": tc.name, "input": tc.input} for tc in response.tool_calls],
                        "stop_reason": response.stop_reason,
                    })

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
                        digest_summary = {
                            "sections": len(digest.sections),
                            "items": digest.item_count,
                            "section_types": [s.type.value for s in digest.sections],
                            "iterations_used": iteration + 1,
                        }
                        iter_obs.update(output=digest_summary)
                        run_obs.update(output=format_digest(digest))
                        return digest
                    except Exception:
                        logger.exception("Failed to parse submit_digest input")
                        messages.append(build_assistant_message_from_response(response))
                        messages.append(build_tool_result_message(
                            [submit_call],
                            {submit_call.id: "Error: invalid digest format. Please try again with valid section types and item data."},
                        ))
                        iter_obs.update(output={"error": "invalid_digest_format"})
                        continue

                if not retrieval_calls and response.stop_reason == "end_turn":
                    logger.warning("Agent stopped without calling submit_digest.")
                    iter_obs.update(output={"error": "stopped_without_submit"})
                    run_obs.update(output={"error": "stopped_without_submit"})
                    return Digest(
                        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        mode="agent",
                    )

                results: dict[str, str] = {}
                for tc in retrieval_calls:
                    logger.info("Executing tool: %s(%s)", tc.name, json.dumps(tc.input, default=str)[:200])
                    with tool_call(tc.name, tc.input) as tc_obs:
                        result = tool_registry.execute(tc.name, **tc.input)
                        tc_obs.update(output=result.to_text()[:2000])
                    results[tc.id] = result.to_text()
                    search_history.append(
                        f"{tc.name}({json.dumps(tc.input, default=str)[:120]}) → "
                        f"{result.to_text()[:120].strip()}"
                    )
                    logger.debug("Tool result: %s", result.to_text()[:600])

                messages.append(build_assistant_message_from_response(response))
                messages.append(build_tool_result_message(response.tool_calls, results))

                iter_obs.update(output={
                    "thought": thought,
                    "tool_calls_made": [tc.name for tc in retrieval_calls],
                })

        logger.warning("Agent hit max iterations (%d) without submitting digest", agent_config.max_iterations)
        run_obs.update(output={"error": "max_iterations_reached", "iterations_used": agent_config.max_iterations})
        return Digest(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            mode="agent",
        )
