from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
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


class Phase(Enum):
    SCAN = "scan"
    DEEP_DIVE = "deep_dive"
    ASSEMBLY = "assembly"


def get_phase(iteration: int, max_iterations: int) -> Phase:
    scan_end = max_iterations // 3
    deep_dive_end = 2 * max_iterations // 3
    if iteration < scan_end:
        return Phase.SCAN
    elif iteration < deep_dive_end:
        return Phase.DEEP_DIVE
    else:
        return Phase.ASSEMBLY


SCAN_PLANNING_PROMPT = """\
You are in the SCAN phase of digest assembly. Your goal is to build a mental map of what's in the content pool — not to select items or commit to anything.

Before planning your searches, reflect:
- What does the reader's context suggest about where to look? What topics, domains, or specific themes should you probe?
- List 2-3 concrete angles for broad searches.

Then plan broad, high-N searches (aim for n=20-30) to cover:
- Recent content (last 3 and 7 days)
- Semantic similarity to the reader's interests (multiple queries)
- Specific source types (e.g., Twitter/social chatter, RSS feeds)
- Any external sources (OpenAlex, Arxiv, Notion) if the reader's context points to specific domains

Do NOT think about sections or summaries yet. Just explore and observe. Be concise — name specific tools and queries.\
"""


DEEP_DIVE_PLANNING_PROMPT = """\
You are in the DEEP DIVE phase of digest assembly. You've completed a broad scan. Now do targeted gap-filling based on what you found.

Briefly reflect:
- What topic areas had strong content in the scan? What was thin or missing?
- What would the reader most want that you haven't found yet?
- Are there personal notes (Notion) that might connect to the strongest candidates from the scan?

Plan 1-3 targeted searches. Prefer external tools (search_arxiv, search_openalex, search_notion) when you need to fill specific gaps. Use specific concept queries, not broad topic words. Avoid repeating searches you've already done.

Be concise — name specific tools and queries.\
"""


ASSEMBLY_PLANNING_PROMPT = """\
You are in the ASSEMBLY phase of digest assembly. You've scanned broadly and filled gaps. Now make editorial decisions and produce the final digest.

Your reflection should cover:

1. **What you've found** — your strongest candidates across section types. What's the standout long-form pick? Which papers are worth a roundup? What headlines or chatter items?
2. **What to include** — finalize your selections. Drop any section where you don't have strong content. A tight digest beats a padded one.
3. **Readiness check** — if you have strong candidates across enough sections, call submit_digest now. Don't search further unless there's a specific gap you can close in one targeted query.

Be concise. If you're ready, say so and call submit_digest.\
"""


SCAN_SYSTEM_PROMPT = """\
You are the editorial agent for Patronus, a personal research and reading assistant.

## Your role right now: SCAN

Your job in this phase is to build a mental map of the content pool. Do NOT make editorial selections. Do NOT think about sections or summaries. Just explore broadly and observe what's available.

- Use broad searches with high result counts (n=20-30 where possible).
- Cover recent content, semantic similarity to the reader's interests, different source types and channels.
- You can use any available retrieval tool — local search, Arxiv, OpenAlex, Notion — if the reader's context points to specific domains.
- After your searches, your internal state should include: which topic areas have strong content, which are thin, what's surprisingly interesting.

You are building a picture. Selection and summarization come later. Do not call submit_digest — it is not available in this phase.\
"""


DEEP_DIVE_SYSTEM_PROMPT = """\
You are the editorial agent for Patronus, a personal research and reading assistant.

## Your role right now: DEEP DIVE

You've completed a broad scan. Your job now is targeted gap-filling — focused searches to find content for areas that were thin, or to enrich the strongest candidates you identified.

- Run targeted queries using search_arxiv, search_openalex, or search_notion based on what the scan revealed and what the reader's context suggests.
- Use search_notion with specific concept queries based on the strongest candidates from your scan — look for personal notes that connect.
- Use search_similar with more specific queries for underrepresented areas.
- Avoid repeating searches you've already done.

This is where external tools earn their keep: you know what you need and can formulate targeted queries. Do not call submit_digest — it is not available in this phase.\
"""


ASSEMBLY_SYSTEM_PROMPT = """\
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
- When you've finished selecting and writing summaries, call submit_digest with the final digest.

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

_PHASE_SYSTEM_PROMPTS: dict[Phase, str] = {
    Phase.SCAN: SCAN_SYSTEM_PROMPT,
    Phase.DEEP_DIVE: DEEP_DIVE_SYSTEM_PROMPT,
    Phase.ASSEMBLY: ASSEMBLY_SYSTEM_PROMPT,
}

_PHASE_PLANNING_PROMPTS: dict[Phase, str] = {
    Phase.SCAN: SCAN_PLANNING_PROMPT,
    Phase.DEEP_DIVE: DEEP_DIVE_PLANNING_PROMPT,
    Phase.ASSEMBLY: ASSEMBLY_PLANNING_PROMPT,
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
    phase: Phase,
    search_history: list[str],
    context_prose: str,
) -> str:
    history_str = "\n".join(f"  - {s}" for s in search_history) if search_history else "  None yet."
    return (
        f"Reader context:\n{context_prose}\n\n"
        f"---\n\n"
        f"Phase: {phase.value.upper().replace('_', ' ')} | Iteration {iteration + 1} of {max_iterations}.\n\n"
        f"Searches completed so far:\n{history_str}\n\n"
        "Reflect on where you are and what to do in this iteration."
    )


def _get_tools_for_phase(phase: Phase, retrieval_definitions: list[dict]) -> list[dict]:
    if phase == Phase.ASSEMBLY:
        return retrieval_definitions + [SUBMIT_DIGEST_TOOL]
    return retrieval_definitions


def plan_and_assemble(
    config: Config,
    context: Context,
    tool_registry: ToolRegistry,
) -> Digest:
    agent_config = config.agent
    if agent_config is None:
        raise ValueError("AgentConfig is required for agent mode.")

    planning_model = agent_config.planning_model or agent_config.model
    retrieval_definitions = tool_registry.get_definitions()

    today_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    initial_user_message = (
        f"Today is {today_str}.\n\n"
        "Here is the reader's current context — their recent intellectual activity, "
        "interests, and what they've been working on:\n\n"
        f"{context.prose}\n\n"
        "Explore the content pool using the available search tools. Start broad, then go deep, "
        "then assemble the digest. Let the structure emerge from what you find."
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_user_message},
    ]
    search_history: list[str] = []
    prev_phase: Phase | None = None

    with agent_run(
        "agent-digest-run",
        {
            "context_prose": context.prose,
            "model": agent_config.model,
            "planning_model": planning_model,
        },
    ) as run_obs:
        for iteration in range(agent_config.max_iterations):
            phase = get_phase(iteration, agent_config.max_iterations)

            if phase != prev_phase:
                logger.info(
                    "Phase transition: %s → %s (iteration %d/%d)",
                    prev_phase.value if prev_phase else "start",
                    phase.value,
                    iteration + 1,
                    agent_config.max_iterations,
                )
                prev_phase = phase

            system_prompt = _PHASE_SYSTEM_PROMPTS[phase]
            planning_prompt = _PHASE_PLANNING_PROMPTS[phase]
            tools_for_phase = _get_tools_for_phase(phase, retrieval_definitions)

            logger.info("Agent iteration %d/%d [%s]", iteration + 1, agent_config.max_iterations, phase.value)

            with iteration_span(f"iteration-{iteration + 1}", input={"phase": phase.value}) as iter_obs:
                planning_user_message = _build_iteration_planning_message(
                    iteration, agent_config.max_iterations, phase, search_history, context.prose,
                )
                with planning_generation(
                    "planning",
                    planning_model,
                    [
                        {"role": "system", "content": planning_prompt},
                        {"role": "user", "content": planning_user_message},
                    ],
                ) as plan_obs:
                    thought = complete(
                        planning_model,
                        system=planning_prompt,
                        user_message=planning_user_message,
                        max_tokens=512,
                    )
                    logger.debug("Iteration %d thought:\n%s", iteration + 1, thought)
                    plan_obs.update(output=thought)

                messages.append({
                    "role": "user",
                    "content": (
                        f"[{phase.value.upper().replace('_', ' ')} — iteration {iteration + 1}/{agent_config.max_iterations}]\n\n"
                        f"{thought}"
                    ),
                })

                with llm_generation(
                    "llm-call",
                    agent_config.model,
                    [{"role": "system", "content": system_prompt}] + messages,
                ) as gen_obs:
                    response: LLMResponse = complete_with_tools(
                        agent_config.model,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_for_phase,
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
                    if tc.name == "submit_digest" and phase == Phase.ASSEMBLY:
                        submit_call = tc
                    else:
                        retrieval_calls.append(tc)

                if submit_call is not None:
                    logger.info("Agent submitted digest on iteration %d [assembly]", iteration + 1)
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
                            "phase": phase.value,
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
                    logger.warning("Agent stopped without calling any tools (iteration %d, phase %s).", iteration + 1, phase.value)
                    iter_obs.update(output={"error": "stopped_without_tools"})
                    run_obs.update(output={"error": "stopped_without_tools"})
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
                    "phase": phase.value,
                    "tool_calls_made": [tc.name for tc in retrieval_calls],
                })

        logger.warning("Agent hit max iterations (%d) without submitting digest", agent_config.max_iterations)
        run_obs.update(output={"error": "max_iterations_reached", "iterations_used": agent_config.max_iterations})
        return Digest(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            mode="agent",
        )
