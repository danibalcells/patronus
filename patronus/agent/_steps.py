from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

from patronus.agent._prompts import (
    ANGLES_SYSTEM_PROMPT,
    CHATTER_SYSTEM_PROMPT,
    NEWS_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    THREADS_SYSTEM_PROMPT,
)
from patronus.config import Config
from patronus.db import Item
from patronus.llm import (
    LLMResponse,
    build_assistant_message_from_response,
    build_tool_result_message,
    complete,
    complete_structured,
    complete_with_tools,
)
from patronus.observability import llm_generation, tool_call
from patronus.tools import ToolRegistry


class NewsItem(BaseModel):
    item_id: str
    cross_ref: Literal["research", "threads", "none"] = "none"


class NewsFilterResult(BaseModel):
    items: list[NewsItem]

logger = logging.getLogger(__name__)

_RESEARCH_MAX_ITERATIONS = 2
_THREADS_MAX_ITERATIONS = 3


def _model(config: Config, field: str) -> str:
    agent_cfg = config.agent
    assert agent_cfg is not None
    return getattr(agent_cfg, field, "") or agent_cfg.model


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y")


def identify_angles(config: Config, inventory: str, reader_context: str) -> str:
    model = _model(config, "angles_model")
    user_message = (
        f"Today is {_today_str()}.\n\n"
        f"## Reader context\n\n{reader_context}\n\n"
        f"## Content inventory\n\n{inventory}\n\n"
        "Identify the editorial angles for today's digest."
    )
    logger.info("Step 2: identifying angles (model=%s)", model)
    with llm_generation(
        "angles",
        model,
        {"system": ANGLES_SYSTEM_PROMPT, "user": user_message},
    ) as obs:
        result = complete(
            model,
            system=ANGLES_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=2048,
        )
        obs.update(output=result)
    logger.debug("Angles output:\n%s", result[:500])
    return result


def filter_news(
    config: Config,
    inventory: str,
    reader_context: str,
    angles: str,
    items_by_short_id: dict[str, Item],
) -> str:
    model = _model(config, "news_model")
    user_message = (
        f"Today is {_today_str()}.\n\n"
        f"## Reader context\n\n{reader_context}\n\n"
        f"## Editorial angles\n\n{angles}\n\n"
        f"## Content inventory\n\n{inventory}\n\n"
        "Select today's news items."
    )
    logger.info("Step 3a: filtering news (model=%s)", model)
    with llm_generation(
        "news-filter",
        model,
        {"system": NEWS_SYSTEM_PROMPT, "user": user_message},
    ) as obs:
        result = complete_structured(
            model,
            system=NEWS_SYSTEM_PROMPT,
            user_message=user_message,
            schema=NewsFilterResult,
            max_tokens=_model_max_tokens(config),
        )
        obs.update(output=_news_filter_to_markdown(result, items_by_short_id))
    formatted = _format_news_filter(result, items_by_short_id)
    logger.debug("News filter: %d items selected", len(result.items))
    return formatted


def _format_news_filter(result: NewsFilterResult, items_by_short_id: dict[str, Item]) -> str:
    if not result.items:
        return "(No news items selected.)"
    lines = [f"Selected {len(result.items)} news items:\n"]
    for news_item in result.items:
        item = items_by_short_id.get(news_item.item_id)
        if item is None:
            logger.warning("News filter selected unknown item_id %r — skipping", news_item.item_id)
            continue
        lines.append(f"TITLE: {item.title or '(no title)'}")
        lines.append(f"URL: {item.url}")
        lines.append(f"SOURCE: {item.source or '?'}")
        lines.append(f"DATE: {item.timestamp or item.ingested_at or '?'}")
        if item.author:
            lines.append(f"AUTHOR: {item.author}")
        if news_item.cross_ref != "none":
            lines.append(f"CROSS_REF: {news_item.cross_ref}")
        if item.text:
            snippet = item.text[:400].replace("\n", " ").strip()
            if len(item.text) > 400:
                snippet += "…"
            lines.append(f"SNIPPET: {snippet}")
        lines.append("")
    return "\n".join(lines)


def _news_filter_to_markdown(result: NewsFilterResult, items_by_short_id: dict[str, Item]) -> str:
    if not result.items:
        return "*(No news items selected.)*"
    lines = [f"**{len(result.items)} items selected**\n"]
    for news_item in result.items:
        item = items_by_short_id.get(news_item.item_id)
        title = item.title if item else news_item.item_id
        source = item.source if item else "?"
        url = item.url if item else "?"
        lines.append(f"**{title}** — {source}")
        lines.append(f"<{url}>")
        if news_item.cross_ref != "none":
            lines.append(f"*→ {news_item.cross_ref}*")
        lines.append("")
    return "\n".join(lines)


def summarize_chatter(
    config: Config,
    tweet_inventory: str,
    reader_context: str,
) -> str:
    model = _model(config, "chatter_model")
    user_message = (
        f"Today is {_today_str()}.\n\n"
        f"## Reader context\n\n{reader_context}\n\n"
        f"## Tweet inventory\n\n{tweet_inventory}\n\n"
        "Summarize the Twitter/social discussions into grouped conversation clusters."
    )
    logger.info("Step 3a': summarizing chatter (model=%s)", model)
    with llm_generation(
        "chatter",
        model,
        {"system": CHATTER_SYSTEM_PROMPT, "user": user_message},
    ) as obs:
        result = complete(
            model,
            system=CHATTER_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=2048,
        )
        obs.update(output=result)
    logger.debug("Chatter output:\n%s", result[:500])
    return result


def scout_research(
    config: Config,
    reader_context: str,
    angles: str,
    tool_registry: ToolRegistry,
) -> str:
    model = _model(config, "research_model")
    logger.info("Step 3b: scouting research (model=%s, max_iter=%d)", model, _RESEARCH_MAX_ITERATIONS)

    initial_message = (
        f"Today is {_today_str()}.\n\n"
        f"## Reader context\n\n{reader_context}\n\n"
        f"## Editorial angles\n\n{angles}\n\n"
        "Curate the research section. Identify the reader's 2-3 active research threads from the "
        "context above, then search for papers specifically about those subfields. Every paper in "
        "your output must come from a tool call. Issue 3-5 parallel tool calls. You have at most "
        "2 iterations."
    )

    return _run_tool_loop(
        step_name="research-scout",
        model=model,
        system=RESEARCH_SYSTEM_PROMPT,
        initial_message=initial_message,
        tool_registry=tool_registry,
        max_iterations=_RESEARCH_MAX_ITERATIONS,
        max_tokens=_model_max_tokens(config),
    )


def pull_threads(
    config: Config,
    reader_context: str,
    angles: str,
    news_output: str,
    research_output: str,
    tool_registry: ToolRegistry,
) -> str:
    model = _model(config, "threads_model")
    logger.info("Step 3c: pulling threads (model=%s, max_iter=%d)", model, _THREADS_MAX_ITERATIONS)

    initial_message = (
        f"Today is {_today_str()}.\n\n"
        f"## Reader context\n\n{reader_context}\n\n"
        f"## Editorial angles\n\n{angles}\n\n"
        f"## News filter output (already selected — avoid duplicating)\n\n{news_output}\n\n"
        f"## Research scout output (already selected — avoid duplicating)\n\n{research_output}\n\n"
        "Find content worth the reader's time right now. Follow the angles — especially peripheral "
        "hooks and thematic connections. Issue 3-5 parallel tool calls per iteration. "
        "You have at most 3 iterations."
    )

    return _run_tool_loop(
        step_name="thread-puller",
        model=model,
        system=THREADS_SYSTEM_PROMPT,
        initial_message=initial_message,
        tool_registry=tool_registry,
        max_iterations=_THREADS_MAX_ITERATIONS,
        max_tokens=_model_max_tokens(config),
    )


def _model_max_tokens(config: Config) -> int:
    agent_cfg = config.agent
    return agent_cfg.max_tokens if agent_cfg else 4096



def _run_tool_loop(
    *,
    step_name: str,
    model: str,
    system: str,
    initial_message: str,
    tool_registry: ToolRegistry,
    max_iterations: int,
    max_tokens: int,
) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_message}]
    tool_definitions = tool_registry.get_definitions()
    outputs: list[str] = []
    response: LLMResponse | None = None

    for iteration in range(max_iterations):
        logger.info("%s: iteration %d/%d", step_name, iteration + 1, max_iterations)

        with llm_generation(
            f"{step_name}-iter{iteration + 1}",
            model,
            {"system": system, "messages": messages},
        ) as obs:
            response = complete_with_tools(
                model,
                system=system,
                messages=messages,
                tools=tool_definitions,
                max_tokens=max_tokens,
            )
            obs.update(output={
                "text": response.text,
                "tool_calls": [{"name": tc.name, "input": tc.input} for tc in response.tool_calls],
                "stop_reason": response.stop_reason,
            })

        if response.text:
            outputs.append(response.text)

        if not response.tool_calls:
            logger.info("%s: no tool calls on iteration %d — stopping", step_name, iteration + 1)
            break

        results: dict[str, str] = {}
        for tc in response.tool_calls:
            logger.info("%s: executing %s(%s)", step_name, tc.name, json.dumps(tc.input, default=str)[:150])
            with tool_call(tc.name, tc.input) as tc_obs:
                result = tool_registry.execute(tc.name, **tc.input)
                tc_obs.update(output=result.to_text()[:2000])
            results[tc.id] = result.to_text()

        messages.append(build_assistant_message_from_response(response))
        messages.append(build_tool_result_message(response.tool_calls, results))
    else:
        # Loop ran to completion without breaking — iterations were exhausted.
        # If the last response still had tool calls, the model never saw the final tool results
        # and never produced its synthesis. Make one additional call to let it finish.
        if response is not None and response.tool_calls:
            logger.info("%s: exhausted iterations with pending tool results — making final synthesis call", step_name)
            messages.append({"role": "user", "content": "You have used all your research iterations. Produce your final curated output now based on everything you found."})
            with llm_generation(f"{step_name}-synthesis", model, {"messages": messages}) as obs:
                final: LLMResponse = complete_with_tools(
                    model,
                    system=system,
                    messages=messages,
                    tools=tool_definitions,
                    max_tokens=max_tokens,
                )
                obs.update(output={"text": final.text, "stop_reason": final.stop_reason})
            if final.text:
                outputs.append(final.text)

    if not outputs and messages:
        all_content = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "assistant"
        )
        return all_content or f"(No output from {step_name})"

    return "\n\n".join(outputs)
