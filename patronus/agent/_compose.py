from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from patronus.agent._prompts import COMPOSE_SYSTEM_PROMPT, SUBMIT_DIGEST_TOOL
from patronus.config import Config
from patronus.digest import Digest, DigestItem, DigestSection, SectionType
from patronus.llm import (
    LLMResponse,
    build_assistant_message_from_response,
    build_tool_result_message,
    complete_with_tools,
)
from patronus.observability import llm_generation, tool_call

logger = logging.getLogger(__name__)

_COMPOSE_MAX_RETRIES = 2


def _model(config: Config) -> str:
    agent_cfg = config.agent
    assert agent_cfg is not None
    return agent_cfg.compose_model or agent_cfg.assembly_model or agent_cfg.model


def _max_tokens(config: Config) -> int:
    agent_cfg = config.agent
    return agent_cfg.max_tokens if agent_cfg else 4096


def _parse_submit_digest(input_data: dict[str, Any]) -> Digest:
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
                published_date=item_data.get("published_date", ""),
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


def compose_digest(
    config: Config,
    reader_context: str,
    angles: str,
    news_output: str,
    chatter_output: str,
    research_output: str,
    threads_output: str,
) -> Digest:
    model = _model(config)
    logger.info("Step 4: composing digest (model=%s)", model)

    today_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    user_message = (
        f"Today is {today_str}.\n\n"
        f"## Reader context\n\n{reader_context}\n\n"
        f"## Editorial angles\n\n{angles}\n\n"
        f"## News filter\n\n{news_output}\n\n"
        f"## Chatter summary\n\n{chatter_output}\n\n"
        f"## Research scout\n\n{research_output}\n\n"
        f"## Thread puller\n\n{threads_output}\n\n"
        "Compose the final digest using submit_digest."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for attempt in range(_COMPOSE_MAX_RETRIES + 1):
        logger.info("Compose attempt %d/%d", attempt + 1, _COMPOSE_MAX_RETRIES + 1)

        with llm_generation(
            f"compose-attempt{attempt + 1}",
            model,
            {"system": COMPOSE_SYSTEM_PROMPT, "messages": messages},
        ) as obs:
            response: LLMResponse = complete_with_tools(
                model,
                system=COMPOSE_SYSTEM_PROMPT,
                messages=messages,
                tools=[SUBMIT_DIGEST_TOOL],
                max_tokens=_max_tokens(config),
            )
            obs.update(output={
                "text": response.text,
                "tool_calls": [{"name": tc.name, "input": tc.input} for tc in response.tool_calls],
                "stop_reason": response.stop_reason,
            })

        submit_call = next((tc for tc in response.tool_calls if tc.name == "submit_digest"), None)

        if submit_call is not None:
            with tool_call("submit_digest", submit_call.input) as tc_obs:
                try:
                    digest = _parse_submit_digest(submit_call.input)
                    tc_obs.update(output={
                        "sections": len(digest.sections),
                        "items": digest.item_count,
                        "section_types": [s.type.value for s in digest.sections],
                    })
                    logger.info(
                        "Digest composed: %d sections, %d items",
                        len(digest.sections),
                        digest.item_count,
                    )
                    return digest
                except Exception:
                    logger.exception("Failed to parse submit_digest input (attempt %d)", attempt + 1)
                    tc_obs.update(output={"error": "parse_failed"})
                    messages.append(build_assistant_message_from_response(response))
                    messages.append(build_tool_result_message(
                        [submit_call],
                        {submit_call.id: "Error: invalid digest format. Check section types and required item fields, then try again."},
                    ))
        else:
            logger.warning("Compose step did not call submit_digest (attempt %d)", attempt + 1)
            if response.text:
                messages.append(build_assistant_message_from_response(response))
                messages.append({"role": "user", "content": "You must call submit_digest to deliver the digest."})

    logger.error("Compose step failed to produce a valid digest after %d attempts", _COMPOSE_MAX_RETRIES + 1)
    return Digest(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="agent",
    )
