from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

_anthropic_client: Optional[object] = None
_google_client: Optional[object] = None
_openai_client: Optional[object] = None


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


def complete(
    model: str,
    *,
    system: str = "",
    user_message: str,
    max_tokens: int = 4096,
) -> str:
    provider, model_name = model.split("/", 1)

    if provider == "anthropic":
        return _complete_anthropic(model_name, system=system, user_message=user_message, max_tokens=max_tokens)
    elif provider == "google":
        return _complete_google(model_name, system=system, user_message=user_message, max_tokens=max_tokens)
    elif provider == "openai":
        return _complete_openai(model_name, system=system, user_message=user_message, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def complete_with_tools(
    model: str,
    *,
    system: str = "",
    messages: list[dict[str, Any]],
    tools: list[dict],
    max_tokens: int = 4096,
) -> LLMResponse:
    provider, model_name = model.split("/", 1)

    if provider == "anthropic":
        return _complete_with_tools_anthropic(
            model_name, system=system, messages=messages, tools=tools, max_tokens=max_tokens,
        )
    elif provider == "openai":
        return _complete_with_tools_openai(
            model_name, system=system, messages=messages, tools=tools, max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Provider '{provider}' does not support tool use yet.")


def build_tool_result_message(tool_calls: list[ToolCall], results: dict[str, str]) -> dict[str, Any]:
    content = []
    for tc in tool_calls:
        content.append({
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": results.get(tc.id, ""),
        })
    return {"role": "user", "content": content}


def build_assistant_message_from_response(response: LLMResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for tc in response.tool_calls:
        content.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.input,
        })
    return {"role": "assistant", "content": content}


def _complete_anthropic(model: str, *, system: str, user_message: str, max_tokens: int) -> str:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        load_dotenv()
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system:
        kwargs["system"] = system

    response = _anthropic_client.messages.create(**kwargs)
    return response.content[0].text


def _complete_with_tools_anthropic(
    model: str,
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict],
    max_tokens: int,
) -> LLMResponse:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        load_dotenv()
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
    }
    if system:
        kwargs["system"] = system

    response = _anthropic_client.messages.create(**kwargs)

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                input=block.input,
            ))

    stop = "tool_use" if response.stop_reason == "tool_use" else "end_turn"

    return LLMResponse(
        text="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        stop_reason=stop,
    )


def _complete_google(model: str, *, system: str, user_message: str, max_tokens: int) -> str:
    global _google_client
    from google import genai
    from google.genai import types

    if _google_client is None:
        load_dotenv()
        _google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    config_kwargs: dict = {"max_output_tokens": max_tokens}
    if system:
        config_kwargs["system_instruction"] = system

    response = _google_client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text


def _complete_openai(model: str, *, system: str, user_message: str, max_tokens: int) -> str:
    global _openai_client
    from openai import OpenAI

    if _openai_client is None:
        load_dotenv()
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    response = _openai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content


def _complete_with_tools_openai(
    model: str,
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict],
    max_tokens: int,
) -> LLMResponse:
    global _openai_client
    import json
    from openai import OpenAI

    if _openai_client is None:
        load_dotenv()
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    oai_messages = _convert_messages_to_openai(messages, system)
    oai_tools = _convert_tools_to_openai(tools)

    response = _openai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=oai_messages,
        tools=oai_tools if oai_tools else None,
    )

    choice = response.choices[0]
    text = choice.message.content
    tool_calls: list[ToolCall] = []

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments),
            ))

    stop = "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"

    return LLMResponse(text=text, tool_calls=tool_calls, stop_reason=stop)


def _convert_messages_to_openai(messages: list[dict[str, Any]], system: str = "") -> list[dict[str, Any]]:
    import json
    oai: list[dict[str, Any]] = []
    if system:
        oai.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for tr in tool_results:
                    oai.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr.get("content", ""),
                    })
                continue

        if role == "assistant" and isinstance(content, list):
            text_parts = []
            tc_list = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tc_list.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    })
            oai_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                oai_msg["content"] = "\n".join(text_parts)
            else:
                oai_msg["content"] = None
            if tc_list:
                oai_msg["tool_calls"] = tc_list
            oai.append(oai_msg)
            continue

        oai.append({"role": role, "content": content if isinstance(content, str) else str(content)})

    return oai


def _convert_tools_to_openai(tools: list[dict]) -> list[dict]:
    oai_tools = []
    for t in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return oai_tools
