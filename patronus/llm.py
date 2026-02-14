from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

_anthropic_client: Optional[object] = None
_google_client: Optional[object] = None
_openai_client: Optional[object] = None


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
