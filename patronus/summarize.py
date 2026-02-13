from __future__ import annotations

import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

_client: Optional[anthropic.Anthropic] = None
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

_SYSTEM_PROMPT = (
    "You generate concise 2-3 sentence summaries of articles for a personalized "
    "reading digest. Focus on why the article is relevant to the reader's specific "
    "interests. Be direct and informative, not promotional."
)


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def summarize_item(
    title: str,
    text: str,
    interest_description: str,
    *,
    model: str = _DEFAULT_MODEL,
) -> str:
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Reader's interest area:\n{interest_description}\n\n"
                    f"Article title: {title}\n\n"
                    f"Article text (may be truncated):\n{text[:10000]}"
                ),
            }
        ],
    )
    return response.content[0].text
