from __future__ import annotations

import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

_client: Optional[anthropic.Anthropic] = None
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

_SYSTEM_PROMPT = (
    "You write very short summaries (2 sentences max, ~40 words total) of pre-curated articles "
    "for a daily reading digest. Say what the piece is about and why it's relevant to the reader's interests. "
    "No filler, no preamble, no promotional language."
    "Don't start with 'This article is about...' or 'The article discusses...' but rather go straight to the point."
    "Don't structure your response in paragraphs, bullets or titles, just write a single paragraph."
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
        max_tokens=128,
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
