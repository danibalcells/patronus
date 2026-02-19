from __future__ import annotations

import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

import patronus.llm as llm

_client: Optional[anthropic.Anthropic] = None
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DIGEST_SUMMARY_MODEL = "google/gemini-3-flash-preview"

_SYSTEM_PROMPT = (
    "You write very short summaries (2 sentences max, ~40 words total) of pre-curated articles "
    "for a daily reading digest. Say what the piece is about and why it's relevant to the reader's interests. "
    "No filler, no preamble, no promotional language."
    "Don't start with 'This article is about...' or 'The article discusses...' but rather go straight to the point."
    "Don't structure your response in paragraphs, bullets or titles, just write a single paragraph."
)

_DIGEST_SYSTEM_PROMPT = (
    "You write an ultra-condensed tagline summary of a daily reading digest. "
    "List the key topics as short comma-separated fragments, like: "
    "\"New Claude release, Zvi on the Pentagon, when to think without words, ...\". "
    "No sentences, no preamble, no filler. Just the fragments, ending with '...'."
    ""
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


def summarize_digest(
    items: list[tuple[str, str]],
    *,
    model: str = _DIGEST_SUMMARY_MODEL,
) -> str:
    items_text = "\n".join(f"- {title}: {summary}" for title, summary in items)
    return llm.complete(
        model,
        system=_DIGEST_SYSTEM_PROMPT,
        user_message=f"Today's digest items:\n\n{items_text}",
        max_tokens=1000,
    )
