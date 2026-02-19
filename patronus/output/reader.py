from __future__ import annotations

import json
import logging
import os
import urllib.request
from urllib.error import HTTPError, URLError

from patronus.config import Config
from patronus.digest import Digest
from patronus.output.feed import format_digest_html
from patronus.summarize import summarize_digest

logger = logging.getLogger(__name__)

_API_URL = "https://readwise.io/api/v3/save/"
_IMAGE_URL = "https://raw.githubusercontent.com/danibalcells/patronus/a9f2cd699ef33a741c314ef50de897ccdc2fe872/image.jpg"


class ReaderOutput:
    def send(self, digest: Digest, config: Config) -> None:
        token = os.environ.get("READWISE_TOKEN")
        if not token:
            logger.warning("READWISE_TOKEN not set, skipping Reader output")
            return

        date_str = (digest.generated_at or "")[:10]
        url = f"https://patronus.feed/digest/{date_str.replace('-', '')}"
        html_content = format_digest_html(digest)

        item_pairs = [(item.title, item.summary) for item in digest.all_items if item.title]
        digest_summary = None
        if item_pairs:
            try:
                digest_summary = summarize_digest(item_pairs)
                logger.info("Digest summary title: %s", digest_summary.title)
                logger.info("Digest summary tagline: %s", digest_summary.tagline)
            except Exception:
                logger.warning("Failed to generate digest summary", exc_info=True)

        title = f"Patronus {date_str}: {digest_summary.title}" if digest_summary else f"Patronus {date_str}"
        tagline = digest_summary.tagline if digest_summary else "A curated daily digest of research and reading."

        payload = json.dumps(
            {
                "url": url,
                "title": title,
                "html": html_content,
                "author": "Patronus",
                "category": "rss",
                "summary": tagline,
                "image_url": _IMAGE_URL,
                "location": "feed",
                "saved_using": "Patronus",
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            _API_URL,
            data=payload,
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                logger.info("Readwise Reader: saved digest (HTTP %d) — %s", status, url)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.warning(
                "Readwise Reader API error (HTTP %d): %s", exc.code, body
            )
        except URLError as exc:
            logger.warning("Readwise Reader request failed: %s", exc.reason)
