from __future__ import annotations

import json
import logging
import os
import urllib.request
from urllib.error import HTTPError, URLError

from patronus.config import Config
from patronus.digest import Digest
from patronus.output.feed import format_digest_html

logger = logging.getLogger(__name__)

_API_URL = "https://readwise.io/api/v3/save/"


class ReaderOutput:
    def send(self, digest: Digest, config: Config) -> None:
        token = os.environ.get("READWISE_TOKEN")
        if not token:
            logger.warning("READWISE_TOKEN not set, skipping Reader output")
            return

        date_str = (digest.generated_at or "")[:10]
        url = f"https://patronus.invalid/digest/{date_str.replace('-', '')}"
        title = f"Patronus Daily Digest \u2014 {date_str}"
        html_content = format_digest_html(digest)

        payload = json.dumps(
            {
                "url": url,
                "title": title,
                "html": html_content,
                "location": "feed",
                "saved_using": "patronus",
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
