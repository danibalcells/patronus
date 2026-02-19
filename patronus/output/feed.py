from __future__ import annotations

import html
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import format_datetime as format_rfc2822
from pathlib import Path

from patronus.config import Config
from patronus.digest import Digest, DigestItem

logger = logging.getLogger(__name__)

_MAX_FEED_ITEMS = 90
_MIDDOT = " \u00b7 "

_R2_ENV_VARS = [
    "CF_R2_ACCOUNT_ID",
    "CF_R2_ACCESS_KEY_ID",
    "CF_R2_SECRET_ACCESS_KEY",
    "CF_R2_BUCKET_NAME",
]


def _get_r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['CF_R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _public_feed_url(key: str) -> str:
    base = os.environ.get("CF_R2_PUBLIC_BASE_URL") or (
        f"https://pub-{os.environ['CF_R2_ACCOUNT_ID']}.r2.dev"
    )
    return f"{base.rstrip('/')}/{key}"


def _fetch_existing_feed(client, bucket: str, key: str) -> str | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")
    except client.exceptions.NoSuchKey:
        return None
    except Exception:
        logger.warning("Failed to fetch existing feed from R2", exc_info=True)
        return None


def _upload_feed(client, bucket: str, key: str, xml_content: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=xml_content.encode("utf-8"),
        ContentType="application/rss+xml",
        CacheControl="no-cache, no-store, must-revalidate",
    )


def _create_feed_root(channel_link: str = "") -> ET.Element:
    rss = ET.Element(
        "rss",
        version="2.0",
        attrib={"xmlns:atom": "http://www.w3.org/2005/Atom"},
    )
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Patronus Daily Digest"
    ET.SubElement(channel, "link").text = channel_link
    ET.SubElement(channel, "description").text = (
        "A curated daily digest of research and reading"
    )
    ET.SubElement(channel, "language").text = "en"
    if channel_link:
        atom_link = ET.SubElement(channel, "atom:link")
        atom_link.set("href", channel_link)
        atom_link.set("rel", "self")
        atom_link.set("type", "application/rss+xml")
    return rss


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _format_item_html(item: DigestItem) -> str:
    parts: list[str] = []
    title = _esc(item.title or "Untitled")
    if item.url:
        parts.append(f'<p><strong><a href="{_esc(item.url)}">{title}</a></strong></p>')
    else:
        parts.append(f"<p><strong>{title}</strong></p>")

    meta: list[str] = []
    if item.source:
        meta.append(_esc(item.source))
    if item.author:
        meta.append(_esc(item.author))
    if item.published_date:
        meta.append(item.published_date[:10])
    if meta:
        parts.append(f"<p><small>{_MIDDOT.join(meta)}</small></p>")

    if item.summary:
        parts.append(f"<p>{_esc(item.summary)}</p>")

    return "\n".join(parts)


def format_digest_html(digest: Digest) -> str:
    parts: list[str] = []

    date_str = digest.generated_at[:10] if digest.generated_at else "unknown"
    parts.append(
        f"<p><em>{digest.mode} \u00b7 {digest.item_count} items \u00b7 {date_str}</em></p>"
    )

    if digest.sections:
        for section in digest.sections:
            parts.append(f"<h3>{_esc(section.title)}</h3>")
            for item in section.items:
                parts.append(_format_item_html(item))
    elif digest.items:
        for item in digest.items:
            parts.append(_format_item_html(item))
    else:
        parts.append("<p>No items in today's digest.</p>")

    return "\n".join(parts)


def _build_rss_item(digest: Digest, item_link: str = "") -> ET.Element:
    item = ET.Element("item")

    date_str = digest.generated_at[:10] if digest.generated_at else "unknown"
    ET.SubElement(item, "title").text = f"Patronus Digest \u2014 {date_str}"
    if item_link:
        ET.SubElement(item, "link").text = item_link
    ET.SubElement(item, "description").text = format_digest_html(digest)

    now = datetime.now(timezone.utc)
    ET.SubElement(item, "pubDate").text = format_rfc2822(now)

    guid = ET.SubElement(item, "guid")
    guid.set("isPermaLink", "false")
    guid.text = f"patronus-digest-{digest.generated_at}"

    return item


def _serialize_rss(root: ET.Element) -> str:
    ET.indent(root)
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _cdata_wrap(match: re.Match) -> str:
        content = html.unescape(match.group(1))
        return f"<description><![CDATA[{content}]]></description>"

    return re.sub(
        r"<description>(.*?)</description>", _cdata_wrap, xml_str, flags=re.DOTALL
    )


def append_to_feed(existing_xml: str | None, digest: Digest, feed_url: str = "") -> str:
    if existing_xml:
        try:
            root = ET.fromstring(existing_xml)
        except ET.ParseError:
            logger.warning("Failed to parse existing feed XML, creating new feed")
            root = _create_feed_root(feed_url)
    else:
        root = _create_feed_root(feed_url)

    channel = root.find("channel")
    if channel is None:
        root = _create_feed_root(feed_url)
        channel = root.find("channel")

    if feed_url:
        link_el = channel.find("link")
        if link_el is not None:
            link_el.text = feed_url
        atom_self = channel.find("atom:link")
        if atom_self is not None:
            atom_self.set("href", feed_url)

    now_str = format_rfc2822(datetime.now(timezone.utc))
    last_build = channel.find("lastBuildDate")
    if last_build is not None:
        last_build.text = now_str
    else:
        ET.SubElement(channel, "lastBuildDate").text = now_str

    new_item = _build_rss_item(digest, item_link=feed_url)

    first_item_idx = None
    for i, child in enumerate(channel):
        if child.tag == "item":
            first_item_idx = i
            break
    if first_item_idx is not None:
        channel.insert(first_item_idx, new_item)
    else:
        channel.append(new_item)

    items = channel.findall("item")
    for old_item in items[_MAX_FEED_ITEMS:]:
        channel.remove(old_item)

    return _serialize_rss(root)


def feed_key(filename: str, tag: str | None) -> str:
    p = Path(filename)
    base, ext = p.stem, p.suffix or ".xml"
    if tag:
        return f"{base}-{tag}{ext}"
    return f"{base}{ext}"


class FeedOutput:
    def __init__(self, tag: str | None = None, filename: str = "feed.xml"):
        self._tag = tag
        self._filename = filename

    def send(self, digest: Digest, config: Config) -> None:
        missing = [v for v in _R2_ENV_VARS if not os.environ.get(v)]
        if missing:
            logger.warning(
                "R2 not configured (missing: %s), skipping feed output",
                ", ".join(missing),
            )
            return

        bucket = os.environ["CF_R2_BUCKET_NAME"]
        key = feed_key(self._filename, self._tag)
        feed_url = _public_feed_url(key)

        try:
            client = _get_r2_client()
            existing = _fetch_existing_feed(client, bucket, key)
            updated = append_to_feed(existing, digest, feed_url=feed_url)
            _upload_feed(client, bucket, key, updated)
            logger.info("Uploaded RSS feed to r2://%s/%s (%s)", bucket, key, feed_url)
        except Exception:
            logger.exception("Failed to upload feed to R2")
