from typing import List, Dict, Optional, TypedDict, Tuple
import os
import json
import argparse
from enum import Enum
from datetime import datetime, timezone
from time import mktime
from collections import defaultdict, Counter
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import feedparser
from trafilatura import fetch_url, extract
from urllib.request import Request, urlopen
from lxml import etree as _lxml_etree
from typing import Any
etree: Any = _lxml_etree
import copy
from dateutil import parser as dateutil_parser

class Bucket(str, Enum):
    REJECT = "reject"
    TECHNICAL_AI_ML = "technical_ai_ml"
    AI_SAFETY_BUSINESS = "ai_safety_business"
    PHILOSOPHY_CONSCIOUSNESS = "philosophy_consciousness"
    POLITICS_CULTURE = "politics_culture"
    SPAIN = "spain"
    CHINA = "china"
    RANDOM_CURIOSITIES = "random_curiosities"

class Article(TypedDict):
    title: str
    link: str
    content: str
    published: Optional[datetime]
    author: Optional[str]
    feed_title: Optional[str]
    feed_link: Optional[str]

class ArticleClassification(BaseModel):
    bucket: Bucket
    reason: str


def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=api_key)


def read_profile(profile_path: str) -> str:
    with open(profile_path, "r") as pf:
        return pf.read()


def read_feed_urls(feeds_path: str) -> List[str]:
    with open(feeds_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _entry_author(entry: object) -> Optional[str]:
    if hasattr(entry, "author") and getattr(entry, "author"):
        return getattr(entry, "author")
    if hasattr(entry, "authors") and getattr(entry, "authors"):
        authors = getattr(entry, "authors")
        try:
            first = authors[0]
            if isinstance(first, dict):
                return first.get("name") or first.get("email")
            return getattr(first, "name", None)
        except Exception:
            return None
    if hasattr(entry, "author_detail") and getattr(entry, "author_detail"):
        detail = getattr(entry, "author_detail")
        try:
            return getattr(detail, "name", None) or getattr(detail, "email", None)
        except Exception:
            return None
    return None


def collect_meta_entries(feed_urls: List[str], per_feed_limit: int = 10) -> List[Dict]:
    meta_entries: List[Dict] = []
    for url in feed_urls:
        parsed = feedparser.parse(url)
        feed_meta: List[Dict] = []
        for e in parsed.entries:
            published_dt: Optional[datetime] = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                published_dt = datetime.fromtimestamp(mktime(e.published_parsed))
            elif hasattr(e, "updated_parsed") and e.updated_parsed:
                published_dt = datetime.fromtimestamp(mktime(e.updated_parsed))
            if published_dt is not None and published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
            feed_meta.append({
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "summary": getattr(e, "summary", getattr(e, "description", "")),
                "published": published_dt,
                "author": _entry_author(e) or getattr(parsed.feed, "title", ""),
                "feed_title": getattr(parsed.feed, "title", ""),
                "feed_link": getattr(parsed.feed, "link", ""),
            })
        feed_meta_sorted: List[Dict] = sorted(
            feed_meta,
            key=lambda x: (x["published"] or datetime.min),
            reverse=True,
        )
        meta_entries.extend(feed_meta_sorted[:per_feed_limit])
    return meta_entries


def build_article_list(meta_entries: List[Dict], total_limit: Optional[int]) -> List[Article]:
    top_meta: List[Dict] = sorted(
        meta_entries,
        key=lambda x: (x["published"] or datetime.min),
        reverse=True,
    )
    if total_limit is not None:
        top_meta = top_meta[:total_limit]
    articles: List[Article] = []
    for m in top_meta:
        link: str = m["link"]
        content_text: str = ""
        html: Optional[str] = fetch_url(link) if link else None
        if html:
            extracted: Optional[str] = extract(html, include_comments=False, include_tables=False)
            if extracted:
                content_text = extracted
        if not content_text:
            content_text = m.get("summary", "")
        raw_author: str = (m.get("author") or "").strip()
        feed_name: str = (m.get("feed_title") or "").strip()
        if raw_author and feed_name and raw_author.lower() != feed_name.lower():
            author_out: Optional[str] = f"{raw_author} - {feed_name}"
        else:
            author_out = raw_author or feed_name or None
        articles.append({
            "title": m["title"],
            "link": link,
            "content": content_text,
            "published": m["published"],
            "author": author_out,
            "feed_title": feed_name or None,
            "feed_link": (m.get("feed_link") or None),
        })
    return articles


def classify_article(client: OpenAI, system_preamble: str, profile_text: str, article: Article) -> ArticleClassification:
    user_prompt: str = (
        f"Profile about Dani (verbatim):\n\n{profile_text}\n\n"
        f"Article to classify:\nTitle: {article['title']}\nLink: {article['link']}\nContent (truncated to 4000 chars):\n{article['content'][:4000]}\n\n"
        f"Return only valid JSON matching this schema (no extra keys):\n{ArticleClassification.model_json_schema()}"
    )
    comp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_preamble},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content: str = comp.choices[0].message.content or "{}"
    try:
        json.loads(content)
    except Exception:
        content = "{}"
    parsed: ArticleClassification = ArticleClassification.model_validate_json(content)
    return parsed


def classify_articles(client: OpenAI, profile_text: str, articles: List[Article]) -> Tuple[List[ArticleClassification], Dict[Bucket, List[Article]]]:
    system_preamble: str = (
        "You are an assistant tasked with selecting content from an RSS feed that is relevant to Dani. "
        "Classify each article into one of the following buckets based on Dani's profile: "
        "REJECT, TECHNICAL_AI_ML, AI_SAFETY_BUSINESS, PHILOSOPHY_CONSCIOUSNESS, POLITICS_CULTURE, SPAIN, CHINA, RANDOM_CURIOSITIES."
    )
    buckets: Dict[Bucket, List[Article]] = defaultdict(list)
    classifications: List[ArticleClassification] = []
    for art in articles:
        classification = classify_article(client, system_preamble, profile_text, art)
        classifications.append(classification)
        buckets[classification.bucket].append(art)
    return classifications, buckets


def build_atom_xml(bucket_key: Bucket, items: List[Article]) -> str:
    from feedgen.feed import FeedGenerator
    from urllib.parse import urlparse
    fg = FeedGenerator()
    fg.id(f"urn:patronus:{bucket_key.value}")
    fg.title(f"Patronus: {bucket_key.value}")
    fg.link(href="https://example.com", rel="alternate")
    fg.subtitle(f"Filtered feed for {bucket_key.value}")
    fg.updated(datetime.now(timezone.utc))
    for it in items:
        fe = fg.add_entry()
        fe.id(it["link"] or it["title"]) 
        fe.title(it["title"])
        if it.get("link"):
            fe.link(href=it["link"]) 
        pub_dt = it.get("published")
        if isinstance(pub_dt, datetime):
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            fe.published(pub_dt)
            fe.updated(pub_dt)
        display_author: str = (it.get("author") or "").strip()
        feed_name: str = (it.get("feed_title") or "").strip()
        author_name_out: str = display_author or feed_name
        if author_name_out:
            fe.author({"name": author_name_out})
        source_title: str = (it.get("feed_title") or "").strip()
        source_link: str = (it.get("feed_link") or "").strip()
        if source_title or source_link:
            fe.source(url=source_link or None, title=source_title or None)
        # content_html: str = it.get("content", "")
        # if content_html:
        #     fe.content(type="html", content=content_html)
    return fg.atom_str(pretty=True).decode("utf-8")


def build_source_index(feed_urls: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]]]:
    def _fetch_raw(url: str) -> bytes:
        req: Request = Request(url, headers={"User-Agent": "Patronus/1.0"})
        with urlopen(req) as resp:
            return resp.read()

    def _detect_kind(root: Any) -> str:
        tag: str = etree.QName(root.tag).localname.lower()
        if tag == "rss":
            return "rss"
        if tag == "feed" and root.tag.startswith("{http://www.w3.org/2005/Atom}"):
            return "atom"
        return "unknown"

    def _index_entries(root: Any, kind: str, feed_url: str, origin: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        idx: Dict[str, Any] = {}
        if kind == "rss":
            items = root.xpath("/rss/channel/item")
            ch_title_el = root.find("channel/title")
            ch_link_el = root.find("channel/link")
            feed_title = ch_title_el.text.strip() if ch_title_el is not None and ch_title_el.text else ""
            feed_link = ch_link_el.text.strip() if ch_link_el is not None and ch_link_el.text else feed_url
            for it in items:
                link_el = it.find("link")
                key = link_el.text.strip() if link_el is not None and link_el.text else None
                if key:
                    idx[key] = it
                    origin[key] = {"feed_url": feed_url, "feed_title": feed_title, "feed_link": feed_link, "kind": kind}
        elif kind == "atom":
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.xpath("/atom:feed/atom:entry", namespaces=ns)
            feed_title_el = root.find("{http://www.w3.org/2005/Atom}title")
            feed_link_hrefs: list[str] = root.xpath("atom:link[@rel='alternate']/@href", namespaces=ns)
            feed_title = feed_title_el.text.strip() if feed_title_el is not None and feed_title_el.text else ""
            feed_link = feed_link_hrefs[0].strip() if feed_link_hrefs else feed_url
            for e in entries:
                hrefs: list[str] = e.xpath("atom:link[@rel='alternate']/@href", namespaces=ns)
                if not hrefs:
                    hrefs = e.xpath("atom:link[not(@rel) or @rel='alternate']/@href", namespaces=ns)
                key = hrefs[0].strip() if hrefs else None
                if key:
                    idx[key] = e
                    origin[key] = {"feed_url": feed_url, "feed_title": feed_title, "feed_link": feed_link, "kind": kind}
        return idx

    rss_index: Dict[str, Any] = {}
    atom_index: Dict[str, Any] = {}
    origin_map: Dict[str, Dict[str, str]] = {}
    for feed_url in feed_urls:
        try:
            raw: bytes = _fetch_raw(feed_url)
            root: Any = etree.fromstring(raw)
            kind: str = _detect_kind(root)
            idx: Dict[str, Any] = _index_entries(root, kind, feed_url, origin_map)
            if kind == "rss":
                rss_index.update(idx)
            elif kind == "atom":
                atom_index.update(idx)
            print(f"Indexed feed kind={kind} entries={len(idx)} url={feed_url}")
        except Exception as e:
            print(f"Feed fetch/index failed url={feed_url} err={e}")
            continue
    return rss_index, atom_index, origin_map


def collect_meta_entries_from_index(rss_index: Dict[str, Any], atom_index: Dict[str, Any], origin_map: Dict[str, Dict[str, str]], per_feed_limit: int = 10) -> List[Dict]:
    def _get_text(el: Any) -> str:
        return el.text.strip() if el is not None and getattr(el, "text", None) else ""

    def _ensure_tz(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    by_feed: Dict[str, List[Dict]] = defaultdict(list)

    for key, node in rss_index.items():
        title = _get_text(node.find("title"))
        link = key
        desc = _get_text(node.find("description"))
        if not desc:
            enc = node.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            desc = _get_text(enc)
        pub_dt: Optional[datetime] = None
        pub_el = node.find("pubDate")
        if pub_el is not None and _get_text(pub_el):
            try:
                pub_dt = _ensure_tz(dateutil_parser.parse(_get_text(pub_el)))
            except Exception:
                pub_dt = None
        author_txt = _get_text(node.find("author"))
        if not author_txt:
            creator = node.find("{http://purl.org/dc/elements/1.1/}creator")
            author_txt = _get_text(creator)
        origin = origin_map.get(key, {})
        by_feed[origin.get("feed_url", "")].append({
            "title": title,
            "link": link,
            "summary": desc,
            "published": pub_dt,
            "author": author_txt or origin.get("feed_title", ""),
            "feed_title": origin.get("feed_title", ""),
            "feed_link": origin.get("feed_link", ""),
        })

    atom_ns = "http://www.w3.org/2005/Atom"
    for key, node in atom_index.items():
        title = _get_text(node.find(f"{{{atom_ns}}}title"))
        link = key
        summary = _get_text(node.find(f"{{{atom_ns}}}summary"))
        pub_dt: Optional[datetime] = None
        upd_el = node.find(f"{{{atom_ns}}}updated")
        pub_el = node.find(f"{{{atom_ns}}}published")
        for d in [upd_el, pub_el]:
            if d is not None and _get_text(d):
                try:
                    pub_dt = _ensure_tz(dateutil_parser.parse(_get_text(d)))
                    break
                except Exception:
                    pub_dt = None
        author_name = _get_text(node.find(f"{{{atom_ns}}}author/{{{atom_ns}}}name"))
        origin = origin_map.get(key, {})
        by_feed[origin.get("feed_url", "")].append({
            "title": title,
            "link": link,
            "summary": summary,
            "published": pub_dt,
            "author": author_name or origin.get("feed_title", ""),
            "feed_title": origin.get("feed_title", ""),
            "feed_link": origin.get("feed_link", ""),
        })

    meta_entries: List[Dict] = []
    def _sort_key(item: Dict) -> datetime:
        dt = item.get("published")
        if isinstance(dt, datetime):
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        return datetime.min.replace(tzinfo=timezone.utc)
    for feed_url, entries in by_feed.items():
        entries_sorted = sorted(entries, key=_sort_key, reverse=True)
        meta_entries.extend(entries_sorted[:per_feed_limit])
    return meta_entries


def upload_bucket_feeds(buckets: Dict[Bucket, List[Article]], rss_index: Dict[str, Any], atom_index: Dict[str, Any]) -> Dict[str, str]:
    from google.cloud import storage
    load_dotenv()
    gcs_bucket_name: str = os.getenv("GCS_BUCKET_NAME", "")
    gcs_prefix: str = os.getenv("GCS_PREFIX", "patronus/feeds/").lstrip("/")
    if not gcs_bucket_name:
        raise RuntimeError("Missing GCS_BUCKET_NAME in environment")

    def _rss_doc(bucket_key: Bucket, selected: List[Article]) -> str:
        rss = etree.Element("rss", attrib={"version": "2.0"})
        channel = etree.SubElement(rss, "channel")
        title = etree.SubElement(channel, "title"); title.text = f"Patronus: {bucket_key.value}"
        link = etree.SubElement(channel, "link"); link.text = "https://example.com"
        desc = etree.SubElement(channel, "description"); desc.text = f"Filtered feed for {bucket_key.value}"
        copied_count: int = 0
        for it in selected:
            key: Optional[str] = (it.get("link") or None)
            if key and key in rss_index:
                channel.append(copy.deepcopy(rss_index[key]))
                copied_count += 1
                print(f"RSS item copied bucket={bucket_key.value} link={key}")
            else:
                print(f"RSS item missing bucket={bucket_key.value} link={key}")
        print(f"RSS bucket={bucket_key.value} copied={copied_count} selected={len(selected)}")
        return etree.tostring(rss, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")

    def _atom_doc(bucket_key: Bucket, selected: List[Article]) -> str:
        ns = "http://www.w3.org/2005/Atom"
        feed = etree.Element("{%s}feed" % ns, nsmap={None: ns})
        id_el = etree.SubElement(feed, "{%s}id" % ns); id_el.text = f"urn:patronus:{bucket_key.value}"
        title_el = etree.SubElement(feed, "{%s}title" % ns); title_el.text = f"Patronus: {bucket_key.value}"
        updated_el = etree.SubElement(feed, "{%s}updated" % ns); updated_el.text = datetime.now(timezone.utc).isoformat()
        copied_count: int = 0
        for it in selected:
            key: Optional[str] = (it.get("link") or None)
            if key and key in atom_index:
                feed.append(copy.deepcopy(atom_index[key]))
                copied_count += 1
                print(f"ATOM entry copied bucket={bucket_key.value} link={key}")
            else:
                print(f"ATOM entry missing bucket={bucket_key.value} link={key}")
        print(f"ATOM bucket={bucket_key.value} copied={copied_count} selected={len(selected)}")
        return etree.tostring(feed, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")

    from google.cloud import storage
    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(gcs_bucket_name)
    public_urls: Dict[str, str] = {}
    for b in Bucket:
        selected_items: List[Article] = buckets.get(b, [])
        rss_xml: str = _rss_doc(b, selected_items)
        atom_xml: str = _atom_doc(b, selected_items)
        rss_key = f"{gcs_prefix}{b.value}.rss.xml"
        atom_key = f"{gcs_prefix}{b.value}.atom.xml"
        rss_blob = gcs_bucket.blob(rss_key)
        rss_blob.upload_from_string(rss_xml, content_type="application/rss+xml")
        atom_blob = gcs_bucket.blob(atom_key)
        atom_blob.upload_from_string(atom_xml, content_type="application/atom+xml; charset=utf-8")
        public_urls[f"{b.value}:rss"] = rss_blob.public_url
        public_urls[f"{b.value}:atom"] = atom_blob.public_url
    return public_urls


def print_summary(classifications: List[ArticleClassification]) -> None:
    counts = Counter([c.bucket.value for c in classifications])
    print("Summary Statistics:")
    for b in [bucket.value for bucket in Bucket]:
        print(f"{b}: {counts.get(b, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patronus RSS filter")
    parser.add_argument("--feeds-path", type=str, default="/Users/dani/code/patronus/feeds")
    parser.add_argument("--profile-path", type=str, default="/Users/dani/code/patronus/Profile.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    client: OpenAI = get_openai_client()
    profile_text: str = read_profile(args.profile_path)
    feed_urls: List[str] = read_feed_urls(args.feeds_path)
    rss_index, atom_index, origin_map = build_source_index(feed_urls)
    meta_entries: List[Dict] = collect_meta_entries_from_index(rss_index, atom_index, origin_map, per_feed_limit=10)
    total_limit: Optional[int] = 40 if args.dry_run else 50
    articles: List[Article] = build_article_list(meta_entries, total_limit=total_limit)
    classifications, buckets = classify_articles(client, profile_text, articles)
    print_summary(classifications)
    public_urls: Dict[str, str] = upload_bucket_feeds(buckets, rss_index, atom_index)
    print({k: v for k, v in public_urls.items()})


if __name__ == "__main__":
    main()

