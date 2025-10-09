from typing import List, Dict, Optional, TypedDict, Tuple, Set
import os
import json
import argparse
from enum import Enum
from datetime import datetime, timezone
from time import mktime
from collections import defaultdict, Counter
import logging
from logging.handlers import SysLogHandler
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import feedparser
from trafilatura import fetch_url, extract
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from lxml import etree as _lxml_etree
from typing import Any
etree: Any = _lxml_etree
import copy
from dateutil import parser as dateutil_parser

logger: logging.Logger = logging.getLogger("patronus")

def setup_logging() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    syslog_handler: Optional[logging.Handler] = None
    for address in ["/dev/log", "/var/run/syslog", ("127.0.0.1", 514)]:
        try:
            h: SysLogHandler = SysLogHandler(address=address)
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("patronus: %(message)s"))
            syslog_handler = h
            break
        except Exception:
            continue
    stream_handler: logging.Handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
    )
    if syslog_handler is not None:
        logger.addHandler(syslog_handler)
    logger.addHandler(stream_handler)


class State:
    def __init__(self, seen_links: Optional[Set[str]] = None, assignments: Optional[Dict[str, str]] = None) -> None:
        self.seen_links: Set[str] = seen_links or set()
        self.assignments: Dict[str, str] = assignments or {}


def _resolve_bucket_value(name: str) -> Optional[str]:
    try:
        _ = Bucket(name)
        return name
    except Exception:
        pass
    try:
        import difflib
        candidates: List[str] = [b.value for b in Bucket]
        matches: List[str] = difflib.get_close_matches(name, candidates, n=1, cutoff=0.8)
        return matches[0] if matches else None
    except Exception:
        return None


def load_state(path: str) -> State:
    if not path:
        return State()
    if not os.path.exists(path):
        logger.info("state file not found, starting fresh path=%s", path)
        return State()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        seen_list_raw: List[str] = data.get("seen_links", [])
        assignments_raw: Dict[str, str] = data.get("assignments", {})
        seen_list: List[str] = [normalize_url(x) for x in seen_list_raw if x]
        assignments: Dict[str, str] = {}
        for link, bucket in assignments_raw.items():
            nlink: str = normalize_url(link)
            resolved: Optional[str] = _resolve_bucket_value(bucket)
            if resolved is not None:
                assignments[nlink] = resolved
        logger.info("state loaded seen=%d assignments=%d", len(seen_list), len(assignments))
        return State(set(seen_list), dict(assignments))
    except Exception:
        logger.exception("failed to load state path=%s", path)
        return State()


def save_state(path: str, state: State) -> None:
    if not path:
        return
    tmp_path: str = f"{path}.tmp"
    data = {
        "seen_links": sorted(list(state.seen_links)),
        "assignments": state.assignments,
    }
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)
    logger.info("state saved path=%s seen=%d assignments=%d", path, len(state.seen_links), len(state.assignments))

class Bucket(str, Enum):
    REJECT = "REJECT"
    TECHNICAL_AI_AND_ML = "TECHNICAL_AI_AND_ML"
    TECH_BEYOND_THE_TECHNICAL = "TECH_BEYOND_THE_TECHNICAL"
    PHILOSOPHY_CONSCIOUSNESS = "PHILOSOPHY_CONSCIOUSNESS"
    POLITICS_CULTURE = "POLITICS_CULTURE"
    SPAIN = "SPAIN"
    CHINA = "CHINA"
    RANDOM_CURIOSITIES = "RANDOM_CURIOSITIES"

    @property
    def display_name(self) -> str:
        mapping: Dict["Bucket", str] = {
            Bucket.REJECT: "Rejected",
            Bucket.TECHNICAL_AI_AND_ML: "Technical AI/ML",
            Bucket.TECH_BEYOND_THE_TECHNICAL: "Tech Beyond The Technical",
            Bucket.PHILOSOPHY_CONSCIOUSNESS: "Philosophy & Consciousness",
            Bucket.POLITICS_CULTURE: "Politics & Culture",
            Bucket.SPAIN: "Spain",
            Bucket.CHINA: "China",
            Bucket.RANDOM_CURIOSITIES: "Random Curiosities",
        }
        return mapping[self]

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
    client: OpenAI = OpenAI(api_key=api_key)
    logger.info("openai client initialized")
    return client


def read_profile(profile_path: str) -> str:
    with open(profile_path, "r") as pf:
        content: str = pf.read()
    logger.info("profile loaded path=%s", profile_path)
    return content


def read_feed_urls(feeds_path: str) -> List[str]:
    with open(feeds_path, "r") as f:
        urls: List[str] = [line.strip() for line in f if line.strip()]
    logger.info("feeds loaded count=%d path=%s", len(urls), feeds_path)
    return urls


def system_preamble_text() -> str:
    return (
        "You are an assistant tasked with selecting content from an RSS feed that is relevant to Dani. "
        "Classify each article into one of the following buckets based on Dani's profile: "
        "REJECT, TECHNICAL_AI_AND_ML, TECH_BEYOND_THE_TECHNICAL, PHILOSOPHY_CONSCIOUSNESS, POLITICS_CULTURE, SPAIN, CHINA, RANDOM_CURIOSITIES."
    )


def normalize_url(url: str) -> str:
    if not url:
        return url
    s = urlsplit(url)
    scheme: str = s.scheme
    netloc: str = s.netloc.lower()
    path: str = s.path or ""
    if path != "/":
        path = path.rstrip("/")
    raw_params = parse_qsl(s.query, keep_blank_values=False)
    filtered_params = [(k, v) for k, v in raw_params if not (k.lower().startswith("utm_") or k.lower() in {"gclid", "fbclid", "igshid", "ref"})]
    filtered_params.sort(key=lambda kv: kv[0])
    query: str = urlencode(filtered_params, doseq=True)
    fragment: str = ""
    return urlunsplit((scheme, netloc, path, query, fragment))


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
                "link": normalize_url(getattr(e, "link", "")),
                "summary": getattr(e, "summary", getattr(e, "description", "")),
                "published": published_dt,
                "author": _entry_author(e) or getattr(parsed.feed, "title", ""),
                "feed_title": getattr(parsed.feed, "title", ""),
                "feed_link": getattr(parsed.feed, "link", ""),
            })
        feed_meta_sorted: List[Dict] = sorted(
            feed_meta,
            key=lambda x: (x["published"] or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        meta_entries.extend(feed_meta_sorted[:per_feed_limit])
    logger.info("collected meta entries total=%d from_feeds=%d", len(meta_entries), len(feed_urls))
    return meta_entries


def build_article_list(meta_entries: List[Dict], total_limit: Optional[int]) -> List[Article]:
    logger.info("building article list input_count=%d total_limit=%s", len(meta_entries), str(total_limit))
    top_meta: List[Dict] = sorted(
        meta_entries,
        key=lambda x: (x["published"] or datetime.min.replace(tzinfo=timezone.utc)),
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
    logger.info("article list built count=%d", len(articles))
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
    system_preamble: str = system_preamble_text()
    buckets: Dict[Bucket, List[Article]] = defaultdict(list)
    classifications: List[ArticleClassification] = []
    for idx, art in enumerate(articles, start=1):
        classification = classify_article(client, system_preamble, profile_text, art)
        classifications.append(classification)
        buckets[classification.bucket].append(art)
        logger.info("classified bucket=%s title=%s link=%s", classification.bucket.value, art.get("title", ""), art.get("link", ""))
        if idx % 10 == 0 or idx == len(articles):
            logger.info("classification progress %d/%d", idx, len(articles))
    return classifications, buckets


def build_atom_xml(bucket_key: Bucket, items: List[Article]) -> str:
    from feedgen.feed import FeedGenerator
    fg = FeedGenerator()
    fg.id(f"urn:patronus:{bucket_key.value}")
    fg.title(f"Patronus: {bucket_key.display_name} (Atom)")
    fg.link(href="https://example.com", rel="alternate")
    fg.subtitle(f"Filtered feed for {bucket_key.display_name} (Atom)")
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
            rss_feed_author: str = ""
            for tag in [
                "managingEditor",
                "webMaster",
                "author",
            ]:
                el = root.find(f"channel/{tag}")
                if el is not None and getattr(el, "text", None) and el.text.strip():
                    rss_feed_author = el.text.strip()
                    break
            if not rss_feed_author:
                dc_creator = root.find("channel/{http://purl.org/dc/elements/1.1/}creator")
                if dc_creator is not None and getattr(dc_creator, "text", None) and dc_creator.text.strip():
                    rss_feed_author = dc_creator.text.strip()
            for it in items:
                link_el = it.find("link")
                orig_key = link_el.text.strip() if link_el is not None and link_el.text else None
                key = normalize_url(orig_key) if orig_key else None
                if key:
                    idx[key] = it
                    origin[key] = {"feed_url": feed_url, "feed_title": feed_title, "feed_link": feed_link, "kind": kind, "feed_author": rss_feed_author}
        elif kind == "atom":
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.xpath("/atom:feed/atom:entry", namespaces=ns)
            feed_title_el = root.find("{http://www.w3.org/2005/Atom}title")
            feed_link_hrefs: list[str] = root.xpath("atom:link[@rel='alternate']/@href", namespaces=ns)
            feed_title = feed_title_el.text.strip() if feed_title_el is not None and feed_title_el.text else ""
            feed_link = feed_link_hrefs[0].strip() if feed_link_hrefs else feed_url
            atom_author_name_el = root.find("{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name")
            atom_feed_author: str = atom_author_name_el.text.strip() if atom_author_name_el is not None and getattr(atom_author_name_el, "text", None) else ""
            for e in entries:
                hrefs: list[str] = e.xpath("atom:link[@rel='alternate']/@href", namespaces=ns)
                if not hrefs:
                    hrefs = e.xpath("atom:link[not(@rel) or @rel='alternate']/@href", namespaces=ns)
                orig_key = hrefs[0].strip() if hrefs else None
                key = normalize_url(orig_key) if orig_key else None
                if key:
                    idx[key] = e
                    origin[key] = {"feed_url": feed_url, "feed_title": feed_title, "feed_link": feed_link, "kind": kind, "feed_author": atom_feed_author}
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
        except Exception as e:
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
            "link": normalize_url(link),
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
            "link": normalize_url(link),
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


def upload_bucket_feeds(buckets: Dict[Bucket, List[Article]], rss_index: Dict[str, Any], atom_index: Dict[str, Any], origin_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    from google.cloud import storage
    load_dotenv()
    gcs_bucket_name: str = os.getenv("GCS_BUCKET_NAME", "")
    gcs_prefix: str = os.getenv("GCS_PREFIX", "patronus/feeds/").lstrip("/")
    if not gcs_bucket_name:
        raise RuntimeError("Missing GCS_BUCKET_NAME in environment")

    def _rss_doc(bucket_key: Bucket, selected: List[Article]) -> str:
        rss = etree.Element("rss", attrib={"version": "2.0"})
        channel = etree.SubElement(rss, "channel")
        title = etree.SubElement(channel, "title"); title.text = f"Patronus: {bucket_key.display_name} (RSS)"
        link = etree.SubElement(channel, "link"); link.text = "https://example.com"
        desc = etree.SubElement(channel, "description"); desc.text = f"Filtered feed for {bucket_key.display_name} (RSS)"
        copied_count: int = 0
        for it in selected:
            key: Optional[str] = (it.get("link") or None)
            if key and key in rss_index:
                item_copy = copy.deepcopy(rss_index[key])
                has_author_el = item_copy.find("author")
                has_dc_creator_el = item_copy.find("{http://purl.org/dc/elements/1.1/}creator")
                if (has_author_el is None or not (getattr(has_author_el, "text", None) or "")) and (has_dc_creator_el is None or not (getattr(has_dc_creator_el, "text", None) or "")):
                    origin = origin_map.get(key, {})
                    fallback = (origin.get("feed_author") or origin.get("feed_title") or "").strip()
                    if fallback:
                        au = etree.SubElement(item_copy, "author"); au.text = fallback
                channel.append(item_copy)
                copied_count += 1
        return etree.tostring(rss, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")

    def _atom_doc(bucket_key: Bucket, selected: List[Article]) -> str:
        ns = "http://www.w3.org/2005/Atom"
        feed = etree.Element("{%s}feed" % ns, nsmap={None: ns})
        id_el = etree.SubElement(feed, "{%s}id" % ns); id_el.text = f"urn:patronus:{bucket_key.value}"
        title_el = etree.SubElement(feed, "{%s}title" % ns); title_el.text = f"Patronus: {bucket_key.display_name} (Atom)"
        updated_el = etree.SubElement(feed, "{%s}updated" % ns); updated_el.text = datetime.now(timezone.utc).isoformat()
        copied_count: int = 0
        for it in selected:
            key: Optional[str] = (it.get("link") or None)
            if key and key in atom_index:
                entry_copy = copy.deepcopy(atom_index[key])
                has_author = entry_copy.find("{%s}author" % ns)
                if has_author is None:
                    origin = origin_map.get(key, {})
                    fallback = (origin.get("feed_author") or origin.get("feed_title") or "").strip()
                    if fallback:
                        a = etree.SubElement(entry_copy, "{%s}author" % ns)
                        n = etree.SubElement(a, "{%s}name" % ns); n.text = fallback
                feed.append(entry_copy)
                copied_count += 1
        return etree.tostring(feed, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")

    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(gcs_bucket_name)
    public_urls: Dict[str, str] = {}
    selection_counts: Dict[str, int] = {}
    for b in Bucket:
        selected_items: List[Article] = buckets.get(b, [])
        selection_counts[b.value] = len(selected_items)
        logger.info("upload start bucket=%s count=%d", b.value, len(selected_items))
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
        logger.info("upload complete bucket=%s rss=%s atom=%s", b.value, rss_key, atom_key)
    logger.info("uploaded artifacts bucket_counts=%s", selection_counts)
    return public_urls


def upload_from_assignments(assignments: Dict[str, str], rss_index: Dict[str, Any], atom_index: Dict[str, Any], origin_map: Dict[str, Dict[str, str]], dry_run: bool) -> Optional[Dict[str, str]]:
    buckets: Dict[Bucket, List[Article]] = {b: [] for b in Bucket}
    for link, bucket_value in assignments.items():
        nlink: str = normalize_url(link)
        if nlink not in rss_index and nlink not in atom_index:
            continue
        resolved_value: Optional[str] = _resolve_bucket_value(bucket_value)
        if resolved_value is None:
            continue
        bucket_key: Bucket = Bucket(resolved_value)
        a: Article = {
            "title": nlink,
            "link": nlink,
            "content": "",
            "published": None,
            "author": None,
            "feed_title": None,
            "feed_link": None,
        }
        buckets[bucket_key].append(a)
    has_any: bool = any(len(v) > 0 for v in buckets.values())
    if not has_any:
        return None
    if dry_run:
        return {}
    return upload_bucket_feeds(buckets, rss_index, atom_index, origin_map)


def print_summary(classifications: List[ArticleClassification]) -> None:
    counts = Counter([c.bucket.value for c in classifications])
    print("Summary Statistics:")
    for b in [bucket.value for bucket in Bucket]:
        print(f"{b}: {counts.get(b, 0)}")
    logger.info("summary %s", dict(counts))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Patronus RSS filter")
    parser.add_argument("--feeds-path", type=str, default="feeds")
    parser.add_argument("--profile-path", type=str, default="Profile.md")
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    client: OpenAI = get_openai_client()
    profile_text: str = read_profile(args.profile_path)
    feed_urls: List[str] = read_feed_urls(args.feeds_path)
    rss_index, atom_index, origin_map = build_source_index(feed_urls)
    meta_entries: List[Dict] = collect_meta_entries_from_index(rss_index, atom_index, origin_map, per_feed_limit=10)
    # Dedupe by link and classify only new links if state is provided
    meta_by_link: Dict[str, Dict] = {m.get("link", ""): m for m in meta_entries if m.get("link")}
    current_links: Set[str] = set(meta_by_link.keys())
    state: State = load_state(args.state_path)
    seen_links: Set[str] = state.seen_links
    new_links: List[str] = [l for l in meta_by_link.keys() if l not in seen_links]
    logger.info("batch selection current=%d seen=%d new=%d", len(current_links), len(seen_links), len(new_links))

    # Cap only the NEW links to limit cost
    total_limit: Optional[int] = 40 if args.dry_run else 50
    if total_limit is not None and len(new_links) > total_limit:
        # preserve recency order based on meta_entries sorting
        ordered_new: List[str] = []
        for m in meta_entries:
            lnk: str = m.get("link", "")
            if lnk and lnk in new_links and lnk not in ordered_new:
                ordered_new.append(lnk)
            if len(ordered_new) >= total_limit:
                break
        new_links = ordered_new

    # Build articles from the selected new metadata
    selected_meta: List[Dict] = [meta_by_link[l] for l in new_links]
    articles: List[Article] = build_article_list(selected_meta, total_limit=None)
    classifications, _buckets = classify_articles(client, profile_text, articles)
    print_summary(classifications)

    # Merge assignments and upload based on full set
    new_assignments: Dict[str, str] = {}
    for art, cls in zip(articles, classifications):
        new_assignments[art["link"]] = cls.bucket.value
    combined_assignments: Dict[str, str] = dict(state.assignments)
    combined_assignments.update(new_assignments)

    result: Optional[Dict[str, str]] = upload_from_assignments(combined_assignments, rss_index, atom_index, origin_map, args.dry_run)
    if args.dry_run:
        print("DRY RUN: skipping upload to GCS. No artifacts were written.")
    else:
        if result is not None:
            print({k: v for k, v in result.items()})
        else:
            print("No artifacts to upload.")

    # Persist state only on real runs and when path provided
    if not args.dry_run and args.state_path:
        state.assignments.update(new_assignments)
        state.seen_links.update(new_links)
        save_state(args.state_path, state)


if __name__ == "__main__":
    main()

