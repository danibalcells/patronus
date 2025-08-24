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


def build_rss_xml(bucket_key: Bucket, items: List[Article]) -> str:
    from feedgen.feed import FeedGenerator
    fg = FeedGenerator()
    fg.title(f"Patronus: {bucket_key.value}")
    fg.link(href="https://example.com", rel="alternate")
    fg.description(f"Filtered feed for {bucket_key.value}")
    for it in items:
        fe = fg.add_entry()
        fe.title(it["title"]) 
        fe.link(href=it["link"])
        if it.get("published"):
            pub_dt = it["published"]
            if isinstance(pub_dt, datetime) and pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            fe.published(pub_dt) 
        author_val = it.get("author")
        if author_val:
            fe.author(name=author_val)
    return fg.rss_str(pretty=True).decode("utf-8")


def upload_bucket_feeds(buckets: Dict[Bucket, List[Article]]) -> Dict[str, str]:
    from google.cloud import storage
    load_dotenv()
    gcs_bucket_name: str = os.getenv("GCS_BUCKET_NAME", "")
    gcs_prefix: str = os.getenv("GCS_PREFIX", "patronus/feeds/").lstrip("/")
    if not gcs_bucket_name:
        raise RuntimeError("Missing GCS_BUCKET_NAME in environment")
    storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(gcs_bucket_name)
    public_urls: Dict[str, str] = {}
    for b in Bucket:
        key = f"{gcs_prefix}{b.value}.xml"
        xml_data: str = build_rss_xml(b, buckets.get(b, []))
        blob = gcs_bucket.blob(key)
        blob.upload_from_string(xml_data, content_type="application/rss+xml")
        public_urls[b.value] = blob.public_url
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
    meta_entries: List[Dict] = collect_meta_entries(feed_urls, per_feed_limit=10)
    total_limit: Optional[int] = 5 if args.dry_run else 50
    articles: List[Article] = build_article_list(meta_entries, total_limit=total_limit)
    classifications, buckets = classify_articles(client, profile_text, articles)
    print_summary(classifications)
    public_urls: Dict[str, str] = upload_bucket_feeds(buckets)
    print({k: v for k, v in public_urls.items()})


if __name__ == "__main__":
    main()

