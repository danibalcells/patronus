import asyncio
import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from openai import OpenAI

from patronus import (
    Bucket,
    Article,
    ArticleClassification,
    get_openai_client,
    read_profile,
    read_feed_urls,
    build_source_index,
    collect_meta_entries_from_index,
    build_article_list,
    classify_article,
    upload_bucket_feeds,
)


class State:
    def __init__(self, seen_links: Optional[Set[str]] = None, assignments: Optional[Dict[str, str]] = None) -> None:
        self.seen_links: Set[str] = seen_links or set()
        self.assignments: Dict[str, str] = assignments or {}


def load_state(path: str) -> State:
    if not os.path.exists(path):
        return State()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        seen_list: List[str] = data.get("seen_links", [])
        assignments: Dict[str, str] = data.get("assignments", {})
        return State(set(seen_list), dict(assignments))
    except Exception:
        return State()


def save_state(path: str, state: State) -> None:
    tmp_path: str = f"{path}.tmp"
    data = {
        "seen_links": sorted(list(state.seen_links)),
        "assignments": state.assignments,
    }
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def system_preamble_text() -> str:
    return (
        "You are an assistant tasked with selecting content from an RSS feed that is relevant to Dani. "
        "Classify each article into one of the following buckets based on Dani's profile: "
        "REJECT, TECHNICAL_AI_ML, TECH_BEYOND_THE_TECHNICAL, PHILOSOPHY_CONSCIOUSNESS, POLITICS_CULTURE, SPAIN, CHINA, RANDOM_CURIOSITIES."
    )


async def perform_upload(
    assignments: Dict[str, str],
    rss_index: Dict[str, object],
    atom_index: Dict[str, object],
    origin_map: Dict[str, Dict[str, str]],
    dry_run: bool,
) -> Optional[Dict[str, str]]:
    buckets: Dict[Bucket, List[Article]] = {b: [] for b in Bucket}
    for link, bucket_value in assignments.items():
        if link not in rss_index and link not in atom_index:
            continue
        bucket_key: Bucket = Bucket(bucket_value)
        a: Article = {
            "title": link,
            "link": link,
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
    return await asyncio.to_thread(upload_bucket_feeds, buckets, rss_index, atom_index, origin_map)


async def classify_and_upload(
    link: str,
    meta: Dict,
    client: OpenAI,
    profile_path: str,
    state: State,
    state_path: str,
    state_lock: asyncio.Lock,
    upload_lock: asyncio.Lock,
    rss_index: Dict[str, object],
    atom_index: Dict[str, object],
    origin_map: Dict[str, Dict[str, str]],
    dry_run: bool,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        profile_text: str = await asyncio.to_thread(read_profile, profile_path)
        articles: List[Article] = await asyncio.to_thread(build_article_list, [meta], None)
        article: Article = articles[0]
        classification: ArticleClassification = await asyncio.to_thread(
            classify_article, client, system_preamble_text(), profile_text, article
        )
        async with state_lock:
            state.assignments[link] = classification.bucket.value
            save_state(state_path, state)
        async with upload_lock:
            await perform_upload(state.assignments, rss_index, atom_index, origin_map, dry_run)


async def polling_loop(
    feeds_path: str,
    profile_path: str,
    state_path: str,
    interval_seconds: int,
    concurrency: int,
    dry_run: bool,
) -> None:
    client: OpenAI = get_openai_client()
    state: State = load_state(state_path)
    state_lock: asyncio.Lock = asyncio.Lock()
    upload_lock: asyncio.Lock = asyncio.Lock()
    semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
    while True:
        try:
            feed_urls: List[str] = read_feed_urls(feeds_path)
            rss_index, atom_index, origin_map = await asyncio.to_thread(build_source_index, feed_urls)
            meta_entries: List[Dict] = await asyncio.to_thread(
                collect_meta_entries_from_index, rss_index, atom_index, origin_map, 10
            )
            meta_by_link: Dict[str, Dict] = {m.get("link", ""): m for m in meta_entries if m.get("link")}
            current_links: Set[str] = set(meta_by_link.keys())
            new_links: Set[str] = current_links - state.seen_links
            if new_links:
                async with state_lock:
                    state.seen_links.update(new_links)
                    save_state(state_path, state)
                tasks: List[asyncio.Task] = []
                for link in new_links:
                    meta = meta_by_link.get(link)
                    if not meta:
                        continue
                    t = asyncio.create_task(
                        classify_and_upload(
                            link,
                            meta,
                            client,
                            profile_path,
                            state,
                            state_path,
                            state_lock,
                            upload_lock,
                            rss_index,
                            atom_index,
                            origin_map,
                            dry_run,
                            semaphore,
                        )
                    )
                    tasks.append(t)
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patronus polling loop")
    parser.add_argument("--feeds-path", type=str, default="/Users/dani/code/patronus/feeds")
    parser.add_argument("--profile-path", type=str, default="/Users/dani/code/patronus/Profile.md")
    parser.add_argument("--state-path", type=str, default="/Users/dani/code/patronus/.patronus_poll_state.json")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(
            polling_loop(
                feeds_path=args.feeds_path,
                profile_path=args.profile_path,
                state_path=args.state_path,
                interval_seconds=args.interval_seconds,
                concurrency=args.concurrency,
                dry_run=args.dry_run,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


