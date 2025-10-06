import asyncio
import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

import logging
from logging.handlers import SysLogHandler


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


logger: logging.Logger = logging.getLogger("patronus.polling_loop")


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


def _resolve_bucket_value(name: str) -> Optional[str]:
    try:
        # Exact match
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
    if not os.path.exists(path):
        return State()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        seen_list: List[str] = data.get("seen_links", [])
        assignments_raw: Dict[str, str] = data.get("assignments", {})
        assignments: Dict[str, str] = {}
        for link, bucket in assignments_raw.items():
            resolved: Optional[str] = _resolve_bucket_value(bucket)
            if resolved is not None:
                assignments[link] = resolved
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
        "REJECT, TECHNICAL_AI_AND_ML, TECH_BEYOND_THE_TECHNICAL, PHILOSOPHY_CONSCIOUSNESS, POLITICS_CULTURE, SPAIN, CHINA, RANDOM_CURIOSITIES."
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
        resolved_value: Optional[str] = _resolve_bucket_value(bucket_value)
        if resolved_value is None:
            continue
        bucket_key: Bucket = Bucket(resolved_value)
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


async def classify_one(
    link: str,
    meta: Dict,
    client: Any,
    profile_path: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str]:
    async with semaphore:
        logger.info("[classify] START link=%s", link)
        profile_text: str = await asyncio.to_thread(read_profile, profile_path)
        articles: List[Article] = await asyncio.to_thread(build_article_list, [meta], None)
        article: Article = articles[0]
        classification: ArticleClassification = await asyncio.to_thread(
            classify_article, client, system_preamble_text(), profile_text, article
        )
        bucket_value: str = classification.bucket.value
        logger.info("[classify] DONE bucket=%s link=%s", bucket_value, link)
        return link, bucket_value


async def polling_loop(
    feeds_path: str,
    profile_path: str,
    state_path: str,
    interval_seconds: int,
    concurrency: int,
    dry_run: bool,
) -> None:
    client: Any = get_openai_client()
    state: State = load_state(state_path)
    state_lock: asyncio.Lock = asyncio.Lock()
    upload_lock: asyncio.Lock = asyncio.Lock()
    semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
    while True:
        try:
            logger.info("[poll] cycle_start ts=%s", datetime.now().isoformat())
            feed_urls: List[str] = read_feed_urls(feeds_path)
            rss_index, atom_index, origin_map = await asyncio.to_thread(build_source_index, feed_urls)
            logger.info("[poll] indexed rss=%d atom=%d feeds=%d", len(rss_index), len(atom_index), len(feed_urls))
            meta_entries: List[Dict] = await asyncio.to_thread(
                collect_meta_entries_from_index, rss_index, atom_index, origin_map, 10
            )
            logger.info("[poll] collected_meta count=%d", len(meta_entries))
            meta_by_link: Dict[str, Dict] = {m.get("link", ""): m for m in meta_entries if m.get("link")}
            current_links: Set[str] = set(meta_by_link.keys())
            new_links: Set[str] = current_links - state.seen_links
            logger.info("[poll] new_links count=%d", len(new_links))
            if new_links:
                for link in new_links:
                    meta = meta_by_link.get(link)
                    if meta:
                        title: str = meta.get("title") or ""
                        logger.info("[new] title=%s link=%s", title, link)
                logger.info("[classify] start count=%d concurrency=%d", len(new_links), concurrency)
                tasks: List[asyncio.Task] = []
                for link in new_links:
                    meta = meta_by_link.get(link)
                    if not meta:
                        continue
                    t = asyncio.create_task(classify_one(link, meta, client, profile_path, semaphore))
                    tasks.append(t)
                done_count: int = 0
                new_assignments: Dict[str, str] = {}
                for coro in asyncio.as_completed(tasks):
                    link_res, bucket_val = await coro
                    new_assignments[link_res] = bucket_val
                    done_count += 1
                    logger.info("[classify] progress %d/%d", done_count, len(tasks))
                bucket_counts: Dict[str, int] = {}
                for _, b in new_assignments.items():
                    bucket_counts[b] = bucket_counts.get(b, 0) + 1
                logger.info("[classify] summary %s", bucket_counts)
                combined_assignments: Dict[str, str] = dict(state.assignments)
                combined_assignments.update(new_assignments)
                selection_counts: Dict[str, int] = {}
                for link, b in combined_assignments.items():
                    if link in rss_index or link in atom_index:
                        selection_counts[b] = selection_counts.get(b, 0) + 1
                logger.info("[upload] begin buckets=%s", selection_counts)
                upload_ok: bool = False
                try:
                    async with upload_lock:
                        await perform_upload(combined_assignments, rss_index, atom_index, origin_map, dry_run)
                    upload_ok = True
                except Exception:
                    logger.exception("[upload] failed")
                if upload_ok:
                    async with state_lock:
                        state.assignments.update(new_assignments)
                        state.seen_links.update(new_links)
                        save_state(state_path, state)
                    logger.info(
                        "[state] saved seen+=%d assignments+=%d total_assignments=%d",
                        len(new_links),
                        len(new_assignments),
                        len(state.assignments),
                    )
                    logger.info("[upload] done")
                else:
                    logger.warning("[state] not saved due to upload failure")
            logger.info("[poll] cycle_end ts=%s", datetime.now().isoformat())
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("[poll] cycle_error")
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
    setup_logging()
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


