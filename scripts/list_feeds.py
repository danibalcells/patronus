import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from urllib.parse import urlparse

from patronus.db import Database


def _display_name(feed_name: str | None, url: str) -> str:
    if feed_name:
        return feed_name
    host = (urlparse(url).hostname or url).removeprefix("www.")
    return host


def main() -> None:
    parser = argparse.ArgumentParser(description="List feeds in the database")
    parser.add_argument("--db", type=str, default="db.sqlite3", help="Path to SQLite database")
    parser.add_argument("--all", action="store_true", dest="all_feeds", help="Include inactive feeds")
    args = parser.parse_args()

    with Database(db_path=args.db) as db:
        feeds = db.get_all_feeds() if args.all_feeds else db.get_active_feeds()

    if not feeds:
        print("No feeds found.")
        return

    col_name = 32
    col_cat = 14
    col_polled = 22

    header = f"{'NAME':<{col_name}} {'CATEGORY':<{col_cat}} {'LAST POLLED':<{col_polled}} URL"
    print(header)
    print("-" * (col_name + col_cat + col_polled + 60))

    for feed in feeds:
        name = _display_name(feed.name, feed.url)
        cat = feed.category or ""
        last_polled = (feed.last_polled or "never")[:19]
        active_marker = "" if feed.active else " [inactive]"
        print(f"{name + active_marker:<{col_name}} {cat:<{col_cat}} {last_polled:<{col_polled}} {feed.url}")

    print(f"\n{len(feeds)} feed(s) listed.")


if __name__ == "__main__":
    main()
