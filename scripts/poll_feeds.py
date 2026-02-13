import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from patronus.db import Database
from patronus.ingest import poll_feeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll active feeds and ingest new items")
    parser.add_argument("--db", type=str, default="db.sqlite3", help="Path to SQLite database")
    parser.add_argument("--limit", type=int, default=None, help="Max new entries per feed")
    parser.add_argument("--feed-limit", type=int, default=None, help="Max number of feeds to process")
    parser.add_argument("--dry-run", action="store_true", help="Limit to 2 entries per feed, 10 feeds max")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for feed fetching / text extraction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    limit = args.limit
    feed_limit = args.feed_limit
    if args.dry_run:
        if limit is None:
            limit = 2
        if feed_limit is None:
            feed_limit = 10

    with Database(db_path=args.db) as db:
        ids = poll_feeds(
            db,
            limit=limit,
            feed_limit=feed_limit,
            skip_embed=args.skip_embed,
            workers=args.workers,
        )
        print(f"Ingested {len(ids)} new item(s)")


if __name__ == "__main__":
    main()
