import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.output.feed import FeedOutput
from patronus.output.reader import ReaderOutput
from patronus.output.telegram import TelegramOutput
from patronus.output.terminal import TerminalOutput
from patronus.pipeline import DigestPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and send the daily digest")
    parser.add_argument("--db", type=str, default="db.sqlite3", help="Path to SQLite database")
    parser.add_argument("--no-penalty", action="store_true", help="Ignore repeat penalty for already-digested items")
    parser.add_argument("--terminal-only", action="store_true", help="Print to terminal only, don't send to Telegram")
    parser.add_argument("--force-notion-refresh", action="store_true", help="Force refresh Notion context, bypassing cache")
    parser.add_argument("--feed", action="store_true", help="Upload digest to RSS feed on R2")
    parser.add_argument("--feed-tag", type=str, default="", help="Tag for the RSS feed file (e.g. 'test1' → feed-test1.xml)")
    parser.add_argument("--reader", action="store_true", help="Deliver digest to Readwise Reader")
    args = parser.parse_args()

    setup_logging()

    config = load_config()
    outputs = []
    if args.terminal_only:
        outputs.append(TerminalOutput())
    else:
        outputs.append(TelegramOutput())
        outputs.append(TerminalOutput())

    if args.feed:
        outputs.append(FeedOutput(tag=args.feed_tag or None))

    if args.reader:
        outputs.append(ReaderOutput())

    with Database(db_path=args.db) as db:
        pipeline = DigestPipeline(config, db, outputs=outputs)
        pipeline.run(skip_penalty=args.no_penalty, notion_force_refresh=args.force_notion_refresh)


if __name__ == "__main__":
    main()
