import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from patronus.config import load_config
from patronus.db import Database
from patronus.telegram import send_digest_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and send the daily digest via Telegram")
    parser.add_argument("--db", type=str, default="db.sqlite3", help="Path to SQLite database")
    parser.add_argument("--no-penalty", action="store_true", help="Ignore repeat penalty for already-digested items")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    config = load_config()
    with Database(db_path=args.db) as db:
        send_digest_message(config, db, skip_penalty=args.no_penalty)


if __name__ == "__main__":
    main()
