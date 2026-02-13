import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus.db import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed feeds from a text file (one URL per line)")
    parser.add_argument("file", type=str, help="Path to feeds file")
    parser.add_argument("--db", type=str, default="db.sqlite3", help="Path to SQLite database")
    args = parser.parse_args()

    with Database(db_path=args.db) as db:
        count = db.seed_feeds_from_file(args.file)
        print(f"Added {count} new feed(s)")


if __name__ == "__main__":
    main()
