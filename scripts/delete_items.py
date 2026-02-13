import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from sqlmodel import select

from patronus.db import Database, Item


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all items from the database")
    parser.add_argument("--db", type=str, default="db.sqlite3", help="Path to SQLite database")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    with Database(db_path=args.db) as db:
        with db._session() as session:
            count = len(session.exec(select(Item)).all())

        if count == 0:
            print("No items in the database.")
            return

        if not args.force:
            answer = input(f"Delete all {count} item(s)? [y/N] ")
            if answer.lower() != "y":
                print("Aborted.")
                return

        with db._session() as session:
            for item in session.exec(select(Item)).all():
                session.delete(item)
            session.commit()

        print(f"Deleted {count} item(s).")


if __name__ == "__main__":
    main()
