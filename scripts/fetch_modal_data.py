import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus.modal_volume import fetch_db, fetch_mirror


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the latest db.sqlite3 and notion_mirror.sqlite3 from the Modal volume"
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Only fetch the main database (db.sqlite3)",
    )
    parser.add_argument(
        "--mirror-only",
        action="store_true",
        help="Only fetch the Notion mirror (notion_mirror.sqlite3)",
    )
    parser.add_argument(
        "--dest",
        metavar="DIR",
        default=".",
        help="Local directory to write files into (default: current directory)",
    )
    args = parser.parse_args()

    if args.db_only and args.mirror_only:
        parser.error("--db-only and --mirror-only are mutually exclusive")

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not args.mirror_only:
        fetch_db(dest)
    if not args.db_only:
        fetch_mirror(dest)

    print("Done.")


if __name__ == "__main__":
    main()
