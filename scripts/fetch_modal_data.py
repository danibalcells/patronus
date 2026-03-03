import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess

VOLUME = "patronus-data"
FILES = {
    "db": "db.sqlite3",
    "mirror": "notion_mirror.sqlite3",
}


def fetch(filename: str, dest: Path) -> None:
    print(f"Fetching {filename} from Modal volume '{VOLUME}'...")
    subprocess.run(
        ["modal", "volume", "get", VOLUME, filename, str(dest), "--force"],
        check=True,
    )
    print(f"  -> saved to {dest / filename}")


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

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.db_only and args.mirror_only:
        parser.error("--db-only and --mirror-only are mutually exclusive")

    targets = list(FILES.values())
    if args.db_only:
        targets = [FILES["db"]]
    elif args.mirror_only:
        targets = [FILES["mirror"]]

    for filename in targets:
        fetch(filename, dest)

    print("Done.")


if __name__ == "__main__":
    main()
