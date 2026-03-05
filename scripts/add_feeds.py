import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import modal

from patronus.modal_volume import fetch_db


def _collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = list(args.urls)
    if args.file:
        with open(args.file) as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith("#"):
                    urls.append(url)
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add feeds to the Modal DB and poll them for new items"
    )
    parser.add_argument("urls", nargs="*", metavar="URL", help="Feed URL(s) to add")
    parser.add_argument("--file", "-f", metavar="FILE", help="File with one feed URL per line")
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip downloading the updated DB locally after the run",
    )
    args = parser.parse_args()

    urls = _collect_urls(args)
    if not urls:
        parser.error("Provide at least one URL as an argument or via --file")

    print(f"Sending {len(urls)} feed(s) to Modal...")
    f = modal.Function.from_name("patronus", "add_feeds")
    result = f.remote(urls)
    print(f"Done — added {result['added']} new feed(s), ingested {result['ingested']} new item(s)")

    if not args.no_sync:
        print("Syncing updated DB locally...")
        fetch_db()
        print("Local DB updated.")


if __name__ == "__main__":
    main()
