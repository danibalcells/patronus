import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from notion_client import Client as NotionClient
from tqdm import tqdm

from patronus import setup_logging
from patronus.config import load_config
from patronus.notion import notion_retry, resolve_data_source_id, _fetch_all_blocks

logger = logging.getLogger(__name__)


def _has_unsupported_block(client: NotionClient, page_id: str) -> bool:
    try:
        blocks = _fetch_all_blocks(client, page_id)
    except Exception:
        logger.warning("Failed to fetch blocks for page %s", page_id, exc_info=True)
        return False
    return any(b.get("type") == "unsupported" for b in blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List pages in a Notion DB that contain unsupported (transcription) blocks"
    )
    parser.add_argument(
        "--db",
        default="notes",
        help="Database name from config (default: notes)",
    )
    args = parser.parse_args()

    setup_logging(level=logging.WARNING)

    config = load_config()
    if config.notion is None:
        print("ERROR: no notion config found", file=sys.stderr)
        sys.exit(1)
    if not config.notion_token:
        print("ERROR: NOTION_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    db_id = config.notion.database_ids.get(args.db)
    if not db_id:
        available = ", ".join(config.notion.database_ids.keys())
        print(f"ERROR: db '{args.db}' not in config. Available: {available}", file=sys.stderr)
        sys.exit(1)

    client = NotionClient(auth=config.notion_token)
    cache: dict[str, str] = {}
    ds_id = resolve_data_source_id(client, db_id, cache)

    pages: list[dict] = []
    cursor = None
    with tqdm(desc=f"Fetching {args.db} pages", unit="page") as bar:
        while True:
            kwargs: dict = {
                "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
            }
            if cursor:
                kwargs["start_cursor"] = cursor
            response = notion_retry(client.data_sources.query)(data_source_id=ds_id, **kwargs)
            batch = response["results"]
            pages.extend(batch)
            bar.update(len(batch))
            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

    print(f"\nScanning {len(pages)} pages for unsupported blocks...\n")

    hits: list[str] = []
    for page in tqdm(pages, desc="Checking blocks", unit="page"):
        if _has_unsupported_block(client, page["id"]):
            hits.append(page.get("url", page["id"]))

    if hits:
        print(f"\nFound {len(hits)} page(s) with transcription blocks:\n")
        for url in hits:
            print(url)
    else:
        print("No pages with unsupported blocks found.")


if __name__ == "__main__":
    main()
