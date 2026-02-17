import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging

from patronus import setup_logging
from patronus.config import load_config
from patronus.notion import notion_retry, _rich_text_to_str


logger = logging.getLogger(__name__)

_PREVIEW_LEN = 80


def _block_preview(block: dict) -> str:
    block_type = block.get("type", "")
    data = block.get(block_type, {})
    if isinstance(data, dict):
        rich_text = data.get("rich_text", [])
        if rich_text:
            text = _rich_text_to_str(rich_text)
            preview = text[:_PREVIEW_LEN]
            return f'"{preview}"' if preview else ""
        expression = data.get("expression", "")
        if expression:
            return f'"{expression[:_PREVIEW_LEN]}"'
        url = data.get("url", "")
        if url:
            return url[:_PREVIEW_LEN]
    return ""


def _print_block_tree(
    client,
    block_id: str,
    depth: int = 0,
    raw: bool = False,
) -> None:
    indent = "  " * depth
    try:
        response = notion_retry(client.blocks.children.list)(block_id=block_id)
    except Exception as exc:
        print(f"{indent}  [ERROR fetching children: {exc}]")
        return

    blocks = response.get("results", [])
    has_more = response.get("has_more", False)

    for block in blocks:
        block_type = block.get("type", "unknown")
        has_children = block.get("has_children", False)
        preview = _block_preview(block)
        preview_str = f" | {preview}" if preview else ""
        children_str = f" has_children={has_children}"
        print(f"{indent}[{block_type}]{children_str}{preview_str}")

        if raw:
            raw_data = block.get(block_type, {})
            if raw_data:
                print(f"{indent}  raw: {json.dumps(raw_data, indent=None)[:200]}")

        if has_children:
            _print_block_tree(client, block["id"], depth + 1, raw=raw)

    if has_more:
        print(f"{indent}  ... (more blocks, pagination not shown)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the raw block tree for a Notion page"
    )
    parser.add_argument("page_id", help="Notion page ID (with or without dashes)")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw block data alongside the tree",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Dump full raw JSON of all top-level blocks",
    )
    args = parser.parse_args()

    setup_logging()

    config = load_config()
    if not config.notion_token:
        print("ERROR: NOTION_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    from notion_client import Client as NotionClient
    client = NotionClient(auth=config.notion_token)

    page_id = args.page_id.replace("-", "")
    formatted = f"{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}"

    print(f"Inspecting page: {formatted}")
    print()

    if args.json_output:
        response = notion_retry(client.blocks.children.list)(block_id=formatted)
        print(json.dumps(response, indent=2))
        return

    _print_block_tree(client, formatted, depth=0, raw=args.raw)


if __name__ == "__main__":
    main()
