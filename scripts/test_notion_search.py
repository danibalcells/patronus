import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus import setup_logging
from patronus.config import load_config
from patronus.tools.notion import SearchNotion

DEFAULT_QUERY = "machine learning interpretability"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local Notion mirror and print results")
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Search query (default: '{DEFAULT_QUERY}')",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    if not config.notion or not config.notion.mirror_path:
        print("Error: notion.mirror_path is not configured in config.yaml")
        sys.exit(1)

    mirror_path = Path(config.notion.mirror_path)
    print(f"Mirror:  {mirror_path} ({'exists' if mirror_path.exists() else 'NOT FOUND'})")
    print(f"Query:   {args.query!r}")
    print(f"Results: up to {args.n}")
    print()

    tool = SearchNotion(config)
    result = tool.execute(query=args.query, n=args.n)

    if not result.items:
        print(result.message)
        return

    print(result.message)
    print("=" * 80)
    for i, item in enumerate(result.items, 1):
        print(f"\n[{i}] {item['title']}")
        print(f"    Source:  {item['source']}")
        print(f"    Edited:  {item['timestamp']}")
        print(f"    URL:     {item['url']}")
        print(f"    Snippet: {item['snippet'][:200]}{'…' if len(item['snippet']) > 200 else ''}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
