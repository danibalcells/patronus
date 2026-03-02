import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.tools.arxiv import SearchArxiv

DEFAULT_QUERY = "mechanistic interpretability"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Arxiv and print results")
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
    parser.add_argument(
        "--sort",
        choices=["relevance", "recency"],
        default="relevance",
        help="Sort order: 'relevance' (default) or 'recency'",
    )
    parser.add_argument(
        "--category",
        default="",
        help="Restrict to an Arxiv category, e.g. cs.LG, cs.AI, stat.ML",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Restrict to papers submitted in the last N days",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Embed abstracts when ingesting (default: off)",
    )
    parser.add_argument(
        "--db",
        default="db.sqlite3",
        help="Path to SQLite database (default: db.sqlite3)",
    )
    args = parser.parse_args()

    setup_logging()

    tool = SearchArxiv()

    print(f"Query:    {args.query!r}")
    print(f"Results:  up to {args.n}")
    print(f"Sort:     {args.sort}")
    if args.category:
        print(f"Category: {args.category}")
    if args.days:
        print(f"Days:     last {args.days}")
    print(f"Embed:    {args.embed}")
    print()

    result = tool.execute(
        query=args.query,
        n=args.n,
        sort_by=args.sort,
        category=args.category or None,
        days=args.days,
    )

    if not result.items:
        print(result.message)
        return

    print(result.message)
    print("=" * 80)
    for i, item in enumerate(result.items, 1):
        print(f"\n[{i}] {item['title']}")
        print(f"    Authors:    {item['author'] or '—'}")
        print(f"    URL:        {item['url']}")
        print(f"    Published:  {item['timestamp'] or '—'}")
        if item.get("categories"):
            print(f"    Categories: {', '.join(item['categories'])}")
        snippet = item["snippet"]
        print(f"    Abstract:   {snippet[:300]}{'…' if len(snippet) > 300 else ''}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
