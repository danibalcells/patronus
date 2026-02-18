import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.tools.openalex import SearchOpenAlex, _FIELD_FILTERS

DEFAULT_QUERY = "mechanistic interpretability"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search OpenAlex and print results")
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
        choices=["relevance", "citations", "recency"],
        default="relevance",
        help="Sort order: 'relevance' (default), 'citations', or 'recency'",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        dest="from_year",
        help="Restrict to papers published on or after this year",
    )
    parser.add_argument(
        "--field",
        choices=list(_FIELD_FILTERS.keys()),
        default="ai",
        help="Restrict to a discipline (default: ai)",
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
    config = load_config()

    if not config.openalex_api_key:
        print("Warning: OPENALEX_API_KEY not set — limited to 100 credits/day.")
        print("Get a free key at https://openalex.org/settings/api\n")

    with Database(db_path=args.db) as db:
        tool = SearchOpenAlex(config, db, embed=args.embed)

        print(f"Query:    {args.query!r}")
        print(f"Results:  up to {args.n}")
        print(f"Sort:     {args.sort}")
        if args.from_year:
            print(f"From:     {args.from_year} onwards")
        if args.field:
            print(f"Field:    {args.field}")
        print(f"Embed:    {args.embed}")
        print()

        result = tool.execute(
            query=args.query,
            n=args.n,
            sort_by=args.sort,
            from_publication_year=args.from_year,
            field=args.field,
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
            print(f"    Citations:  {item['citation_count']}")
            if item.get("topics"):
                print(f"    Topics:     {', '.join(item['topics'])}")
            snippet = item["snippet"]
            print(f"    Abstract:   {snippet[:300]}{'…' if len(snippet) > 300 else ''}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
