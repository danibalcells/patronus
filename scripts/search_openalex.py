import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.tools.openalex import (
    GetCitingPapers,
    GetReferencedPapers,
    SearchOpenAlex,
    _FIELD_FILTERS,
)

DEFAULT_QUERY = "mechanistic interpretability"


def _print_results(result: object, n: int) -> None:
    if not result.items:
        print(result.message)
        return
    print(result.message)
    print("=" * 80)
    for i, item in enumerate(result.items, 1):
        print(f"\n[{i}] {item['title'] or '(no title)'}")
        print(f"    Authors:    {item['author'] or '—'}")
        print(f"    URL:        {item['url']}")
        print(f"    Published:  {item['timestamp'] or '—'}")
        print(f"    Citations:  {item['citation_count']}")
        if item.get("topics"):
            print(f"    Topics:     {', '.join(item['topics'])}")
        snippet = item["snippet"]
        print(f"    Abstract:   {snippet[:300]}{'…' if len(snippet) > 300 else ''}")
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search OpenAlex or explore citation graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s 'belief state geometry' --field ai\n"
            "  %(prog)s --cites 10.48550/arxiv.1706.03762 --sort citations\n"
            "  %(prog)s --references 10.48550/arxiv.1706.03762 -n 10\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--cites",
        metavar="DOI_OR_ID",
        help="Find papers that cite the given DOI or OpenAlex ID",
    )
    mode.add_argument(
        "--references",
        metavar="DOI_OR_ID",
        help="Find papers cited by the given DOI or OpenAlex ID (its bibliography)",
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Search query, used when neither --cites nor --references is given (default: '{DEFAULT_QUERY}')",
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
        default=None,
        help=(
            "Sort order. Search default: 'relevance'. "
            "--cites default: 'recency'. --references default: 'citations'."
        ),
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        dest="from_year",
        help="Restrict to papers published on or after this year (search and --cites only)",
    )
    parser.add_argument(
        "--field",
        choices=list(_FIELD_FILTERS.keys()),
        default="ai",
        help="Restrict to a discipline (search only, default: ai)",
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
        if args.cites:
            sort = args.sort or "recency"
            print(f"Citing papers for: {args.cites}")
            print(f"Results:  up to {args.n}  |  Sort: {sort}")
            if args.from_year:
                print(f"From:     {args.from_year} onwards")
            print()
            result = GetCitingPapers(config, db, embed=args.embed).execute(
                doi_or_id=args.cites,
                n=args.n,
                sort_by=sort,
                from_publication_year=args.from_year,
            )

        elif args.references:
            sort = args.sort or "citations"
            print(f"References of: {args.references}")
            print(f"Results:  up to {args.n}  |  Sort: {sort}")
            print()
            result = GetReferencedPapers(config, db, embed=args.embed).execute(
                doi_or_id=args.references,
                n=args.n,
                sort_by=sort,
            )

        else:
            sort = args.sort or "relevance"
            print(f"Query:    {args.query!r}")
            print(f"Results:  up to {args.n}  |  Sort: {sort}")
            if args.from_year:
                print(f"From:     {args.from_year} onwards")
            if args.field:
                print(f"Field:    {args.field}")
            print(f"Embed:    {args.embed}")
            print()
            result = SearchOpenAlex(config, db, embed=args.embed).execute(
                query=args.query,
                n=args.n,
                sort_by=sort,
                from_publication_year=args.from_year,
                field=args.field,
            )

        _print_results(result, args.n)


if __name__ == "__main__":
    main()
