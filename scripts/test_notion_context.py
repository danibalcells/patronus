import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.notion import NotionSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Notion context and save to DB snapshot")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the 24h summary cache and regenerate from the mirror (or live API)",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    db = Database()
    source = NotionSource(db=db)

    mirror_path = config.notion.mirror_path if config.notion else ""
    print(f"Source:   {'mirror (' + mirror_path + ')' if mirror_path else 'live Notion API'}")
    print(f"Databases: {list(config.notion.database_ids.keys()) if config.notion else []}")
    print(f"Lookback:  {config.notion.lookback_days if config.notion else '?'} days")
    print(f"Cache:     {'BYPASSED (force-refresh)' if args.force_refresh else 'enabled (24h TTL)'}")
    print()

    context = source.get_context(config, force_refresh=args.force_refresh)

    if not context:
        print("No context generated (too few entries, mirror empty, or Notion unavailable).")
        return

    print(f"Generated context ({len(context)} chars):")
    print("=" * 80)
    print(context)
    print("=" * 80)

    snapshot = db.get_latest_context_snapshot("notion")
    if snapshot:
        print(f"\nSaved to DB: snapshot {snapshot.id} at {snapshot.generated_at}")


if __name__ == "__main__":
    main()
