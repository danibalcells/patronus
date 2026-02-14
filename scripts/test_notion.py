import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.notion import NotionSource


def main() -> None:
    setup_logging()
    config = load_config()
    db = Database()
    source = NotionSource(db=db)

    print("Fetching Notion context...")
    print(f"Databases: {list(config.notion.database_ids.keys())}")
    print(f"Lookback: {config.notion.lookback_days} days")
    print()

    context = source.get_context(config)

    if not context:
        print("No context generated (too few entries or Notion unavailable).")
        return

    print(f"Generated context ({len(context)} chars):")
    print("=" * 80)
    print(context)
    print("=" * 80)

    snapshot = db.get_latest_context_snapshot("notion")
    if snapshot:
        print(f"\nSaved to DB as snapshot {snapshot.id} at {snapshot.generated_at}")


if __name__ == "__main__":
    main()
