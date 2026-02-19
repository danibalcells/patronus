import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.digest import Digest, DigestItem
from patronus.output.reader import ReaderOutput
from patronus.output.terminal import TerminalOutput
from patronus.pipeline import DigestPipeline


def _load_latest_digest(db: Database) -> Digest:
    records = db.get_latest_digests(1)
    if not records:
        raise SystemExit("No digest found in the database.")

    record = records[0]
    print(f"[no-regen] Loading digest from {record.generated_at} ({record.item_count} items)")

    item_records = db.get_digest_items(record.id)
    digest_items: list[DigestItem] = []
    for ir in item_records:
        item = db.get_item(ir.item_id)
        digest_items.append(DigestItem(
            item_id=ir.item_id,
            summary=ir.summary or "",
            title=item.title or "" if item else "",
            url=item.url if item else "",
            source=item.source or "" if item else "",
            author=item.author or "" if item else "",
        ))

    return Digest(items=digest_items, generated_at=record.generated_at, mode="deterministic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent pipeline manually against the production database")
    parser.add_argument("--force-notion-refresh", action="store_true", help="Bypass the 24h Notion context cache")
    parser.add_argument("--max-iterations", type=int, help="Override agent max_iterations from config")
    parser.add_argument("--no-regen", action="store_true", help="Skip regeneration; pull and display the latest digest from the DB")
    parser.add_argument("--reader", action="store_true", help="Deliver digest to Readwise Reader")
    parser.add_argument("--verbose", action="store_true", help="Show tool call results (sets patronus logger to DEBUG)")
    args = parser.parse_args()

    setup_logging()
    if args.verbose:
        logging.getLogger("patronus").setLevel(logging.DEBUG)

    config = load_config()
    if args.max_iterations and config.agent:
        config.agent.max_iterations = args.max_iterations

    outputs = [TerminalOutput()]
    if args.reader:
        outputs.append(ReaderOutput())

    with Database() as db:
        if args.no_regen:
            digest = _load_latest_digest(db)
            for output in outputs:
                output.send(digest, config)
        else:
            pipeline = DigestPipeline(config, db, outputs=outputs)
            pipeline.run(notion_force_refresh=args.force_notion_refresh)


if __name__ == "__main__":
    main()
