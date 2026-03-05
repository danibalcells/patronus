import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from sqlalchemy import text

from patronus.db import Database
from patronus.modal_volume import fetch_db, push_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MERGEABLE_FIELDS = [
    "title",
    "author",
    "source",
    "text",
    "embedding",
    "topic_cluster",
    "timestamp",
    "item_type",
    "source_item_id",
]


def _abs_to_pdf(url: str) -> str:
    return url.replace("arxiv.org/abs/", "arxiv.org/pdf/", 1)


def migrate(db_path: str, dry_run: bool = False) -> None:
    db = Database(db_path)

    with db.engine.connect() as conn:
        cols = "id, url, " + ", ".join(_MERGEABLE_FIELDS)
        abs_rows = conn.execute(
            text(f"SELECT {cols} FROM items WHERE url LIKE '%arxiv.org/abs/%'")
        ).mappings().fetchall()

        logger.info("Found %d arxiv /abs/ items", len(abs_rows))

        renamed = 0
        merged = 0

        for abs_row in abs_rows:
            abs_id = abs_row["id"]
            abs_url = abs_row["url"]
            pdf_url = _abs_to_pdf(abs_url)

            pdf_row = conn.execute(
                text(f"SELECT {cols} FROM items WHERE url = :url"),
                {"url": pdf_url},
            ).mappings().fetchone()

            if pdf_row is None:
                logger.info("Rename  %s", abs_url)
                if not dry_run:
                    conn.execute(
                        text("UPDATE items SET url = :pdf_url WHERE id = :id"),
                        {"pdf_url": pdf_url, "id": abs_id},
                    )
                renamed += 1
            else:
                pdf_id = pdf_row["id"]
                updates: dict[str, object] = {}
                for field in _MERGEABLE_FIELDS:
                    if pdf_row[field] is None and abs_row[field] is not None:
                        updates[field] = abs_row[field]

                if updates:
                    logger.info(
                        "Merge   %s ← fields %s from abs entry",
                        pdf_url,
                        list(updates.keys()),
                    )
                    if not dry_run:
                        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
                        updates["_id"] = pdf_id
                        conn.execute(
                            text(f"UPDATE items SET {set_clause} WHERE id = :_id"),
                            updates,
                        )
                else:
                    logger.info("Dedup   %s (no new fields to merge)", pdf_url)

                if not dry_run:
                    conn.execute(
                        text(
                            "UPDATE digest_items SET item_id = :pdf_id"
                            " WHERE item_id = :abs_id"
                        ),
                        {"pdf_id": pdf_id, "abs_id": abs_id},
                    )
                    conn.execute(
                        text(
                            "UPDATE items SET source_item_id = :pdf_id"
                            " WHERE source_item_id = :abs_id"
                        ),
                        {"pdf_id": pdf_id, "abs_id": abs_id},
                    )
                    conn.execute(
                        text("DELETE FROM items WHERE id = :id"),
                        {"id": abs_id},
                    )
                merged += 1

        if not dry_run:
            conn.commit()

    logger.info(
        "Done%s — %d renamed, %d deduplicated",
        " (dry run)" if dry_run else "",
        renamed,
        merged,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate arXiv /abs/ URLs to /pdf/ in the local SQLite DB."
    )
    parser.add_argument("--db", default="db.sqlite3", help="Path to SQLite DB file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing anything",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download the DB from the Modal volume before migrating",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the DB back to the Modal volume after migrating",
    )
    args = parser.parse_args()

    if args.fetch:
        fetch_db(Path(args.db).resolve().parent)

    migrate(args.db, dry_run=args.dry_run)

    if args.push:
        if args.dry_run:
            logger.info("Dry run — skipping push.")
        else:
            push_db(args.db)


if __name__ == "__main__":
    main()
