from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_FTS_TABLE = "pages_fts"
_SNIPPET_LENGTH = 300


@dataclass
class MirrorPage:
    id: str
    title: str
    content_snippet: str
    source_db: str
    created_at: str
    last_edited_at: str
    url: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NotionMirror:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
                id            TEXT PRIMARY KEY,
                title         TEXT NOT NULL DEFAULT '',
                content       TEXT NOT NULL DEFAULT '',
                source_db     TEXT NOT NULL DEFAULT '',
                url           TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL DEFAULT '',
                last_edited_at TEXT NOT NULL DEFAULT '',
                embedding     BLOB,
                synced_at     TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS sync_meta (
                source_db     TEXT PRIMARY KEY,
                last_synced_at TEXT NOT NULL DEFAULT ''
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                title,
                content,
                content='pages',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
                INSERT INTO pages_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
                INSERT INTO pages_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;
        """)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> NotionMirror:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def get_last_synced_at(self, source_db: str) -> str | None:
        row = self._conn.execute(
            "SELECT last_synced_at FROM sync_meta WHERE source_db = ?", (source_db,)
        ).fetchone()
        return row["last_synced_at"] if row else None

    def set_last_synced_at(self, source_db: str, ts: str) -> None:
        self._conn.execute(
            "INSERT INTO sync_meta (source_db, last_synced_at) VALUES (?, ?)"
            " ON CONFLICT(source_db) DO UPDATE SET last_synced_at = excluded.last_synced_at",
            (source_db, ts),
        )
        self._conn.commit()

    def upsert_page(
        self,
        *,
        page_id: str,
        title: str,
        content: str,
        source_db: str,
        url: str,
        created_at: str,
        last_edited_at: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        synced_at = _now_utc()
        self._conn.execute(
            """
            INSERT INTO pages (id, title, content, source_db, url, created_at, last_edited_at, embedding, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title          = excluded.title,
                content        = excluded.content,
                source_db      = excluded.source_db,
                url            = excluded.url,
                created_at     = excluded.created_at,
                last_edited_at = excluded.last_edited_at,
                embedding      = COALESCE(excluded.embedding, pages.embedding),
                synced_at      = excluded.synced_at
            """,
            (page_id, title, content, source_db, url, created_at, last_edited_at, embedding_blob, synced_at),
        )
        self._conn.commit()

    def page_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM pages").fetchone()
        return row[0]

    def latest_synced_at(self) -> str | None:
        row = self._conn.execute(
            "SELECT MAX(last_synced_at) FROM sync_meta"
        ).fetchone()
        return row[0] if row else None

    def is_stale(self, max_age_hours: int = 24) -> bool:
        ts = self.latest_synced_at()
        if ts is None:
            return True
        try:
            last = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            return age_hours > max_age_hours
        except (ValueError, AttributeError):
            return True

    # ------------------------------------------------------------------
    # Search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        source_dbs: Optional[list[str]] = None,
    ) -> list[MirrorPage]:
        params: list[object] = [query, limit]
        source_filter = ""
        if source_dbs:
            placeholders = ",".join("?" * len(source_dbs))
            source_filter = f"AND p.source_db IN ({placeholders})"
            params = [query] + list(source_dbs) + [limit]
            sql = f"""
                SELECT p.id, p.title, p.content, p.source_db, p.url, p.created_at, p.last_edited_at
                FROM pages_fts f
                JOIN pages p ON p.rowid = f.rowid
                WHERE pages_fts MATCH ?
                  {source_filter}
                ORDER BY rank
                LIMIT ?
            """
        else:
            sql = """
                SELECT p.id, p.title, p.content, p.source_db, p.url, p.created_at, p.last_edited_at
                FROM pages_fts f
                JOIN pages p ON p.rowid = f.rowid
                WHERE pages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_mirror_page(r) for r in rows]

    def get_recent(
        self,
        source_dbs: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[MirrorPage]:
        conditions = []
        params: list[object] = []

        if source_dbs:
            placeholders = ",".join("?" * len(source_dbs))
            conditions.append(f"source_db IN ({placeholders})")
            params.extend(source_dbs)

        if since is not None:
            conditions.append("last_edited_at >= ?")
            params.append(since.isoformat())

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT id, title, content, source_db, url, created_at, last_edited_at
            FROM pages
            {where}
            ORDER BY last_edited_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_row_to_mirror_page(r) for r in rows]


def _row_to_mirror_page(row: sqlite3.Row) -> MirrorPage:
    content: str = row["content"] or ""
    snippet = content[:_SNIPPET_LENGTH] + ("…" if len(content) > _SNIPPET_LENGTH else "")
    return MirrorPage(
        id=row["id"],
        title=row["title"] or "",
        content_snippet=snippet,
        source_db=row["source_db"] or "",
        created_at=row["created_at"] or "",
        last_edited_at=row["last_edited_at"] or "",
        url=row["url"] or "",
    )


def open_mirror(path: str) -> NotionMirror:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return NotionMirror(path)
