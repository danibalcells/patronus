from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy import LargeBinary, event, text
from sqlmodel import Field, Session, SQLModel, create_engine, select


def _new_id() -> str:
    return uuid.uuid4().hex


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


class Item(SQLModel, table=True):
    __tablename__ = "items"

    id: str = Field(default_factory=_new_id, primary_key=True)
    url: str = Field(unique=True, nullable=False, index=True)
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    source_type: str = Field(nullable=False, index=True)
    text: Optional[str] = None
    embedding: Optional[bytes] = Field(default=None, sa_type=LargeBinary)
    topic_cluster: Optional[str] = Field(default=None, index=True)
    timestamp: Optional[str] = Field(default=None, index=True)
    ingested_at: str = Field(default_factory=_now_utc)
    item_type: str = Field(default="article", index=True)
    source_item_id: Optional[str] = Field(default=None, foreign_key="items.id", index=True)
    read: bool = Field(default=False)
    digest_history: str = Field(default="[]")


class Feed(SQLModel, table=True):
    __tablename__ = "feeds"

    id: str = Field(default_factory=_new_id, primary_key=True)
    url: str = Field(unique=True, nullable=False, index=True)
    name: Optional[str] = None
    category: Optional[str] = None
    last_polled: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    active: bool = Field(default=True, index=True)


class DigestRecord(SQLModel, table=True):
    __tablename__ = "digests"

    id: str = Field(default_factory=_new_id, primary_key=True)
    generated_at: str = Field(default_factory=_now_utc)
    item_count: int = Field(default=0)
    formatted_text: Optional[str] = None


class DigestItemRecord(SQLModel, table=True):
    __tablename__ = "digest_items"

    id: str = Field(default_factory=_new_id, primary_key=True)
    digest_id: str = Field(foreign_key="digests.id", index=True)
    item_id: str = Field(index=True)
    summary: Optional[str] = None
    score: float = Field(default=0.0)
    matched_topic: Optional[str] = None


class ContextSnapshot(SQLModel, table=True):
    __tablename__ = "context_snapshots"

    id: str = Field(default_factory=_new_id, primary_key=True)
    source_type: str = Field(index=True)
    content: str = Field(default="")
    generated_at: str = Field(default_factory=_now_utc)


class Database:
    def __init__(self, db_path: str = "db.sqlite3") -> None:
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_conn: object, connection_record: object) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        SQLModel.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        with self.engine.connect() as conn:
            fks = conn.execute(text("PRAGMA foreign_key_list('digest_items')")).fetchall()
            has_item_id_fk = any(row[3] == "item_id" for row in fks)
            if not has_item_id_fk:
                return

            conn.execute(text("PRAGMA foreign_keys=OFF"))
            conn.execute(text("""
                CREATE TABLE digest_items_new (
                    id            TEXT PRIMARY KEY,
                    digest_id     TEXT NOT NULL REFERENCES digests(id),
                    item_id       TEXT NOT NULL,
                    summary       TEXT,
                    score         REAL NOT NULL DEFAULT 0.0,
                    matched_topic TEXT
                )
            """))
            conn.execute(text(
                "INSERT INTO digest_items_new "
                "SELECT id, digest_id, item_id, summary, score, matched_topic FROM digest_items"
            ))
            conn.execute(text("DROP TABLE digest_items"))
            conn.execute(text("ALTER TABLE digest_items_new RENAME TO digest_items"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_digest_items_digest_id ON digest_items(digest_id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_digest_items_item_id ON digest_items(item_id)"
            ))
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return Session(self.engine)

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def add_item(
        self,
        url: str,
        source_type: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        source: Optional[str] = None,
        text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        topic_cluster: Optional[str] = None,
        timestamp: Optional[str] = None,
        item_type: str = "article",
        source_item_id: Optional[str] = None,
    ) -> str:
        blob = serialize_embedding(embedding) if embedding is not None else None
        item = Item(
            url=url,
            source_type=source_type,
            title=title,
            author=author,
            source=source,
            text=text,
            embedding=blob,
            topic_cluster=topic_cluster,
            timestamp=timestamp,
            item_type=item_type,
            source_item_id=source_item_id,
        )
        with self._session() as session:
            session.add(item)
            session.commit()
            session.refresh(item)
            return item.id

    def get_item(self, item_id: str) -> Optional[Item]:
        with self._session() as session:
            return session.get(Item, item_id)

    def get_item_by_url(self, url: str) -> Optional[Item]:
        with self._session() as session:
            return session.exec(select(Item).where(Item.url == url)).first()

    def get_item_count(self) -> int:
        with self._session() as session:
            return len(session.exec(select(Item.id)).all())

    def get_unread_count(self) -> int:
        with self._session() as session:
            return len(session.exec(select(Item.id).where(Item.read == False)).all())

    def get_feed_count(self) -> int:
        with self._session() as session:
            return len(session.exec(select(Feed.id).where(Feed.active == True)).all())

    def get_unread_items(self) -> list[Item]:
        with self._session() as session:
            return list(
                session.exec(
                    select(Item)
                    .where(Item.read == False)
                    .order_by(Item.timestamp.desc())
                ).all()
            )

    def mark_read(self, item_id: str) -> None:
        with self._session() as session:
            item = session.get(Item, item_id)
            if item is None:
                return
            item.read = True
            session.add(item)
            session.commit()

    def store_embedding(self, item_id: str, embedding: np.ndarray) -> None:
        with self._session() as session:
            item = session.get(Item, item_id)
            if item is None:
                return
            item.embedding = serialize_embedding(embedding)
            session.add(item)
            session.commit()

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        with self._session() as session:
            item = session.get(Item, item_id)
            if item is None or item.embedding is None:
                return None
            return deserialize_embedding(item.embedding)

    def update_digest_history(self, item_id: str, date: str) -> None:
        with self._session() as session:
            item = session.get(Item, item_id)
            if item is None:
                return
            history: list[str] = json.loads(item.digest_history or "[]")
            history.append(date)
            item.digest_history = json.dumps(history)
            session.add(item)
            session.commit()

    # ------------------------------------------------------------------
    # Feeds
    # ------------------------------------------------------------------

    def add_feed(
        self,
        url: str,
        name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        feed = Feed(url=url, name=name, category=category)
        with self._session() as session:
            session.add(feed)
            session.commit()
            session.refresh(feed)
            return feed.id

    def get_feed(self, feed_id: str) -> Optional[Feed]:
        with self._session() as session:
            return session.get(Feed, feed_id)

    def get_feed_by_url(self, url: str) -> Optional[Feed]:
        with self._session() as session:
            return session.exec(select(Feed).where(Feed.url == url)).first()

    def get_active_feeds(self) -> list[Feed]:
        with self._session() as session:
            return list(
                session.exec(
                    select(Feed).where(Feed.active == True).order_by(Feed.name)
                ).all()
            )

    def update_feed_poll(
        self,
        feed_id: str,
        last_polled: Optional[str] = None,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> None:
        with self._session() as session:
            feed = session.get(Feed, feed_id)
            if feed is None:
                return
            feed.last_polled = last_polled or _now_utc()
            feed.etag = etag
            feed.last_modified = last_modified
            session.add(feed)
            session.commit()

    def deactivate_feed(self, feed_id: str) -> None:
        with self._session() as session:
            feed = session.get(Feed, feed_id)
            if feed is None:
                return
            feed.active = False
            session.add(feed)
            session.commit()

    def seed_feeds_from_file(self, path: str) -> int:
        count = 0
        with open(path) as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                if self.get_feed_by_url(url) is not None:
                    continue
                self.add_feed(url=url)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Digests
    # ------------------------------------------------------------------

    def save_digest(
        self,
        generated_at: str,
        item_count: int,
        formatted_text: Optional[str],
        items: list[dict[str, object]],
    ) -> str:
        digest = DigestRecord(
            generated_at=generated_at,
            item_count=item_count,
            formatted_text=formatted_text,
        )
        with self._session() as session:
            session.add(digest)
            session.flush()
            for entry in items:
                record = DigestItemRecord(
                    digest_id=digest.id,
                    item_id=str(entry["item_id"]),
                    summary=str(entry.get("summary") or ""),
                    score=float(entry.get("score", 0.0)),
                    matched_topic=str(entry.get("matched_topic") or ""),
                )
                session.add(record)
            session.commit()
            return digest.id

    def get_latest_digests(self, n: int = 5) -> list[DigestRecord]:
        with self._session() as session:
            return list(
                session.exec(
                    select(DigestRecord)
                    .order_by(DigestRecord.generated_at.desc())
                    .limit(n)
                ).all()
            )

    def get_digest_items(self, digest_id: str) -> list[DigestItemRecord]:
        with self._session() as session:
            return list(
                session.exec(
                    select(DigestItemRecord)
                    .where(DigestItemRecord.digest_id == digest_id)
                ).all()
            )

    # ------------------------------------------------------------------
    # Context Snapshots
    # ------------------------------------------------------------------

    def save_context_snapshot(self, source_type: str, content: str) -> str:
        snapshot = ContextSnapshot(source_type=source_type, content=content)
        with self._session() as session:
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
            return snapshot.id

    def get_latest_context_snapshot(self, source_type: str) -> Optional[ContextSnapshot]:
        with self._session() as session:
            return session.exec(
                select(ContextSnapshot)
                .where(ContextSnapshot.source_type == source_type)
                .order_by(ContextSnapshot.generated_at.desc())
                .limit(1)
            ).first()
