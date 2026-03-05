from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VOLUME = "patronus-data"
DB_FILENAME = "db.sqlite3"
MIRROR_FILENAME = "notion_mirror.sqlite3"


def fetch(filename: str, dest_dir: str | Path = ".") -> None:
    dest_dir = Path(dest_dir)
    logger.info("Fetching %s from Modal volume '%s'…", filename, VOLUME)
    subprocess.run(
        ["modal", "volume", "get", VOLUME, filename, str(dest_dir), "--force"],
        check=True,
    )
    logger.info("Fetched → %s", dest_dir / filename)


def push(local_path: str | Path, remote_filename: str | None = None) -> None:
    local_path = Path(local_path)
    remote = remote_filename or local_path.name
    logger.info("Pushing %s to Modal volume '%s'…", local_path, VOLUME)
    subprocess.run(
        ["modal", "volume", "put", VOLUME, str(local_path), remote, "--force"],
        check=True,
    )
    logger.info("Pushed → %s:%s", VOLUME, remote)


def fetch_db(dest_dir: str | Path = ".") -> None:
    fetch(DB_FILENAME, dest_dir)


def push_db(local_path: str | Path = DB_FILENAME) -> None:
    push(local_path, DB_FILENAME)


def fetch_mirror(dest_dir: str | Path = ".") -> None:
    fetch(MIRROR_FILENAME, dest_dir)


def push_mirror(local_path: str | Path = MIRROR_FILENAME) -> None:
    push(local_path, MIRROR_FILENAME)
