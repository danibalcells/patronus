import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from patronus.config import load_config
from patronus.db import Database
from patronus.telegram import run_bot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    config = load_config()
    db = Database()
    try:
        run_bot(config, db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
