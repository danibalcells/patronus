import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.bot import run_bot


def main() -> None:
    setup_logging()

    config = load_config()
    db = Database()
    try:
        run_bot(config, db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
