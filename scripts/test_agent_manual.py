import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from patronus import setup_logging
from patronus.config import load_config
from patronus.db import Database
from patronus.output.terminal import TerminalOutput
from patronus.pipeline import DigestPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent pipeline manually against the production database")
    parser.add_argument("--force-notion-refresh", action="store_true", help="Bypass the 24h Notion context cache")
    parser.add_argument("--max-iterations", type=int, help="Override agent max_iterations from config")
    parser.add_argument("--verbose", action="store_true", help="Show tool call results (sets patronus logger to DEBUG)")
    args = parser.parse_args()

    setup_logging()
    if args.verbose:
        logging.getLogger("patronus").setLevel(logging.DEBUG)

    config = load_config()
    if args.max_iterations and config.agent:
        config.agent.max_iterations = args.max_iterations

    with Database() as db:
        pipeline = DigestPipeline(config, db, outputs=[TerminalOutput()])
        pipeline.run(notion_force_refresh=args.force_notion_refresh)


if __name__ == "__main__":
    main()
