# Scripts

Entry points for running Patronus components. Each script is thin (~20-50 lines) and delegates to library functions in `patronus/`.

## Production Scripts

### `run_bot.py`
Start the Telegram bot (long-running process).

```bash
python scripts/run_bot.py
```

Run via systemd for automatic restart on failure.

### `send_digest.py`
Generate and send the daily digest.

```bash
# Normal daily digest (uses cached Notion context if fresh)
python scripts/send_digest.py

# Force fresh Notion context (bypass 24h cache)
python scripts/send_digest.py --force-notion-refresh

# Terminal output only (don't send to Telegram)
python scripts/send_digest.py --terminal-only
```

Typically run via cron at 8am daily.

### `poll_feeds.py`
Poll all active RSS/Atom feeds and ingest new items.

```bash
python scripts/poll_feeds.py
```

Run via cron every 2 hours.

### `seed_feeds.py`
One-off: Seed the database from a feeds file.

```bash
python scripts/seed_feeds.py path/to/feeds.txt
```

## Development & Testing Scripts

### `test_agent_manual.py`
**NEW:** Run the full pipeline manually with visible logging and formatted output. Uses the same DigestPipeline as production with real Notion context (cached by default).

```bash
# Basic run with test database
python scripts/test_agent_manual.py

# Force fresh Notion (bypass 24h cache)
python scripts/test_agent_manual.py --force-notion-refresh

# Use production database instead of test data
python scripts/test_agent_manual.py --use-prod-db

# Recreate test database with fresh data
python scripts/test_agent_manual.py --recreate-db

# Use specific database path
python scripts/test_agent_manual.py --db-path /tmp/test.db
```

**Use this for:**
- Debugging agent behavior with real personalization
- Seeing tool calls and LLM interactions
- Manual quality checks of digests
- Understanding pipeline flow
- Development iteration

**Why this is better than integration tests:**
- Uses full production pipeline (DigestPipeline)
- Real Notion context (same as send_digest.py)
- Shows all intermediate steps with logging
- Formatted, readable output
- Fast (uses 24h cache by default)
- Easy to run repeatedly

### `test_notion.py`
Manual test: Fetch and print Notion context.

```bash
python scripts/test_notion.py
```

## Design Pattern

All scripts follow this pattern:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patronus.config import load_config
from patronus.module import do_work

def main():
    config = load_config()
    do_work(config, ...)

if __name__ == "__main__":
    main()
```

No business logic in scripts — delegate to the library.

## See Also

- Integration tests: `tests/INTEGRATION_TESTS.md`
- Architecture: `README.md` in project root
