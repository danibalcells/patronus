# Patronus

A personal research and reading assistant that aggregates content from RSS/Atom feeds, ranks it by relevance to current intellectual interests using embedding similarity, and delivers a curated daily digest via Telegram.

The system monitors dozens of feeds spanning technical ML research, tech commentary, philosophy, linguistics, and more. It filters at volume so you don't have to — embedding-based ranking (not LLM-as-judge) selects ~7 high-signal items per day, and an LLM generates contextual summaries only after selection.

The project is implemented in stages, each independently useful.

## Stage 1 architecture

### Module structure

```
patronus/                   # Core library — all business logic lives here
├── __init__.py
├── config.py               # Load YAML config + env vars → Config dataclass
├── db.py                   # SQLModel models (Item, Feed) + Database class
├── ingest.py               # Feed polling, content extraction, manual URL add
├── embed.py                # Embedding API wrapper (text-embedding-3-small)
├── interests.py            # Load interest vectors (Stage 1: YAML→embed)
├── rank.py                 # Cosine similarity, recency boost, diversity selection
├── summarize.py            # Claude API for contextual summaries
├── digest.py               # Orchestrator: rank → select → summarize → Digest
└── telegram.py             # Bot handlers + direct message sending

scripts/                    # Thin entry points — each is ~20 lines
├── poll_feeds.py           # Cron: poll all active feeds, ingest new items
├── send_digest.py          # Cron: generate digest, send via Telegram
├── seed_feeds.py           # One-off: seed DB from feeds file
└── run_bot.py              # Systemd: start the Telegram bot (long-running)

config/
├── config.yaml             # Schedule, digest size, model names, Telegram chat ID
└── interests.yaml          # Static interest descriptions (one paragraph per topic)

tests/
└── ...
```

### Module responsibilities

**`config.py`** — Single source of truth for runtime configuration. Loads `config/config.yaml` for schedule, model settings, Telegram chat ID, digest size, etc. Loads `config/interests.yaml` for per-topic interest descriptions. API keys come from env vars (`.env`), not YAML. Exposes a `Config` dataclass that other modules accept as a parameter.

**`db.py`** — SQLModel models (`Item`, `Feed`) and a `Database` class that wraps all SQLite access. Feed URLs are seeded from a file but the DB is the source of truth for feed state. No other module touches SQLite directly.

**`embed.py`** — Thin wrapper around the OpenAI embedding API. Exposes `embed_text()` and `embed_batch()`. Stateless — callers manage caching. Easy to swap models later by changing only this module.

**`interests.py`** — Loads interest descriptions and produces embedding vectors. In Stage 1, reads from `config/interests.yaml` and embeds via `embed.py`. In Stage 2, this gets swapped for Notion-derived centroids. This module is the seam between stages — downstream code (`rank.py`, `digest.py`) receives `dict[str, np.ndarray]` and doesn't care where it came from.

**`ingest.py`** — Feed polling and content extraction. `poll_feeds()` iterates active feeds from the DB, parses with `feedparser`, deduplicates by URL, extracts full text with `trafilatura`, embeds via `embed.py`, and stores via `db.add_item()`. `ingest_url()` handles manual URL adds (same pipeline, single URL). Returns new item IDs so post-ingest hooks (e.g., Stage 3 interrupt alerts) can be added without structural changes.

**`rank.py`** — Ranking and selection. `rank_unread()` loads all unread items, computes cosine similarity against each topic centroid (max across centroids), applies a gentle recency boost, and returns a sorted list of `ScoredItem`s. `select_digest()` applies a topic diversity constraint (no more than ~3 from any single cluster) and returns the top N. All numpy, no API calls.

**`summarize.py`** — Calls the Claude API to generate 2–3 sentence contextual summaries for selected items, given the matched interest description as context. One API call per item.

**`digest.py`** — Orchestrates the full daily digest pipeline: load interest vectors → rank unread items → select top ~7 → summarize each → build a `Digest` dataclass. Also handles formatting for Telegram delivery. Both the `/digest` bot command and the cron script call the same `generate_digest()` function.

**`telegram.py`** — Telegram bot using `python-telegram-bot`. Handles `/add <url>` (triggers `ingest.ingest_url()`), `/digest` (triggers `digest.generate_digest()`), and `/status`. Also exposes `send_digest()` for the cron script to send messages without a running bot process.

### Data flow

```
┌─────────────────────────────────────────────────────────┐
│                        SCHEDULING                        │
│                                                          │
│  Cron (every 2h):         Cron (daily 8am):              │
│  scripts/poll_feeds.py    scripts/send_digest.py         │
│         │                        │                       │
│         ▼                        ▼                       │
│    ingest.poll_feeds()     digest.generate_digest()      │
│         │                        │                       │
│    ┌────┴────┐          ┌────────┼──────────┐            │
│    ▼         ▼          ▼        ▼          ▼            │
│  embed    db.add    interests  rank    summarize         │
│    │      _item()   .load()  .rank()  .summarize()       │
│    │         │          │        │          │             │
│    │         ▼          │        │          │             │
│    └──────►SQLite◄──────┘────────┘          │             │
│                                             ▼            │
│  Systemd (long-running):              telegram           │
│  scripts/run_bot.py                   .send_digest()     │
│    /add → ingest.ingest_url()                            │
│    /digest → digest.generate_digest()                    │
│    /status → db queries                                  │
└─────────────────────────────────────────────────────────┘
```

### Dependency graph

No cycles. Each module depends only on modules above it in this list:

```
config         ← (no internal deps)
db             ← (no internal deps)
embed          ← config
interests      ← config, embed
ingest         ← config, db, embed
rank           ← config, db
summarize      ← config
digest         ← config, db, embed, interests, rank, summarize
telegram       ← config, db, ingest, digest
```

### Config structure

**`config/config.yaml`:**
```yaml
digest:
  size: 7
  max_per_topic: 3
  schedule: "08:00"
  timezone: "Europe/Madrid"

polling:
  interval_hours: 2

embedding:
  model: "text-embedding-3-small"

summarization:
  model: "claude-sonnet-4-20250514"

telegram:
  chat_id: "..."
```

**`config/interests.yaml`:**
```yaml
topics:
  technical_ml:
    name: "Technical AI/ML"
    description: |
      Technical machine learning research, especially mechanistic
      interpretability, training dynamics, ...
  tech_strategy:
    name: "Tech & Strategy"
    description: |
      ...
```

**Environment variables (`.env`):**
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
```

### Scripts and deployment

Each script in `scripts/` is a thin entry point that loads config, opens the DB, calls a library function, and exits. Scripts add the project root to `sys.path` so `import patronus` works without installation.

**Crontab:**
```
0 */2 * * * cd /path/to/patronus && .venv/bin/python scripts/poll_feeds.py
0 8  * * * cd /path/to/patronus && .venv/bin/python scripts/send_digest.py
```

**Systemd (Telegram bot):**
```ini
[Service]
WorkingDirectory=/path/to/patronus
ExecStart=/path/to/patronus/.venv/bin/python scripts/run_bot.py
Restart=on-failure
```

### Design decisions

- **Embeddings for ranking, LLM for summaries only.** LLM-as-judge always finds a reason something is relevant. Embedding similarity measures geometric proximity to actual intellectual activity without confabulating.
- **SQLModel over raw sqlite3.** The schema spec called for raw sqlite3, but SQLModel was chosen during implementation — it's thin enough (typed access over SQLite) and already done with tests.
- **scripts/ + cron over APScheduler.** The scheduled jobs are independent and short-lived. Cron is transparent, debuggable, and doesn't require an extra dependency. The Telegram bot is the only long-running process.
- **`interests.py` as the stage boundary.** This module is the seam between Stage 1 (static YAML descriptions) and Stage 2 (live Notion centroids). Everything downstream receives `dict[str, np.ndarray]` and doesn't care where it came from.
- **Deliberately lossy.** The system does not try to ensure you see everything. It tries to ensure that what you see is worth your time.

## Development

### Setup
```bash
uv sync
source .venv/bin/activate
```

### Tests
```bash
pytest
```

### Legacy code
The `old/` folder contains Patronus v1 — a many-to-many RSS filter that classified articles into topic buckets and published curated feeds to GCS. That approach was replaced by the embedding-based system described above.
