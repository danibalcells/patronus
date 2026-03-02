# Patronus

A personal research and reading assistant. It aggregates content from dozens of RSS feeds, builds a daily inventory, and uses an LLM agent to compose a newspaper-style digest — personalized by live context from Notion. By default the digest is sent to Telegram and printed to the terminal; it can also be saved to Readwise Reader or published as an XML feed.

## How it works

Three scheduled jobs run each day:

**Ingestion** (every 2h, cron): `scripts/poll_feeds.py` polls active feeds, extracts full text with trafilatura, embeds with OpenAI, and stores items in SQLite.

**Notion mirror sync** (nightly, 2am cron): `scripts/sync_notion_mirror.py` incrementally syncs Notion pages (journal, work diary, library highlights) into a local SQLite mirror with an FTS5 full-text index. This keeps Notion API calls out of the digest hot path.

**Digest generation** (daily, 8am cron): `scripts/send_digest.py` runs `DigestPipeline`, which:

1. **Reader context** — fetches recent Notion entries (journal, work diary, library highlights), summarizes them into a prose context document via an LLM, and caches the result for 24h. The summary captures active projects, open questions, peripheral interests, and recurring emotional themes.

2. **Inventory** — queries all items ingested since the last digest, groups them by source feed, flags previously-featured items and URL-based story clusters. No LLM involved.

3. **Angles** (1 LLM call) — reads the inventory and reader context together, produces ~10–15 editorial hypotheses: what threads connect today's content to the reader's current thinking, which topics are over-saturated, which peripheral interests have hooks.

4. **News filter** (1 LLM call) — selects and summarizes news items from the inventory; consolidates story clusters; flags items that might also belong in research or threads.

5. **Research scout** (1–2 LLM calls with tools) — starts from inventory papers, issues parallel tool calls to deepen the view (related older work, OpenAlex, Notion search), and produces a curated paper list with summaries and Notion connection notes.

6. **Thread puller** (2–3 LLM calls with tools) — finds content worth the reader's time *right now* by following the angles and peripheral interest hooks. The only genuinely exploratory step: issues parallel tool calls per iteration, follows leads, surfaces both main-work relevance and serendipitous connections.

7. **Compose** (1 LLM call) — assembles the final digest from all upstream outputs, handles cross-section dedup, writes the final summaries, and delivers via `submit_digest`.

The agent path falls back to a deterministic rank → select → summarize pipeline if Notion context is unavailable or the agent produces an empty digest. Set `digest.mode: "deterministic"` in config to always use the deterministic path.

## Module structure

```
patronus/
├── config.py               # Config dataclasses + YAML/env loading
├── db.py                   # SQLite via SQLModel: Item, Feed, DigestRecord, ContextSnapshot
├── llm.py                  # Provider-agnostic LLM client (Anthropic, OpenAI, Google)
├── embed.py                # Embedding API wrapper
├── ingest.py               # Feed polling, dedup, text extraction, embedding storage
├── rank.py                 # Cosine similarity ranking + diversity selection (deterministic path)
├── summarize.py            # Per-item LLM summaries (deterministic path)
├── interests.py            # Static YAML interest vectors (PersonalizationSource + deterministic fallback)
├── context.py              # PersonalizationSource protocol, Context dataclass, merge_sources()
├── notion.py               # NotionSource: fetches, summarizes, and caches Notion context
├── notion_mirror.py        # Local SQLite mirror of Notion pages with FTS5 full-text search
├── digest.py               # Digest/DigestSection/DigestItem models; deterministic pipeline
├── pipeline.py             # DigestPipeline orchestrator: sources → agent → outputs
├── observability.py        # Langfuse tracing helpers
├── agent/
│   ├── _prompts.py         # All prompt strings and submit_digest tool schema
│   ├── _inventory.py       # Step 0: build_inventory() — zero LLM cost
│   ├── _steps.py           # Steps 2–3: identify_angles, filter_news, scout_research, pull_threads
│   ├── _compose.py         # Step 4: compose_digest() → Digest via submit_digest tool
│   └── run.py              # plan_and_assemble() orchestrator; Langfuse spans per step
├── tools/
│   ├── base.py             # Tool ABC, ToolResult dataclass
│   ├── __init__.py         # ToolRegistry
│   ├── local.py            # SearchSimilar, SearchRecent, SearchByTopic, SearchBySource
│   ├── arxiv.py            # SearchArxiv (ingests results into DB on retrieval)
│   ├── openalex.py         # SearchOpenAlex, GetCitingPapers, GetReferencedPapers
│   └── notion.py           # SearchNotion (queries the local mirror)
└── output/
    ├── __init__.py         # Output protocol
    ├── telegram.py         # MarkdownV2 formatting + Telegram delivery
    ├── terminal.py         # Pretty-printed stdout
    └── feed.py             # XML/Atom feed

scripts/
├── poll_feeds.py           # Cron: poll all active feeds
├── send_digest.py          # Cron: generate and deliver digest
├── add_feeds.py            # One-off: add feed URL(s) to Modal DB and poll them
├── list_feeds.py           # One-off: print all feeds in the local DB
├── seed_feeds.py           # One-off: seed DB from a feeds file (bootstrap only)
├── run_bot.py              # Systemd: Telegram bot (long-running)
├── sync_notion_mirror.py   # Nightly: sync Notion DBs to local mirror
└── test_notion_context.py  # Manual: fetch and print Notion context

config/
├── config.yaml
└── interests.yaml          # Static interest descriptions (deterministic fallback)
```

## Key abstractions

Three protocols are the only extension points. Everything else is concrete.

**`PersonalizationSource`** (`context.py`) — produces a prose context string for the agent. `InterestsSource` loads static YAML descriptions; `NotionSource` fetches live Notion content. `merge_sources()` concatenates prose from all available sources, skipping any that fail.

**`Tool`** (`tools/base.py`) — a retrieval action the agent can call. Each tool has a name, description, input schema, and `execute()` method. `ToolRegistry` produces the tool definitions list for the LLM API. Adding a tool = adding a file.

**`Output`** (`output/__init__.py`) — delivers a formatted digest. The pipeline dispatches to all configured outputs; each owns its own formatting.

## Config

```yaml
digest:
  mode: "agent"           # "agent" or "deterministic"
  size: 10                # used by deterministic path
  schedule: "08:00"
  timezone: "Europe/Madrid"

agent:
  model: "..."                  # default for all steps
  angles_model: "..."           # override per step
  news_model: "..."
  research_model: "..."
  threads_model: "..."
  compose_model: "..."
  notion_context_model: "..."
  digest_summary_model: "..."   # deterministic path only
  inventory_lookback_days: 2
  max_tokens: 5000

notion:
  database_ids:
    journal: "..."
    work_diary: "..."
    library: "..."
  lookback_days: 14
  cache_ttl_hours: 24
  mirror_path: "notion_mirror.sqlite3"   # empty = live API only

embedding:
  model: "..."

telegram:
  chat_id: "..."
```

```
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
TELEGRAM_BOT_TOKEN=...
NOTION_TOKEN=secret_...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

## Deployment

```
# crontab
0 */2 * * * cd /path/to/patronus && .venv/bin/python scripts/poll_feeds.py
0 2  * * * cd /path/to/patronus && .venv/bin/python scripts/sync_notion_mirror.py
0 8  * * * cd /path/to/patronus && .venv/bin/python scripts/send_digest.py
```

```ini
# systemd (Telegram bot)
[Service]
WorkingDirectory=/path/to/patronus
ExecStart=/path/to/patronus/.venv/bin/python scripts/run_bot.py
Restart=on-failure
```

```bash
# send_digest.py flags
--terminal-only          # print to stdout only, skip Telegram
--no-penalty             # ignore repeat penalty for previously digested items
--force-notion-refresh   # bypass Notion context cache
--feed                   # publish to RSS feed on R2
--reader                 # send to Readwise Reader

# add_feeds.py — runs ingestion on Modal, then syncs local DB
python scripts/add_feeds.py https://example.com/feed https://other.com/rss
python scripts/add_feeds.py --file feeds.txt   # one URL per line
python scripts/add_feeds.py https://... --no-sync  # skip local DB download

# list_feeds.py — inspect the local DB
python scripts/list_feeds.py
python scripts/list_feeds.py --all  # include inactive feeds
```

> Feed URLs are stored in the database only. The `feeds` file no longer exists;
> use `scripts/list_feeds.py` to inspect what's in the DB and `scripts/add_feeds.py`
> to add new ones.

## Dependency graph

No cycles.

```
config           ← (no internal deps)
db               ← (no internal deps)
context          ← (no internal deps)
notion_mirror    ← (no internal deps)
llm              ← config
embed            ← config, llm
rank             ← config, db
ingest           ← config, db, embed
interests        ← config, embed, context
summarize        ← config, llm
notion           ← config, llm, context, notion_mirror
tools/*          ← config, db, embed, rank
agent/*          ← config, llm, tools, digest
digest           ← config, db, interests, rank, summarize
pipeline         ← config, db, context, digest, agent, tools, output
output/*         ← config, digest
bot              ← config, db, ingest, pipeline
```

## Design decisions

- **Full visibility before editorial judgment.** The inventory gives the agent all items since the last digest before any LLM call. Nothing is invisible because the agent didn't think to search for it.
- **Separation of retrieval and editorial work.** Cheap/deterministic steps (inventory, news filter, angles) run first. More capable models are reserved for the steps that genuinely need judgment (research scout, thread puller, compose).
- **Peripheral interests get structural support.** The angles step explicitly extracts dormant curiosities from the reader context and turns them into editorial hypotheses. The thread puller is the only truly exploratory step and follows those hooks.
- **Interest vectors only in the deterministic fallback.** The agent path uses prose context and tool-based retrieval. Embeddings are computed at ingest time and used by local search tools, not by the agent directly.
- **Provider-agnostic model config.** `llm.py` routes `"provider/model"` strings. Each digest step has its own model config key; all fall back to `agent.model`. Switching the model for any step is a one-line config change.
- **Notion context is cached.** Fetching and summarizing Notion is expensive. Results are cached in the DB with a 24h TTL. On LLM failure the system falls back to stale cache rather than failing completely. `--force-notion-refresh` bypasses the cache.
- **Local Notion mirror eliminates API calls from the hot path.** `notion_mirror.py` maintains a SQLite copy with an FTS5 index. `NotionSource` reads from it instead of the live API; the mirror is synced nightly by `sync_notion_mirror.py`.
- **Three extension points, not a framework.** `PersonalizationSource`, `Tool`, and `Output` are the only abstractions. Adding a new context source, retrieval tool, or delivery channel doesn't require touching the pipeline.
- **Deterministic fallback preserved.** `digest.mode: "deterministic"` runs the original rank → select → summarize path. Also used automatically if Notion context is unavailable or the agent produces an empty digest.

## Development

```bash
uv sync
source .venv/bin/activate
pytest
```
