# Patronus

A personal research and reading assistant that aggregates content from multiple sources, filters at volume, and delivers a curated daily digest. The system monitors dozens of feeds spanning technical ML research, tech commentary, philosophy, linguistics, and more — selecting high-signal items and summarizing them so you don't have to triage manually.

The project is implemented in stages, each independently useful. Stage 1 uses a deterministic embedding-based pipeline. Stage 2 replaces the fixed pipeline with an LLM agent that plans a newspaper-style digest using retrieval tools, personalized by live context from Notion.

## Stage 1 architecture - being phased out

Stage 1 is a deterministic pipeline: embed items, rank by cosine similarity against static interest descriptions, select top N with diversity constraints, summarize with an LLM, deliver via Telegram.

### Module structure

```
patronus/                   # Core library — all business logic lives here
├── __init__.py
├── config.py               # Load YAML config + env vars → Config dataclass
├── db.py                   # SQLModel models (Item, Feed) + Database class
├── ingest.py               # Feed polling, content extraction, manual URL add
├── embed.py                # Embedding API wrapper (text-embedding-3-small)
├── interests.py            # Load interest vectors (YAML → embed)
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

**`embed.py`** — Thin wrapper around the OpenAI embedding API. Exposes `embed_text()` and `embed_batch()`. Stateless — callers manage caching.

**`interests.py`** — Loads interest descriptions from `config/interests.yaml` and produces embedding vectors via `embed.py`. Downstream code (`rank.py`, `digest.py`) receives `dict[str, np.ndarray]` and doesn't care where it came from.

**`ingest.py`** — Feed polling and content extraction. `poll_feeds()` iterates active feeds from the DB, parses with `feedparser`, deduplicates by URL, extracts full text with `trafilatura`, embeds via `embed.py`, and stores via `db.add_item()`. `ingest_url()` handles manual URL adds (same pipeline, single URL).

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

**Daily digest script:**
```bash
python scripts/send_digest.py [--db PATH] [--terminal-only] [--no-penalty] [--force-notion-refresh]
```
- `--db`: Path to SQLite database (default: `db.sqlite3`)
- `--terminal-only`: Print to terminal only, don't send to Telegram
- `--no-penalty`: Ignore repeat penalty for already-digested items
- `--force-notion-refresh`: Force refresh Notion context, bypassing cache

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

## Stage 2 architecture - being implemented

Stage 2 replaces the deterministic rank-select-summarize pipeline with an LLM agent that acts as an editor. The agent receives personalization context (from Notion and/or static interests), plans a newspaper-style digest with typed sections, and uses retrieval tools to find items for each section. Output is decoupled from generation — the same digest can be sent to Telegram, printed to terminal, or published as an XML feed.

### Module structure

```
patronus/
├── __init__.py
├── config.py               # Extended: models, notion, output settings
├── db.py                   # Extended: new query methods for agent tools
├── llm.py                  # NEW — provider-agnostic LLM completion + tool use
├── embed.py                # Changed — provider routing (same interface)
├── ingest.py               # Unchanged (RSS/Atom only)
├── rank.py                 # Unchanged (backend for search_similar tool)
├── summarize.py            # Unchanged (available for agent or Stage 1 fallback)
├── context.py              # NEW — PersonalizationSource protocol + context merging
├── interests.py            # Changed — implements PersonalizationSource
├── notion.py               # NEW — implements PersonalizationSource (Notion API)
├── agent.py                # NEW — LLM editor: plan sections, call tools, assemble
├── digest.py               # Changed — section-based Digest model, two pipeline paths
├── pipeline.py             # NEW — top-level orchestrator
├── bot.py                  # Renamed from telegram.py — bot command handlers only
├── tools/                  # NEW — agent retrieval tools
│   ├── __init__.py         # Tool protocol, ToolRegistry, get_tool_definitions()
│   ├── base.py             # Tool ABC, ToolResult dataclass
│   ├── local.py            # SearchSimilar, SearchRecent, SearchByTopic, SearchBySource
│   └── arxiv.py            # SearchArxiv (ingests results into DB on retrieval)
└── output/                 # NEW — pluggable delivery layer
    ├── __init__.py         # Output protocol
    ├── telegram.py         # Telegram formatting + delivery
    ├── terminal.py         # Pretty-printed terminal output
    └── feed.py             # XML/Atom feed generation

scripts/
├── poll_feeds.py           # Unchanged
├── send_digest.py          # Uses pipeline.DigestPipeline
├── seed_feeds.py           # Unchanged
├── run_bot.py              # Runs bot.py
└── test_notion.py          # Manual test: fetch Notion context and print

config/
├── config.yaml             # Extended with models, notion, output sections
└── interests.yaml          # Unchanged
```

### What changed from Stage 1

| Module | Status | What changed |
|--------|--------|--------------|
| `config.py` | ✅ Changed | `NotionConfig`, `AgentConfig` dataclasses, `digest.mode` field, env vars |
| `db.py` | ✅ Changed | `ContextSnapshot` table for caching personalization context |
| `llm.py` | ✅ **New** | Provider-agnostic LLM client — `complete()` + `complete_with_tools()` with `ToolCall`/`LLMResponse` types |
| `embed.py` | Changed | Provider routing via `llm.py` (same `embed_text`/`embed_batch` interface) |
| `ingest.py` | Unchanged | RSS/Atom polling, content extraction, manual URL add |
| `rank.py` | Unchanged | Cosine similarity + selection (used by `tools/local.py` as backend) |
| `summarize.py` | Unchanged | Per-item summaries (used by Stage 1 fallback path) |
| `context.py` | ✅ **New** | `PersonalizationSource` protocol, `Context` dataclass, `merge_sources()` |
| `interests.py` | ✅ Changed | `InterestsSource` class implements `PersonalizationSource`; `load_interest_vectors()` preserved |
| `notion.py` | ✅ **New** | `NotionSource` implements `PersonalizationSource` — pulls from 5 Notion DBs, extracts text blocks (including synced blocks), summarizes via LLM, caches to DB |
| `agent.py` | ✅ **New** | LLM editor agent: receives context, calls tools via `submit_digest`, returns structured `Digest` |
| `digest.py` | ✅ Changed | `SectionType` enum, `DigestSection` model; `generate_digest` dispatches to agent or Stage 1 path |
| `pipeline.py` | ✅ **New** | `DigestPipeline` orchestrator — wires sources, tools, agent, and outputs |
| `bot.py` | ✅ Renamed | Was `telegram.py` — bot command handlers, uses pipeline for digest generation |
| `tools/` | ✅ **New** | `ToolRegistry`, `Tool` ABC, local retrieval tools, Arxiv skeleton |
| `output/` | ✅ **New** | `Output` protocol, `TelegramOutput`, `TerminalOutput`, feed skeleton |

### Key abstractions

Three protocols define the extension points. The pipeline orchestrator depends on these interfaces, not on concrete implementations.

**`PersonalizationSource`** (defined in `context.py`) — anything that provides context about current intellectual activity. Each source produces a prose context string for the agent and optionally a set of interest vectors for retrieval tools.

```python
class PersonalizationSource(Protocol):
    def get_context(self, config: Config) -> str: ...
    def get_interest_vectors(self, config: Config) -> dict[str, np.ndarray] | None: ...
```

Implementations: `interests.py` (static YAML descriptions → both context and vectors), `notion.py` (live Notion content → context string, vectors optional). The pipeline merges all sources: contexts are concatenated, vector dicts are merged. If a source is unavailable (e.g. Notion API down), the pipeline degrades gracefully — fewer sources, not a failure.

**`Output`** (defined in `output/__init__.py`) — anything that can deliver a formatted digest. Each output owns its formatting — Telegram deals with MarkdownV2 escaping and message splitting, terminal deals with width, feed deals with XML.

```python
class Output(Protocol):
    def send(self, digest: Digest, config: Config) -> None: ...
```

Implementations: `output/telegram.py`, `output/terminal.py`, `output/feed.py`. The pipeline dispatches to all configured outputs.

**`Tool`** (defined in `tools/base.py`) — a retrieval action the agent can invoke. Each tool has a name, description, and parameter schema (for the LLM tool use API) plus an `execute` method.

```python
class Tool(ABC):
    name: str
    description: str
    input_schema: dict

    @abstractmethod
    def execute(self, **params) -> ToolResult: ...
```

`ToolRegistry` collects all tools and produces the tool definitions list for the LLM API. The agent never imports tools directly — it goes through the registry. Adding a tool means adding a file and registering it.

### Module responsibilities (new and changed modules)

**`llm.py`** ✅ — Provider-agnostic LLM client. Exposes `complete(model, *, system, user_message, max_tokens)` for simple completions and `complete_with_tools(model, *, system, messages, tools, max_tokens)` for tool-use workflows. Both accept `"provider/model-name"` strings (e.g. `"anthropic/claude-sonnet-4-20250514"`, `"openai/gpt-4o-mini"`). Tool use returns a common `LLMResponse` dataclass containing `text`, `tool_calls: list[ToolCall]`, and `stop_reason`. Helper functions `build_tool_result_message()` and `build_assistant_message_from_response()` construct the message dicts for multi-turn tool-use conversations. Anthropic and OpenAI tool use are implemented; Google can be added when needed. Uses lazy singleton clients per provider.

**`context.py`** ✅ — Defines the `PersonalizationSource` protocol (`get_context()`, `get_interest_vectors()`) and a `Context` dataclass (prose summary + optional interest vectors). `merge_sources(sources, config)` concatenates context strings and merges vector dicts from all available sources, gracefully skipping any source that throws.

**`interests.py`** ✅ (changed) — `InterestsSource` class implements `PersonalizationSource`. `get_context()` returns topic descriptions as prose. `get_interest_vectors()` returns the embedded vectors via the existing `load_interest_vectors()` function, which is preserved for backward compatibility.

**`notion.py`** ✅ — `NotionSource` class implements `PersonalizationSource`. Queries 5 Notion databases (Journal, Work Diary, Notes, Library highlights, Reviews) filtered by `last_edited_time`. Extracts text from all block types (paragraphs, headings, lists, toggles, checkboxes, quotes, callouts, code, equations, bookmarks, table rows) including synced blocks (transparently resolves both original and reference synced blocks). Sends all extracted content in a single LLM call (via `llm.complete()`) with a summarization prompt targeting ~5k tokens of output. Fallback logic: if fewer than `min_entries_threshold` entries found in the configured lookback window, expands to `fallback_lookback_days`; if still below threshold, returns empty string. Caches generated summaries to a `ContextSnapshot` table via the DB with a configurable TTL (`cache_ttl_hours`, default 24 hours). By default, uses cached context if available and fresh (age < TTL). Accepts a `force_refresh` parameter to bypass cache and fetch fresh data. On LLM failure with stale cache available, falls back to stale cache rather than failing completely.

**`agent.py`** ✅ — The LLM editor agent. `plan_and_assemble(config, context, tool_registry)` runs the agent loop: sends the personalization context and an editorial system prompt, lets the agent call retrieval tools via `llm.complete_with_tools()`, and collects structured output via a `submit_digest` tool that the agent calls to deliver the final `Digest` with typed sections. The agent decides which sections to include, how many items each gets, and writes summaries as part of assembly. The loop is capped at `max_iterations` (configurable) to prevent runaway API calls.

**`digest.py`** ✅ (changed) — The `Digest` data model gains sections via `SectionType` enum and `DigestSection` dataclass. `DigestItem` is extended with explicit `item_id`, `title`, `url`, `source`, `author` fields for agent-produced items (Stage 1 items still use `scored_item`). `generate_digest()` checks `config.digest.mode` and dispatches to `generate_digest_deterministic()` (Stage 1) or the `DigestPipeline` (Stage 2). Formatting logic moves out to the output layer.

**`pipeline.py`** ✅ — The top-level orchestrator. `DigestPipeline` is initialized with a config, DB, list of `PersonalizationSource`s, and list of `Output`s. `run()` merges context from all sources, generates a digest (agent or deterministic), saves to DB, and dispatches to all outputs. Falls back to deterministic if agent produces an empty digest or no personalization context is available. Scripts and the bot instantiate a pipeline and call `run()`.

**`bot.py`** (renamed from `telegram.py`) — Telegram bot command handlers (`/add`, `/digest`, `/status`). The `/digest` command instantiates a pipeline and calls `run()`. Message delivery is handled by `output/telegram.py`.

**`tools/local.py`** — Four retrieval tools backed by the local DB: `SearchSimilar` (embed query → cosine similarity via `rank.py`), `SearchRecent` (items from last N days), `SearchByTopic` (items matching a topic cluster using interest vectors), `SearchBySource` (filter by feed/source type). All return item metadata the agent uses for editorial decisions.

**`tools/arxiv.py`** — `SearchArxiv` tool. Queries the Arxiv API, returns results to the agent, and ingests them into the DB (with source tagging) so they're available for future retrieval. Additional external tools (OpenAlex, citation search) follow the same pattern — one file per API.

**`output/telegram.py`** — Implements `Output`. Formats a `Digest` with typed sections into MarkdownV2 messages (respecting Telegram's 4096-char limit and per-section formatting), sends via the Telegram API.

**`output/terminal.py`** — Implements `Output`. Pretty-prints a digest to stdout.

**`output/feed.py`** — Implements `Output`. Serializes a digest as an XML/Atom feed.

### Data flow

```
┌──────────────────────────────────────────────────────────────┐
│                         INGESTION (unchanged)                 │
│                                                               │
│  Cron (every 2h): scripts/poll_feeds.py                       │
│    ingest.poll_feeds() → feedparser → trafilatura → embed     │
│    │                                                          │
│    ▼                                                          │
│  SQLite (items with embeddings)                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    DIGEST GENERATION (daily)                   │
│                                                               │
│  scripts/send_digest.py → pipeline.DigestPipeline.run()       │
│                                                               │
│  1. PERSONALIZATION                                           │
│     context.merge_sources([interests, notion])                │
│     ├── interests.get_context()    → topic descriptions       │
│     ├── interests.get_interest_vectors() → embeddings         │
│     └── notion.get_context()       → recent activity summary  │
│     Result: Context(prose, vectors)                           │
│                                                               │
│  2. AGENT PLANNING + RETRIEVAL                                │
│     agent.plan_and_assemble(context, tool_registry)           │
│     │                                                         │
│     │  Agent calls tools via LLM tool use:                    │
│     │  ├── SearchSimilar   → embed + cosine (rank.py)         │
│     │  ├── SearchRecent    → DB query                         │
│     │  ├── SearchByTopic   → DB + interest vectors            │
│     │  ├── SearchBySource  → DB query                         │
│     │  └── SearchArxiv     → Arxiv API (+ ingest into DB)     │
│     │                                                         │
│     │  Agent decides sections, selects items, writes summaries│
│     ▼                                                         │
│     Digest(sections=[DigestSection(...), ...])                │
│                                                               │
│  3. DELIVERY                                                  │
│     for output in outputs:                                    │
│         output.send(digest)                                   │
│     ├── output/telegram.py  → Telegram message                │
│     ├── output/terminal.py  → stdout                          │
│     └── output/feed.py      → XML/Atom file                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    BOT (long-running)                          │
│                                                               │
│  scripts/run_bot.py → bot.run_bot()                           │
│    /add <url> → ingest.ingest_url()                           │
│    /digest    → pipeline.DigestPipeline.run()                 │
│    /status    → db queries                                    │
└──────────────────────────────────────────────────────────────┘
```

### Dependency graph

No cycles. New modules in **bold**.

```
config           ← (no internal deps)
db               ← (no internal deps)
llm              ← config
embed            ← config, llm
rank             ← config, db
interests        ← config, embed, context
notion           ← config, llm, context
summarize        ← config, llm
context          ← (no internal deps — defines protocol + dataclass)
tools/*          ← config, db, embed, rank
agent            ← config, llm, tools
digest           ← config, db, interests, rank, summarize, agent
pipeline         ← config, db, context, digest, agent, tools, output
bot              ← config, db, ingest, pipeline
output/*         ← config, digest
ingest           ← config, db, embed
```

### Config structure

**`config/config.yaml`** (Stage 2 additions):
```yaml
digest:
  size: 10
  max_per_topic: 3
  schedule: "08:00"
  timezone: "Europe/Madrid"
  mode: "agent"                   # "agent" (Stage 2) or "deterministic" (Stage 1)

polling:
  interval_hours: 2

embedding:
  model: "text-embedding-3-small"

summarization:
  model: "claude-haiku-4-5-20251001"

agent:
  model: "anthropic/claude-sonnet-4-20250514"
  max_iterations: 10
  max_tokens: 4096

notion:
  database_ids:
    journal: "..."
    work_diary: "..."
    library: "..."
  lookback_days: 14
  fallback_lookback_days: 30
  min_entries_threshold: 3
  max_chars_per_entry: 3000
  summary_model: "google/gemini-2.5-flash-lite"
  cache_ttl_hours: 24

outputs:
  - type: telegram
    chat_id: "..."
  - type: terminal
  - type: feed
    path: "output/digest.xml"
```

**Environment variables (`.env`):**
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
TELEGRAM_BOT_TOKEN=...
NOTION_TOKEN=secret_...
```

### Design decisions

- **LLM as editor, not search engine.** The agent decides what the digest should look like and uses retrieval tools to find items. It never sees all items at once. This contains confabulation risk to the editorial layer while getting the flexibility of agentic planning.
- **Newspaper sections, not a ranked list.** Different content types deserve different treatment. Papers get one-line roundups. Long-form gets full summaries. News gets headlines. The agent decides section structure based on what's available and what's relevant.
- **Three extension points, not a framework.** `PersonalizationSource`, `Output`, and `Tool` are the only abstractions. Everything else is concrete. This avoids over-engineering while making the system pluggable where it matters — adding a new context source, output format, or retrieval tool doesn't require touching the pipeline.
- **Provider-agnostic model config.** `llm.py` routes `"provider/model"` strings so any part of the pipeline can use any provider's model. Switching the summarizer from Claude to GPT-4o-mini is a one-line config change, not a code change.
- **Stage 1 as fallback.** The deterministic pipeline is preserved. `digest.mode: "deterministic"` in config bypasses the agent entirely and runs the original rank → select → summarize path. Useful for comparison and as a safety net.
- **`telegram.py` splits into `bot.py` + `output/telegram.py`.** Bot commands are an input interface (control plane). Digest delivery is an output concern. Separating them lets the pipeline dispatch to Telegram without importing bot machinery.
- **Tools as a package.** Each external API (Arxiv, later OpenAlex, citations) has different auth, parsing, and error handling. One file per API, a shared base class, and a registry that produces tool definitions for the LLM API. Adding a tool = adding a file.
- **Notion as optional.** If Notion is unavailable, the pipeline degrades to static interests — fewer personalization sources, not a failure. The `PersonalizationSource` protocol makes this natural: the pipeline merges whatever sources are available.
- **Provider-agnostic tool use, not a framework.** `llm.py` exposes a common `complete_with_tools()` interface with `ToolCall` and `LLMResponse` types. The agent works with these abstractions — swapping the agent model from Claude to Gemini or GPT is a config change. LangChain/LangGraph would add abstraction without value for a single-purpose agent with a fixed set of tools.
- **Notion context caching.** Fetching and summarizing Notion content is expensive (multiple API calls + LLM summarization). The system caches Notion context summaries in the DB with a configurable TTL (default 24 hours via `cache_ttl_hours`). By default, the digest uses cached context if available and fresh. The cache can be bypassed with `--force-notion-refresh` flag in `scripts/send_digest.py`. On LLM failure, the system falls back to stale cache if available rather than failing completely. This improves reliability and reduces API costs during development and testing.

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
