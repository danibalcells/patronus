# Patronus: Codebase Complexity Review

Framework: John Ousterhout's *A Philosophy of Software Design*. The three complexity types tracked throughout are **change amplification** (a small conceptual change touches many files), **cognitive load** (understanding a change requires holding too much context), and **unknown unknowns** (it's not obvious what you need to know to change something safely).

---

## 0. Agent pipeline: code flow and concept map

This section traces exactly what happens when you run `scripts/send_digest.py` (or `tests/test_agent.py`) in agent mode, and disambiguates the several distinct "summarize" and "select" operations that occur.

### Entry points

There are two main ways to run the agent pipeline. Both converge on `DigestPipeline.run()` — the difference is purely in which outputs are wired up, and where the DB files live.

**`scripts/test_agent_manual.py`** — local dev runner

```
python scripts/test_agent_manual.py [--force-notion-refresh] [--reader] [--no-regen]

  load_config()                          # reads local config/config.yaml + .env
  Database()                             # local db.sqlite3 in project root
  DigestPipeline(config, db, outputs=[TerminalOutput()])
  pipeline.run()                         # full agent pipeline, prints to terminal

  --no-regen flag: skips the pipeline entirely, loads the last saved digest
                   from the DB and re-renders it. Useful for testing output
                   formatting without paying for LLM calls.
  --reader flag:   adds ReaderOutput() to the outputs list.
  --max-iterations: overrides AgentConfig.max_iterations at runtime.
```

**No Telegram here.** This script is designed for local iteration — terminal only.

**`modal_app.py`** — production deployment

```
modal run modal_app.py                   # triggers main() locally, runs send_digest.remote()
modal run modal_app.py --job digest      # same
modal run modal_app.py --job poll        # trigger feed poll
modal run modal_app.py --job sync        # trigger Notion mirror sync

send_digest() runs on Modal at 07:30 ET daily:
  load_config()
  config.notion.mirror_path = "/data/notion_mirror.sqlite3"   ← patched to volume
  Database(db_path="/data/db.sqlite3")                         ← patched to volume
  DigestPipeline(config, db, outputs=[TerminalOutput(), FeedOutput(), ReaderOutput()])
  pipeline.run()
  volume.commit()                        # persists volume changes
```

**No Telegram in production.** The prod outputs are terminal (for Modal logs), RSS feed (uploaded to Cloudflare R2), and Readwise Reader. Telegram delivery, if used at all, would be added here.

The Modal Volume at `/data/` is shared across all three scheduled functions (`send_digest`, `poll_feeds`, `sync_notion`) and is the source of truth for `db.sqlite3` and `notion_mirror.sqlite3`. Each function calls `volume.commit()` after writing to persist changes.

**`scripts/send_digest.py`** is a local alternative that adds Telegram as an output and uses the local DB. It's structurally identical to the Modal function.

### Entry point → pipeline

```
DigestPipeline(config, db, outputs=[...])
pipeline.run()
  → pipeline._generate_agent()
```

### Phase 1: Build reader context (`context.py` + `notion.py` + `notion_mirror.py`)

```
merge_sources([NotionSource(db)])        # context.py
  → NotionSource.get_context(config)    # notion.py
      Check DB for cached ContextSnapshot (< 24h old)
      If cache hit:  return cached prose string
      If cache miss:
        Read pages from local SQLite mirror (notion_mirror.py)
        — journal entries, work diary, library highlights —
        Pack into one large text block
        LLM call #1: complete()          # llm.py
          Model: notion_context_model
          Input:  all Notion page text (~200k chars max)
          Output: a prose "reader context" document
          ("The reader is working on belief geometry in LLMs.
            Active questions: does context shift probing axes?
            Peripheral interests: consciousness, phenomenology...")
        Save result to ContextSnapshot in DB
        Return prose string
```

The reader context is a single prose document about what the person is doing and thinking about *right now*. Everything downstream receives this string and uses it to make editorial decisions.

### Phase 2: Build inventory (`agent/_inventory.py`) — no LLM

```
build_inventory(config, db)
  db.get_items_since(cutoff)             # all items since last digest (2-day default)
  db.get_recently_digested_item_ids()    # items seen in last 7 days → flagged
  Group by source, assign short numeric IDs (1, 2, 3...)
  Format as structured text:
    ## Arxiv (12 items)
      ID: 1
      Title: "Probing Belief Geometry in LLMs"
      URL: https://arxiv.org/...
      Snippet: "We investigate..."
      Flag: PREVIOUSLY_FEATURED        ← if already digested
    ...
  Returns: (main_inventory_str, tweet_inventory_str, short_id_map)
```

The short IDs exist solely so the LLM can reference items by number rather than URL. They're remapped back to real DB IDs at the end of the pipeline.

**Nothing from `rank.py` or `embed.py` is used here.** The inventory is all items, presented in full, not ranked.

### Phase 3: Agent steps (`agent/_steps.py`, `agent/run.py`)

All five steps run sequentially. Each receives the outputs of the previous ones as plain text strings.

#### Step 2 — `identify_angles()`

```
LLM call #2: complete()
  Model: angles_model
  Input:  reader context + full inventory text
  Output: plain text listing 5–10 editorial angles

Example output:
  "- [PRIMARY] Two papers in today's feed address orthogonal
     subspaces in residual streams, directly relevant to the
     reader's belief geometry work...
   - [SECONDARY] The Anthropic/Pentagon story is over-represented
     (7 items). Select the policy analysis piece only..."
```

`identify_angles` is **not** summarizing anything. It's identifying *editorial directions* — which themes in today's content connect to what the reader cares about, which stories are over-saturated, which peripheral hooks are worth following. The output is prose for the downstream steps to read; it is never shown to the reader.

#### Step 3a — `filter_news()`

```
LLM call #3: complete_structured() → NewsFilterResult
  Model: news_model
  Input:  reader context + angles + full inventory
  Output: { items: [{ item_id: "7", cross_ref: "none" }, ...] }
          — just IDs, no summaries written here
```

This is pure **editorial selection by the LLM**. The model reads the inventory text and, guided by the angles, picks which items belong in the news section. No embeddings. No cosine similarity. The LLM is acting as an editor who reads everything and picks the most relevant items.

The selected items are then formatted into a text block (title, URL, snippet) that feeds into the compose step.

#### Step 3a' — `summarize_chatter()`

```
LLM call #4: complete()
  Model: chatter_model
  Input:  reader context + tweet inventory (tweets only)
  Output: grouped conversation clusters as text
          ("TOPIC: AI safety debate\n
            SUMMARY: [@handle](url) argues that...\n
            ITEM_IDS: 23, 24, 31")
```

Tweets are separated from articles in the inventory. This step processes only the tweet section and groups discussions by topic.

#### Step 3b — `scout_research()` — tool loop

```
Tool loop: up to 2 iterations
  Each iteration:
    LLM call #N: complete_with_tools()
      Model: research_model
      Input:  reader context + angles + tool results so far
      Tools available: search_arxiv, search_openalex, search_notion,
                       search_similar, search_recent, search_by_topic

    LLM issues 3–5 parallel tool calls, e.g.:
      search_arxiv(query="belief geometry transformer residual stream")
      search_openalex(query="linear probing directions truth LLMs")
      search_notion(query="probing methods")

    Each tool call:
      search_arxiv  → fetches from arxiv API, ingests results into DB, returns items
      search_similar → embeds the query (embed.py), cosine-searches stored embeddings
      search_notion  → FTS5 query on the local mirror (notion_mirror.py)

  Model receives tool results, then either stops or runs one more iteration.
  Final output: curated paper list as prose text, never shown directly to reader.
```

This is where `embed.py` and `rank.py`-style retrieval happen — but on-demand during tool calls, not as an upfront ranking pass. `search_similar` embeds the query string at call time and ranks items by cosine similarity to that query. This is different from the deterministic path's upfront ranking of all items against static interest vectors.

#### Step 3c — `pull_threads()` — tool loop

```
Tool loop: up to 3 iterations (same structure as scout_research)
  Input additionally includes news and research outputs (to avoid duplication)
  All tools available
  Output: 3–8 content proposals as prose text
```

#### Step 4 — `compose_digest()` (`agent/_compose.py`)

```
LLM call #M: complete_with_tools() — forced to call submit_digest
  Model: compose_model
  Input:  reader context + angles + news text + chatter text +
          research text + threads text
  Tools:  [submit_digest only]

  Model writes final reader-facing summaries and calls:
    submit_digest({
      sections: [
        { type: "long_form_pick", title: "...", items: [...] },
        { type: "whats_new",      title: "...", items: [...] },
        { type: "research_roundup", ... },
        ...
      ]
    })

_parse_submit_digest() → Digest dataclass
  Short IDs (e.g. "7") → real DB IDs (e.g. "abc123")
```

The compose step is the **only place where reader-facing summaries are written** in the agent path. No per-item summarization happens before this step. All the upstream steps pass text to each other; compose sees everything and writes the final copy.

### Phase 4: Save + dispatch outputs

```
pipeline._save_digest(digest)
  format_telegram(digest)      ← generates Telegram MarkdownV2 (stored in DB as formatted_text,
                                  but never read back — see R3 in refactoring plan)
  db.save_digest(...)          ← DigestRecord + DigestItemRecord rows

for output in self._outputs:
  output.send(digest, config)  ← each output independently formats the same Digest object
```

Each output receives the same `Digest` object and formats it independently. There is no shared rendering step. The `Digest` is a structured data object (sections → items with title, url, summary, published\_date); each formatter decides what to show and how to show it.

| Output | Format | Extra LLM call? | Delivery |
|---|---|---|---|
| `TerminalOutput` | Plain text with ASCII/emoji | No | `print()` |
| `TelegramOutput` | MarkdownV2, split at 4096 chars | No | Telegram Bot API |
| `FeedOutput` | HTML (`format_digest_html`) wrapped in RSS/Atom XML | No | Upload to Cloudflare R2 |
| `ReaderOutput` | HTML (`format_digest_html`) + title/tagline | Yes (`summarize_digest`) | POST to Readwise Reader API |

**`format_digest_html`** (in `output/feed.py`) is shared by both `FeedOutput` and `ReaderOutput` — they call the same function to produce the HTML body. `ReaderOutput` additionally calls `summarize_digest()` to generate a short title like `"ML scaling limits, Hofstadter on loops"` and a tagline for display in the Readwise inbox.

The Telegram and terminal formatters are completely separate from the HTML formatters and share no code.

### What `summarize_item` is and when it runs

`summarize_item` (in `summarize.py`) is a **deterministic-path-only function**. It takes one article's title and full text, calls the LLM, and returns a 2-sentence summary for display. It is called once per selected item in `generate_digest_deterministic()`.

**In the agent path, `summarize_item` is never called.** Summaries are written by the compose step (Step 4), which receives the full item text via the inventory and tool results and writes all summaries in one LLM call.

### Modules active vs inactive in the agent path

| Module | Agent path | Deterministic path | Notes |
|---|---|---|---|
| `config.py` | ✓ | ✓ | |
| `db.py` | ✓ | ✓ | |
| `llm.py` | ✓ | ✓ (via summarize.py) | |
| `embed.py` | ✓ (via tools) | ✓ (at ingest; at ranking) | Agent: only on-demand via tool calls |
| `observability.py` | ✓ | ✓ | |
| `context.py` | ✓ | ✗ | |
| `notion.py` | ✓ | ✗ | |
| `notion_mirror.py` | ✓ | ✗ | |
| `interests.py` | ✗ | ✓ (fallback only) | Used if Notion unavailable |
| `rank.py` | ✗ | ✓ | No upfront ranking in agent path |
| `summarize.py` `summarize_item()` | ✗ | ✓ | Agent compose step writes summaries instead |
| `summarize.py` `summarize_digest()` | ✗ | ✗ | Called by `ReaderOutput` only, regardless of path |
| `agent/_inventory.py` | ✓ | ✗ | |
| `agent/_steps.py` | ✓ | ✗ | |
| `agent/_compose.py` | ✓ | ✗ | |
| `agent/_prompts.py` | ✓ | ✗ | |
| `agent/run.py` | ✓ | ✗ | |
| `tools/*` | ✓ | ✗ | |
| `digest.py` `generate_digest_deterministic()` | ✗ (unless fallback) | ✓ | |
| `pipeline.py` | ✓ | ✓ | Orchestrates both paths |
| `output/*` | ✓ | ✓ | Same formatters for both paths |

---

## 1. Structural overview

### Configuration and bootstrapping (`config.py`, `config/config.yaml`, `config/interests.yaml`)

**Responsibility:** Typed dataclass tree loaded from YAML and environment variables.
**Public interface:** `load_config()` → `Config`. Callers receive a fully constructed object.
**Depth: Deep.** One call hides YAML parsing, env loading, nested object construction, and interests file loading. Callers are completely insulated from the file format.

### Data persistence (`db.py`)

**Responsibility:** All SQLite interactions — items, feeds, digest records, context snapshots.
**Public interface:** `Database` class with named methods. No SQL leaks to callers.
**Depth: Deep.** 438 lines, clean API surface. One-shot schema migration baked into `__init__`. Callers never write SQL.

### LLM abstraction (`llm.py`)

**Responsibility:** Provider-agnostic completions — `complete`, `complete_with_tools`, `complete_structured` — plus message format conversion and utility helpers.
**Public interface:** Three top-level functions + two message-builder helpers + `LLMResponse`/`ToolCall` dataclasses.
**Depth: Deep — with one significant leak.** 449 lines of provider routing and format translation hidden behind simple signatures. The exception: `build_tool_result_message` and `build_assistant_message_from_response` return Anthropic-shaped message dicts. Callers must construct `tool_result`/`tool_use` blocks. This is an information leak that will cost you if you try to run the full agent loop on a non-Anthropic provider.

### Embeddings (`embed.py`)

**Responsibility:** Batch and single text embedding.
**Public interface:** `embed_text(text, model)`, `embed_batch(texts, model)`.
**Depth: Deep for its size** (59 lines). Clean. However, it's hard-wired to OpenAI via a direct `openai` import, unlike `llm.py` which is provider-agnostic. This creates an unknown unknown: the embedding model is configurable in `config.yaml`, but the provider is not — you cannot switch embeddings to a non-OpenAI provider without editing `embed.py`.

### Feed ingestion (`ingest.py`)

**Responsibility:** Poll RSS feeds, extract full text via trafilatura, embed, dedup, store.
**Public interface:** `poll_feeds(config, db)` is the main entry point.
**Depth: Deep.** 490 lines — tweet HTML parsing, link extraction, concurrent fetch, embedding — all internal. Callers see one function.

### Personalization context (`context.py`, `interests.py`, `notion.py`, `notion_mirror.py`)

**Responsibility:** Build the reader context string that guides the agent.
- `context.py` (56 lines): `PersonalizationSource` protocol + `Context` dataclass + `merge_sources()`.
- `interests.py` (29 lines): Static YAML interest vectors via embeddings.
- `notion.py` (681 lines): Live Notion API fetching, LLM summarization, 24h caching, rate-limit retry.
- `notion_mirror.py`: Local SQLite FTS5 mirror of Notion; synced nightly.

**Public interface:** `PersonalizationSource` protocol + `merge_sources()`. Callers only see the protocol.
**Depth: Mixed.** `notion.py` is deep — callers see `get_context()` while it handles retries, API pagination, cache invalidation, and summarization. `context.py` is thin by design; it's mostly dispatch code. One coupling problem: `merge_sources()` does `isinstance(source, NotionSource)` internally to pass `force_refresh` — the function that's supposed to treat all sources uniformly is special-casing one of them.

### Ranking (`rank.py`)

**Responsibility:** Cosine similarity scoring, time decay, topic-capped digest selection.
**Public interface:** `rank_unread(items, interest_vectors)` → `list[ScoredItem]`; `select_digest(scored, size)`.
**Depth: Deep for its size** (108 lines). Pure functions, no side effects, no dependencies except `db.Item` and `numpy`. The best-designed module in the codebase.

### Agent pipeline (`agent/`)

**Responsibility:** LLM-driven digest generation.
- `_inventory.py`: Build a structured text inventory from recent DB items — no LLM.
- `_prompts.py`: All system prompts + `submit_digest` tool schema.
- `_steps.py`: Individual pipeline steps — angles, news filter, chatter, research, threads.
- `_compose.py`: Final compose step with retry logic.
- `run.py`: Orchestrator calling steps in sequence with observability spans.

**Public interface:** `plan_and_assemble(config, context, registry, db)` → `Digest`.
**Depth: Mixed.** `run.py` is appropriately thin orchestration. `_steps.py` hides the tool-loop machinery behind five clean function signatures. `_prompts.py` is a pure data file; putting all prompts here is good practice (change the editorial behavior in one place).

### Tools (`tools/`)

**Responsibility:** Retrieval actions the agent can call.
**Public interface:** `Tool` ABC (name, description, input\_schema, execute) + `ToolRegistry`.
**Depth: Good.** The ABC pattern is clean, each tool is self-contained, and `ToolRegistry` is minimal. The only structural oddity: registration is done via module-level functions (`register_local_tools`, `register_arxiv_tools`, etc.) called from `pipeline.py`. This means adding a tool requires editing `pipeline.py`, not just adding a file.

### Digest model (`digest.py`)

**Responsibility:** `Digest`, `DigestSection`, `DigestItem`, `SectionType` models + the deterministic generation pipeline.
**Public interface:** The models + `generate_digest_deterministic()` + `generate_digest()`.
**Depth: Shallow-ish.** The file mixes model definitions with business logic. `generate_digest()` uses a lazy import of `pipeline.py` to avoid a circular dependency — a signal that the module boundaries are not quite right. `DigestItem` carries `scored_item: ScoredItem | None`, which bleeds the internal ranking type into the output model.

### Orchestration (`pipeline.py`)

**Responsibility:** Wire together personalization sources, tool registry, agent, and outputs.
**Public interface:** `DigestPipeline(config, db, sources, outputs).run()`.
**Depth: Thin orchestration** (132 lines). Appropriate — this is the composition root. One hidden coupling: `_save_digest()` directly imports `format_telegram` to produce the stored DB record. Telegram formatting is now in the storage path.

### Outputs (`output/`)

**Responsibility:** Deliver formatted digests — Telegram, terminal, XML/Atom feed, Readwise.
**Public interface:** `Output` protocol — `send(digest, config)`.
**Depth: Medium.** Telegram has complex MarkdownV2 escaping logic. The `Output` protocol itself is minimal and clean.

### Observability (`observability.py`)

**Responsibility:** Langfuse tracing spans with graceful no-op fallback.
**Public interface:** Context managers: `pipeline_run`, `agent_run`, `iteration_span`, `llm_generation`, `tool_call`.
**Depth: Deep for its size** (137 lines). Completely transparent to callers — no Langfuse dependency if keys aren't set.

---

## 2. Dependency and coupling analysis

### Hub files

These files are imported by many others. Changes here amplify.

| File | Role | Risk |
|---|---|---|
| `config.py` | Imported by essentially everything | Every new feature adds a field here + `config.yaml` |
| `db.py` | Imported by ingest, rank, tools, agent, pipeline, digest | Schema additions cascade |
| `llm.py` | Imported by summarize, notion, agent steps/compose | Provider format leak makes provider switches harder |
| `digest.py` | Imported by pipeline, output/\*, agent/\_compose, agent/run, bot | Mixes models with logic; changes to either affect all importers |

### Unnecessary couplings

**1. `pipeline.py` → `output/telegram.py` for storage.** `_save_digest()` calls `format_telegram()` to produce the `formatted_text` stored in `DigestRecord`. This string is never read back by the application — it's write-only. The Telegram formatter is now in the storage path for no benefit.

**2. `context.py` isinstance-checks `NotionSource`.** `merge_sources()` inspects the concrete type of a source to route the `force_refresh` kwarg. The protocol abstraction is undermined. Any new "refreshable" source would require another isinstance branch.

**3. `DigestItem.scored_item: ScoredItem | None`.** The ranking type from `rank.py` lives on the output model. Output formatters and storage code have to understand `ScoredItem` to extract `score` and `matched_topic`. The deterministic path needs it; the agent path sets it to `None`.

**4. `summarize.py` bypasses `llm.py`.** `summarize_item` creates its own `anthropic.Anthropic` client. It does not use the provider-agnostic `llm.complete()`. If you set `digest_summary_model` in config to a Google or OpenAI model, the model name is passed to the Anthropic client and the call fails. (The newer `summarize_digest` function correctly uses `llm.complete_structured`; `summarize_item` was not migrated.)

---

## 3. Git history analysis

### Most frequently changed files (last 3 months)

```
15  patronus/agent.py  (now split into agent/)
13  config/config.yaml
10  patronus/config.py
 9  patronus/db.py
 9  patronus/summarize.py
 7  patronus/pipeline.py
 6  patronus/digest.py
 6  patronus/output/reader.py
 5  patronus/notion.py
 5  patronus/llm.py
```

### Hotspots (large AND frequently changed)

- **`config.py` (139 lines, 10 changes) + `config.yaml` (13 changes):** The main change amplifier. Nearly every new feature touches this pair. The files are small, but the coupling cost is paid every time.
- **`db.py` (438 lines, 9 changes):** More active than expected. Schema additions + one migration. Reasonably managed but worth watching.
- **`agent/` (was `agent.py`, 15 changes):** The refactor into a package was the right call. The activity has been distributed and the module is now readable.

### Temporal coupling

- `config.py` ↔ `config.yaml`: Always change together (expected, correct).
- `pipeline.py` ↔ `output/telegram.py` + `output/terminal.py`: Output changes require touching the pipeline, confirming the registration pattern creates coupling.
- `digest.py` ↔ `output/*`: Digest model changes cascade to formatters.
- `llm.py` ↔ `summarize.py`: Changed together in one batch — reflecting the partial migration.

### Stable foundations (large files that rarely change)

- **`rank.py` (108 lines, ~0 changes):** Pure functions, no side effects. This is what a deep module looks like.
- **`embed.py` (59 lines, ~1 change):** Stable. The OpenAI coupling is a latent risk but hasn't been a problem.
- **`observability.py` (137 lines, ~0 changes):** The facade pattern is working.
- **`tools/base.py` (~0 changes):** The abstraction is right — nothing needs to change about it.
- **`notion.py` (681 lines, 5 changes):** Large and stable-ish. The complexity is warranted by what it does.

---

## 4. Change amplification scenarios

### Add a new output channel (e.g. email)

Files: `output/email.py` (new) → `config.py` → `config/config.yaml` → `scripts/send_digest.py`.
**4 files.** The `Output` protocol earns its keep here. Config boilerplate is the only overhead.

### Add a new agent pipeline step (e.g. "newsletter scout")

Files: `agent/_steps.py` → `agent/run.py` → `agent/_prompts.py` → `config.py` → `config/config.yaml` → possibly `agent/_compose.py`.
**5–6 files.** The per-step model override in `AgentConfig` is the main amplifier. If you didn't need per-step model config, it would be 3–4 files.

### Add a new retrieval tool

Files: `tools/new_tool.py` (new) → `pipeline.py` (add `register_new_tools` call).
**2 files.** This is the best-performing scenario. The `Tool` ABC + `ToolRegistry` pattern works exactly as intended.

### Add a field to `Item` (e.g. `language`)

Files: `db.py` (model + migration if needed) → `ingest.py` (populate) → `agent/_inventory.py` (display in inventory text) → possibly `output/telegram.py` or `output/terminal.py`.
**3–5 files.** Reasonable. The `Item` model is well-contained.

### Switch embedding provider from OpenAI to another

Files: `embed.py` (replace client, add provider routing) → `config.py` (update `EmbeddingConfig` to add provider) → `config/config.yaml`.
**3 files.** Looks manageable, but the change is harder than it appears: `embed.py` hard-imports `openai` and the switching logic doesn't exist. You're writing new routing code, not just changing a config value. The assumption that this would be as easy as switching an LLM model is an **unknown unknown**.

---

## 5. Complexity inventory

### Shallow modules

- **`context.py`** (56 lines): The Protocol and `merge_sources()` add minimal abstraction value given two implementations. The `isinstance` hack in `merge_sources` actually makes it *more* complex than just calling `source.get_context(config)`. Not a problem to fix on its own, but it points at the coupling issue.

### Information leakage

- **Anthropic message format in `llm.py`:** `build_tool_result_message` and `build_assistant_message_from_response` emit Anthropic-style dicts. Callers in `_steps.py` and `_compose.py` depend on this format. Running the agentic loop with an OpenAI model requires these to return OpenAI-compatible formats, but they don't.
- **`DigestItem.scored_item`:** Internal ranking state on the output model. The `pipeline.py` storage code accesses `item.scored_item.score` — meaning saving a digest requires understanding the ranking internals.
- **`context.py` isinstance:** `merge_sources` knows about `NotionSource` despite the Protocol being designed to abstract it away.

### Unnecessary abstraction

- None that rise to the level of a real problem. The codebase is lean.

### Duplicated concepts

- **`_now_utc()`** is defined identically in `db.py`, `digest.py`, and `notion_mirror.py`.
- **`_today_str()`** is computed in both `_steps.py` and `run.py`. `run.py` computes `today_str` but doesn't pass it down — each step recomputes independently.
- **Model defaults duplicated:** `summarize.py` has hardcoded `_DEFAULT_MODEL = "claude-sonnet-4-20250514"` and `_DIGEST_SUMMARY_MODEL = "google/gemini-3-flash-preview"`. `config.py` also has `AgentConfig.notion_context_model` and `AgentConfig.digest_summary_model` defaults. It's not obvious which one wins.

### Scattered configuration

- `summarize.py` hardcodes model defaults that shadow config values. If you update `digest_summary_model` in `config.yaml` but the wrong fallback triggers, nothing breaks visibly — the hardcoded model runs silently.

### Deep modules that work well

- **`db.py`**: 438 lines, clean API, no SQL leaking.
- **`llm.py`**: Three entry points hide all provider specifics. Minor message-format leak doesn't affect the happy path.
- **`ingest.py`**: Complex tweet parsing, link extraction, and threading are fully internal.
- **`notion.py`**: Rate limiting, retries, caching, summarization — none of it visible to callers.
- **`rank.py`**: Pure functions. The gold standard.
- **`observability.py`**: Langfuse dependency is completely transparent.

---

## 6. Refactoring plan

Ordered by impact on change amplification and unknown unknowns.

---

### R1: Fix `summarize.py` Anthropic coupling (cognitive load + unknown unknowns)

**Problem:** `summarize_item` creates its own `anthropic.Anthropic` client, bypassing `llm.py`. Setting `digest_summary_model` to a non-Anthropic model in config silently fails — the model string is passed directly to the Anthropic client.

**Change:** Replace the direct client in `summarize_item` with `llm.complete()`. Remove `_get_client()`, `_client`, and the top-level `import anthropic`. The `model` parameter already passes through, so this is mechanical.

**Files:** `patronus/summarize.py` only.

**Benefit:** Makes model switching actually work. Removes a hidden provider dependency. Reduces the file by ~15 lines.

---

### R2: Flatten `DigestItem.scored_item` into explicit fields (information leakage)

**Problem:** `DigestItem` carries `scored_item: ScoredItem | None`, a ranking-internal type, on the output model. `pipeline.py._save_digest()` accesses `item.scored_item.score` and `item.scored_item.matched_topic`. Callers that just want to display or store a digest must understand `ScoredItem`.

**Change:** Add `score: float = 0.0` and `matched_topic: str = ""` directly to `DigestItem`. Update `digest.py`'s deterministic path to set these fields directly when building `DigestItem`. Remove `scored_item`. Update `pipeline.py._save_digest()` to read the flat fields.

**Files:** `patronus/digest.py`, `patronus/pipeline.py`.

**Benefit:** Output formatters and storage code become independent of the ranking domain. The agent path (which never sets `scored_item`) no longer carries a null field. `rank.py` becomes a pure implementation detail of the deterministic path.

---

### R3: Remove `formatted_text` from `DigestRecord` or drop the Telegram coupling (unnecessary coupling)

**Problem:** `pipeline.py._save_digest()` calls `format_telegram()` to produce `formatted_text` for the DB. This string is written but never read back by the application. Telegram's formatting logic is in the storage path for no benefit.

**Change:** Remove the `format_telegram` call and the `formatted_text` field from `DigestRecord`. If you ever want to retrieve the formatted digest from the DB, store a format-agnostic representation (e.g. structured JSON via `digest.sections`) rather than Telegram-specific MarkdownV2.

**Files:** `patronus/pipeline.py`, `patronus/db.py`.

**Benefit:** `pipeline.py` no longer depends on `output/telegram.py`. Changing Telegram formatting can't accidentally corrupt DB records. Removes dead code.

---

### R4: Fix `context.py` isinstance check (information leakage breaking abstraction)

**Problem:** `merge_sources()` does `isinstance(source, NotionSource)` to route `force_refresh`. The Protocol abstraction exists precisely to avoid this.

**Change:** Add `force_refresh: bool = False` to the `PersonalizationSource.get_context` signature. Update `NotionSource.get_context` to accept and honor it. Update `InterestsSource.get_context` to accept and ignore it. Update `merge_sources()` to pass `force_refresh` uniformly.

**Files:** `patronus/context.py`, `patronus/notion.py`, `patronus/interests.py`, `patronus/pipeline.py`.

**Benefit:** `merge_sources()` treats all sources uniformly. Any future refreshable source works without touching `merge_sources`. The isinstance import disappears.

---

### R5: Deduplicate `_now_utc()` (duplicated concept)

**Problem:** Three identical implementations in `db.py`, `digest.py`, `notion_mirror.py`.

**Change:** Keep the one in `db.py` (already the most used). Import it in `digest.py`. For `notion_mirror.py` (which uses a separate SQLite connection and has no `db.py` dependency), define it locally or in a minimal `patronus/_util.py`.

**Files:** `patronus/digest.py`, `patronus/notion_mirror.py`.

**Benefit:** One canonical time-formatting function. If the format changes, you change it once.

---

### R6: Decouple tool registration from `pipeline.py` (change amplification)

**Problem:** Adding a new tool requires editing `pipeline.py` to call `register_new_tools`. The pipeline is responsible for tool wiring, which means it changes for every new tool.

**Change:** Register tools in `ToolRegistry.__init__` or via a `register_all(config, db, registry)` function in `tools/__init__.py` that `pipeline.py` calls as a single import. Individual tool modules remain self-contained.

**Files:** `patronus/tools/__init__.py`, `patronus/pipeline.py`.

**Benefit:** Adding a tool is a one-file change (`tools/new_tool.py` + one line in `tools/__init__.py`) instead of touching `pipeline.py`. The pipeline no longer needs to know about individual tool modules.
