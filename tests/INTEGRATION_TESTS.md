# Integration Tests for Patronus Agent & Tools

This directory contains integration tests for the DAN-17 agent implementation and related Stage 2 modules.

## Overview

The integration tests verify that the agent, tools, and pipeline work correctly with real LLM and embedding API calls. These tests complement the existing unit tests (which use mocks) by ensuring the full system works end-to-end.

## Test Files

### `test_agent_integration.py`
Tests the agent's ability to:
- Generate digests with proper sections
- Use search tools effectively
- Handle various context inputs
- Respect personalization context
- Write appropriate summaries for different section types

**Requirements:** `ANTHROPIC_API_KEY` (for Claude API)

### `test_pipeline_integration.py`
Tests the complete pipeline:
- Full agent-mode digest generation
- Multiple personalization sources
- Multiple outputs
- Fallback to deterministic mode
- Database persistence
- Error handling

**Requirements:** `ANTHROPIC_API_KEY` (for agent), `OPENAI_API_KEY` (for embeddings in deterministic tests)

### `test_tools_integration.py`
Tests retrieval tools with real data:
- `SearchSimilar` with real embeddings and semantic search
- `SearchRecent` with time-based filtering
- `SearchByTopic` with topic clustering
- `SearchBySource` with source filtering
- Tool coordination and registry execution

**Requirements:** `OPENAI_API_KEY` (for embedding generation)

## Setup

### 1. Install Dependencies

```bash
uv sync
source .venv/bin/activate
```

### 2. Set API Keys

Create or update your `.env` file:

```bash
# Required for agent tests
ANTHROPIC_API_KEY=sk-ant-...

# Required for embedding and some deterministic tests
OPENAI_API_KEY=sk-...

# Optional (not needed for current integration tests)
GOOGLE_API_KEY=...
```

### 3. Verify Configuration

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('ANTHROPIC_API_KEY:', 'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

## Running Tests

### Run All Unit Tests (Default)

By default, pytest skips integration tests:

```bash
pytest
```

### Run Only Integration Tests

```bash
pytest -m integration
```

### Run All Tests (Unit + Integration)

```bash
pytest -m ""
```

### Run Specific Integration Test File

```bash
pytest -m integration tests/test_agent_integration.py
```

### Run Specific Test Class or Function

```bash
# Run all agent integration tests
pytest -m integration tests/test_agent_integration.py::TestAgentIntegration

# Run a specific test
pytest -m integration tests/test_agent_integration.py::TestAgentIntegration::test_agent_generates_digest_with_sections
```

### Verbose Output

```bash
pytest -m integration -v
```

### Show Print Statements and Logs

```bash
pytest -m integration -s
```

## Cost Considerations & Caching

Integration tests make real API calls, which incur costs:

- **Anthropic API (Claude):** ~$0.01-0.05 per test run depending on context size
- **OpenAI API (embeddings):** ~$0.0001 per embedding, very low cost
- **Total cost for full integration suite:** ~$0.10-0.50 per run

### Notion Context Caching (DAN-25)

The pipeline tests use **cached Notion context by default** (24-hour TTL), which significantly reduces:
- Test execution time (no Notion API calls)
- API costs (no Claude summarization calls)
- Overall reliability (tests don't fail if Notion is slow)

Tests call `pipeline.run(notion_force_refresh=False)` to use cached context. The cache is stored in the test database's `ContextSnapshot` table.

To minimize costs:
1. Run integration tests only when needed (not on every commit)
2. Let caching work - first run is slower, subsequent runs are fast
3. Use `-k` to run specific tests: `pytest -m integration -k "test_agent_generates"`
4. Run unit tests (free) first: `pytest` (default)

## Test Structure

### Integration Test Patterns

All integration tests follow these patterns:

1. **Skip if API keys missing:** Tests are automatically skipped if required API keys aren't set
   ```python
   @pytest.mark.skipif(
       not os.getenv("ANTHROPIC_API_KEY"),
       reason="ANTHROPIC_API_KEY not set - skipping integration test",
   )
   ```

2. **Use real databases:** Tests use temporary SQLite databases (`tmp_path` fixture)

3. **Create realistic test data:** Tests populate databases with diverse, realistic content

4. **Verify real behavior:** Tests check that the agent makes sensible decisions, tools return relevant results, etc.

### Fixtures

Key fixtures used across integration tests:

- `integration_config`: A Config object with realistic settings
- `test_db`: A temporary database populated with test items
- `test_db_with_real_embeddings`: Database with items embedded using real OpenAI API
- `tool_registry`: ToolRegistry with all local tools registered

## Continuous Integration

For CI/CD pipelines:

```bash
# Run only unit tests in CI (fast, no API costs)
pytest

# Run integration tests only on main branch or manually
if [[ "$GITHUB_REF" == "refs/heads/main" ]] || [[ "$RUN_INTEGRATION_TESTS" == "true" ]]; then
  pytest -m integration
fi
```

## Manual Testing with Visible Output

For debugging and development, use the **manual test script** instead of integration tests:

```bash
# Run full pipeline manually (uses test DB + real Notion context)
python scripts/test_agent_manual.py

# Force fresh Notion context (bypass 24h cache)
python scripts/test_agent_manual.py --force-notion-refresh

# Use production database
python scripts/test_agent_manual.py --use-prod-db --force-notion-refresh

# Recreate test database
python scripts/test_agent_manual.py --recreate-db

# Use specific database path
python scripts/test_agent_manual.py --db-path /tmp/my_test.db
```

The script shows:
- Full pipeline execution (same as production)
- Personalization context from Notion + interests
- All LLM tool calls with parameters
- Complete digest structure with formatted output
- Cache status and source information

**This mirrors production exactly:**
- Uses DigestPipeline (not just agent)
- Real Notion context via NotionSource
- Real interests via InterestsSource  
- Default 24h caching behavior
- Same flow as `send_digest.py`

**This is better than tests for:**
- Understanding actual pipeline behavior
- Debugging with real personalization
- Iterating on prompts and context
- Seeing actual digest quality
- Manual verification before deployment

## Debugging Integration Tests

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
pytest -m integration -s
```

### Inspect Agent Tool Calls

Add logging to see what tools the agent is calling:

```bash
pytest -m integration -s --log-cli-level=INFO tests/test_agent_integration.py
```

### See Progress During Long Tests

Don't pipe output to `/dev/null` or use `-q`. Use `-v` instead:

```bash
# Good - shows progress
pytest -m integration -v

# Bad - no progress feedback
pytest -m integration -q 2>/dev/null
```

### Check Database State

Tests use temporary databases. To inspect, modify a test to not clean up:

```python
@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path) + "/test.db"
    print(f"Database at: {db_path}")  # Will show path
    db = Database(db_path=db_path)
    # ... populate ...
    yield db
    # Don't close to inspect
```

## Expected Test Duration

**First run (no cache):**
- `test_agent_integration.py`: ~30-60 seconds (agent makes LLM calls with tool use)
- `test_pipeline_integration.py`: ~30-60 seconds (full pipeline with agent)
- `test_tools_integration.py`: ~5-10 seconds (embedding calls, no agent)

**Subsequent runs (with cache):**
- `test_agent_integration.py`: ~30-60 seconds (same, uses mock sources)
- `test_pipeline_integration.py`: ~20-30 seconds (faster with Notion cache)
- `test_tools_integration.py`: ~5-10 seconds (same, no Notion)

Total: ~1-2 minutes first run, ~40-80 seconds with cache.

## Troubleshooting

### Tests Skip with "API key not set"

**Solution:** Check your `.env` file and ensure you've loaded it:
```bash
cat .env | grep API_KEY
source .venv/bin/activate  # Load environment
```

### "Module not found" errors

**Solution:** Ensure you're in the project root and have activated the virtualenv:
```bash
cd /Users/dani/code/patronus
source .venv/bin/activate
pytest -m integration
```

### API Rate Limits

If you hit rate limits:
1. Wait a minute and retry
2. Run fewer tests at once: `pytest -m integration -k "specific_test"`
3. Check your API key quota on the provider's dashboard

### Slow Tests

Integration tests are inherently slower due to API calls. To speed up development:
1. Run unit tests first: `pytest` (fast)
2. Run integration tests less frequently
3. Use `-k` to run only relevant integration tests

## Adding New Integration Tests

When adding new integration tests:

1. **Mark as integration:**
   ```python
   @pytest.mark.integration
   @pytest.mark.skipif(
       not os.getenv("REQUIRED_API_KEY"),
       reason="API key not set",
   )
   class TestMyFeatureIntegration:
       ...
   ```

2. **Use fixtures for setup:** Reuse existing fixtures or create new ones in `conftest.py`

3. **Make tests realistic:** Use diverse test data that resembles production

4. **Document API requirements:** Add comments about which APIs are needed

5. **Keep tests focused:** Each test should verify one specific behavior

6. **Add cost estimates:** Document approximate API costs in comments

## Related Documentation

- Unit tests (with mocks): See `test_agent.py`, `test_tools.py`, `test_pipeline.py`
- Project architecture: See `README.md` in project root
- Linear issue: DAN-17 - Agent digest editor (core Stage 2)
