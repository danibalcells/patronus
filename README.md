## Patronus

Curate high-signal RSS/Atom content into topic-specific feeds using an LLM + a human profile, and publish the curated feeds to Google Cloud Storage (GCS).

### Key features
- **Classification by profile**: Articles are classified into buckets using `OpenAI` Chat Completions with `Profile.md` as guidance.
- **Non-destructive dry run**: Safely test end-to-end without uploading or persisting state.
- **Per-bucket outputs**: Generates both RSS and Atom feeds per category.
- **Continuous mode**: Async polling loop that detects new links and classifies concurrently.

### Buckets
`REJECT`, `TECHNICAL_AI_AND_ML`, `TECH_BEYOND_THE_TECHNICAL`, `PHILOSOPHY_CONSCIOUSNESS`, `POLITICS_CULTURE`, `SPAIN`, `CHINA`, `RANDOM_CURIOSITIES`.


## Repository structure
- `patronus.py`: Batch run. Reads feeds + profile, classifies, prints summary, and (if not dry-run) uploads curated feeds to GCS.
- `polling_loop.py`: Continuous async loop. Re-indexes feeds, classifies new links concurrently, and (if not dry-run) uploads and persists state.
- `Profile.md`: Curation criteria that drive classification buckets.
- `feeds` (path provided at runtime): Text file with feed URLs, one per line.
- `pyproject.toml`: Python project metadata and dependencies.
- `uv.lock`: Lockfile if you use `uv` for dependency management.


## Requirements
- Python 3.11+
- Google Cloud account with a project and a GCS bucket
- `gcloud` CLI installed and authenticated for local development


## Installation
You can use either `uv` or plain `pip`.

### Using uv (recommended if available)
```bash
cd /home/dani/code/patronus
uv sync
```

### Using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install feedgen feedparser google-cloud-storage openai pydantic python-dotenv trafilatura jupyter ipython
```


## Configuration
Set these environment variables (use a shell export or a `.env` file; `python-dotenv` is loaded automatically):

- `OPENAI_API_KEY` (required): API key used by the OpenAI client.
- `GCS_BUCKET_NAME` (required for real uploads): Destination bucket for curated feeds.
- `GCS_PREFIX` (optional): Path prefix within the bucket. Defaults to `patronus/feeds/`.

Example `.env`:
```bash
OPENAI_API_KEY=sk-...
GCS_BUCKET_NAME=my-curated-feeds-bucket
GCS_PREFIX=patronus/feeds/
```


## Google Cloud authentication
The Python GCS client uses Application Default Credentials (ADC).

Local developer setup (user credentials):
```bash
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
```

Service account option:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json
gcloud config set project <YOUR_PROJECT_ID>
```

Ensure the identity used (user or service account) has permissions to write to the bucket (e.g., `roles/storage.objectAdmin` on the target bucket).


## Feeds and profile
- `Profile.md` describes the curation rules and buckets.
- The feeds file is a plain text file, one feed URL per line. Example:
```text
https://example.com/feed.xml
https://another.example.org/atom.xml
```


## Running the batch mode
Classifies a limited set of recent entries and uploads curated feeds (unless `--dry-run`).

```bash
python /home/dani/code/patronus/patronus.py \
  --feeds-path /abs/path/to/feeds \
  --profile-path /home/dani/code/patronus/Profile.md
```

Dry run (no uploads, safer for testing):
```bash
python /home/dani/code/patronus/patronus.py \
  --feeds-path /abs/path/to/feeds \
  --profile-path /home/dani/code/patronus/Profile.md \
  --dry-run
```

Notes:
- In dry run, classification still happens and a summary is printed; uploads are skipped.
- The batch script caps total items (40 in dry run, 50 otherwise).


## Running the continuous polling loop
Polls on an interval, classifies new links concurrently, and uploads curated feeds.

```bash
python /home/dani/code/patronus/polling_loop.py \
  --feeds-path /abs/path/to/feeds \
  --profile-path /home/dani/code/patronus/Profile.md \
  --state-path /home/dani/code/patronus/.patronus_poll_state.json \
  --interval-seconds 60 \
  --concurrency 8
```

Dry run (no uploads or disk state persistence):
```bash
python /home/dani/code/patronus/polling_loop.py \
  --feeds-path /abs/path/to/feeds \
  --profile-path /home/dani/code/patronus/Profile.md \
  --state-path /home/dani/code/patronus/.patronus_poll_state.json \
  --interval-seconds 60 \
  --concurrency 8 \
  --dry-run
```

Notes:
- In dry run, in-memory state is updated so duplicates are avoided during the same process run, but nothing is written to disk and nothing is uploaded.
- On real runs, `--state-path` stores `seen_links` and `assignments` to avoid reprocessing across restarts.


## Output
For each bucket, two artifacts are produced (on real runs):
- RSS: `${GCS_PREFIX}${BUCKET}.rss.xml`
- Atom: `${GCS_PREFIX}${BUCKET}.atom.xml`

Public URLs are logged/printed after upload.


## Logging
- Logs are emitted to stdout and, when available, to the system log via `SysLogHandler`.
- Useful signals include poll cycles, number of indexed items, new link counts, classification progress, and upload status.


## How classification works (high level)
- For each candidate article, the HTML is fetched and cleaned with `trafilatura` (falling back to the feed summary when needed).
- The OpenAI Chat Completions API (`gpt-5-mini`, JSON mode) assigns a bucket using the guidance in `Profile.md`.
- The system aggregates articles by bucket, then composes per-bucket RSS and Atom feeds, preserving as much original metadata as possible and adding fallbacks for missing author info.


## Troubleshooting
- If uploads fail, verify `GCS_BUCKET_NAME`, project selection (`gcloud config get-value project`), and credentials (`gcloud auth application-default print-access-token`).
- If articles reprocess repeatedly, confirm that `--state-path` is consistent and writable on real runs. In dry run, no disk state is saved.
- Ensure feed URLs are reachable and return valid RSS/Atom; malformed feeds will be skipped.


