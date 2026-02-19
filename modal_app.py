import modal

app = modal.App("patronus")

volume = modal.Volume.from_name("patronus-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("patronus")
    .add_local_dir("config", remote_path="/root/config")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

VOLUME_DIR = "/data"
SECRETS = [modal.Secret.from_name("patronus-secrets")]


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    secrets=SECRETS,
    schedule=modal.Cron("30 7 * * *", timezone="America/New_York"),
    timeout=900,
)
def send_digest(tag: str = ""):
    from patronus import setup_logging
    from patronus.config import load_config
    from patronus.db import Database
    from patronus.output.feed import FeedOutput
    from patronus.output.terminal import TerminalOutput
    from patronus.pipeline import DigestPipeline

    setup_logging()
    config = load_config()
    if config.notion:
        config.notion.mirror_path = f"{VOLUME_DIR}/notion_mirror.sqlite3"

    outputs = [TerminalOutput(), FeedOutput(tag=tag or None)]

    with Database(db_path=f"{VOLUME_DIR}/db.sqlite3") as db:
        pipeline = DigestPipeline(config, db, outputs=outputs)
        pipeline.run()

    volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    secrets=SECRETS,
    schedule=modal.Cron("0 */2 * * *"),
    timeout=600,
)
def poll_feeds():
    from patronus import setup_logging
    from patronus.db import Database
    from patronus.ingest import poll_feeds as _poll

    setup_logging()

    with Database(db_path=f"{VOLUME_DIR}/db.sqlite3") as db:
        ids = _poll(db, workers=4)
        print(f"Ingested {len(ids)} new item(s)")

    volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    secrets=SECRETS,
    schedule=modal.Cron("0 2 * * *"),
    timeout=600,
)
def sync_notion():
    import sys

    sys.path.insert(0, "/root/scripts")

    from patronus import setup_logging
    from patronus.config import load_config
    from patronus.notion_mirror import open_mirror
    from sync_notion_mirror import sync

    setup_logging()
    config = load_config()

    with open_mirror(f"{VOLUME_DIR}/notion_mirror.sqlite3") as mirror:
        counts = sync(mirror, config)
        total = sum(counts.values())
        print(f"Synced {total} pages across {len(counts)} databases")

    volume.commit()


@app.local_entrypoint()
def main(job: str = "digest", tag: str = ""):
    if job == "digest":
        send_digest.remote(tag=tag)
    elif job == "poll":
        poll_feeds.remote()
    elif job == "sync":
        sync_notion.remote()
    else:
        raise SystemExit(f"Unknown job: {job}. Use: digest, poll, sync")
