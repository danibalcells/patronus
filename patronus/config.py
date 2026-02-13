from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TopicConfig:
    name: str
    description: str


@dataclass
class DigestConfig:
    size: int = 7
    max_per_topic: int = 3
    schedule: str = "08:00"
    timezone: str = "Europe/Madrid"


@dataclass
class PollingConfig:
    interval_hours: int = 2


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"


@dataclass
class SummarizationConfig:
    model: str = "claude-sonnet-4-20250514"


@dataclass
class TelegramConfig:
    chat_id: str = ""


@dataclass
class Config:
    digest: DigestConfig
    polling: PollingConfig
    embedding: EmbeddingConfig
    summarization: SummarizationConfig
    telegram: TelegramConfig
    topics: dict[str, TopicConfig]
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    telegram_bot_token: str = ""


def load_config(
    config_path: Optional[str | Path] = None,
    interests_path: Optional[str | Path] = None,
) -> Config:
    load_dotenv()

    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
    if interests_path is None:
        interests_path = _PROJECT_ROOT / "config" / "interests.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    with open(interests_path) as f:
        interests_raw = yaml.safe_load(f) or {}

    topics = {}
    for key, topic_data in interests_raw.get("topics", {}).items():
        topics[key] = TopicConfig(
            name=topic_data["name"],
            description=topic_data["description"].strip(),
        )

    return Config(
        digest=DigestConfig(**raw.get("digest", {})),
        polling=PollingConfig(**raw.get("polling", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        summarization=SummarizationConfig(**raw.get("summarization", {})),
        telegram=TelegramConfig(**raw.get("telegram", {})),
        topics=topics,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    )
