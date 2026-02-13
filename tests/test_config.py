from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from patronus.config import Config, DigestConfig, load_config


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        "digest:\n"
        "  size: 5\n"
        "  max_per_topic: 2\n"
        "  schedule: '09:00'\n"
        "  timezone: 'US/Eastern'\n"
        "polling:\n"
        "  interval_hours: 4\n"
        "embedding:\n"
        "  model: 'text-embedding-3-large'\n"
        "summarization:\n"
        "  model: 'claude-sonnet-4-20250514'\n"
        "telegram:\n"
        "  chat_id: '12345'\n"
    )
    interests_yaml = tmp_path / "interests.yaml"
    interests_yaml.write_text(
        "topics:\n"
        "  ml:\n"
        "    name: 'Machine Learning'\n"
        "    description: 'Technical ML research and papers.'\n"
        "  philosophy:\n"
        "    name: 'Philosophy'\n"
        "    description: 'Philosophy of mind and consciousness.'\n"
    )
    return tmp_path


@pytest.fixture()
def minimal_config_dir(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text("")
    (tmp_path / "interests.yaml").write_text("")
    return tmp_path


class TestLoadConfig:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test", "TELEGRAM_BOT_TOKEN": "bot-tok"})
    def test_loads_full_config(self, config_dir: Path) -> None:
        cfg = load_config(
            config_path=config_dir / "config.yaml",
            interests_path=config_dir / "interests.yaml",
        )
        assert isinstance(cfg, Config)
        assert cfg.digest.size == 5
        assert cfg.digest.max_per_topic == 2
        assert cfg.digest.schedule == "09:00"
        assert cfg.digest.timezone == "US/Eastern"
        assert cfg.polling.interval_hours == 4
        assert cfg.embedding.model == "text-embedding-3-large"
        assert cfg.telegram.chat_id == "12345"
        assert cfg.openai_api_key == "sk-test"
        assert cfg.anthropic_api_key == "sk-ant-test"
        assert cfg.telegram_bot_token == "bot-tok"

    def test_loads_topics(self, config_dir: Path) -> None:
        cfg = load_config(
            config_path=config_dir / "config.yaml",
            interests_path=config_dir / "interests.yaml",
        )
        assert len(cfg.topics) == 2
        assert "ml" in cfg.topics
        assert "philosophy" in cfg.topics
        assert cfg.topics["ml"].name == "Machine Learning"
        assert cfg.topics["ml"].description == "Technical ML research and papers."
        assert cfg.topics["philosophy"].name == "Philosophy"

    def test_defaults_on_empty_yaml(self, minimal_config_dir: Path) -> None:
        cfg = load_config(
            config_path=minimal_config_dir / "config.yaml",
            interests_path=minimal_config_dir / "interests.yaml",
        )
        assert cfg.digest.size == 7
        assert cfg.digest.max_per_topic == 3
        assert cfg.digest.schedule == "08:00"
        assert cfg.digest.timezone == "America/New_York"
        assert cfg.polling.interval_hours == 2
        assert cfg.embedding.model == "text-embedding-3-small"
        assert cfg.topics == {}

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(
                config_path=tmp_path / "nonexistent.yaml",
                interests_path=tmp_path / "also-nonexistent.yaml",
            )

    def test_topic_description_stripped(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("")
        (tmp_path / "interests.yaml").write_text(
            "topics:\n"
            "  test:\n"
            "    name: 'Test'\n"
            "    description: |\n"
            "      A description with trailing newline.\n"
        )
        cfg = load_config(
            config_path=tmp_path / "config.yaml",
            interests_path=tmp_path / "interests.yaml",
        )
        assert not cfg.topics["test"].description.endswith("\n")
