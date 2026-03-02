from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Load environment variables from .env at test startup."""
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)

    try:
        import patronus.observability as obs
        obs._initialized = False
        obs._langfuse_client = None
    except ImportError:
        pass
