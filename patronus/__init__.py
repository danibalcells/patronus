import logging


_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "urllib3",
    "notion_client",
]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(name)s | %(message)s")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
