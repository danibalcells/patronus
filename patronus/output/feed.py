from __future__ import annotations

import logging

from patronus.config import Config
from patronus.digest import Digest

logger = logging.getLogger(__name__)


class FeedOutput:
    def send(self, digest: Digest, config: Config) -> None:
        logger.info("XML/Atom feed output is not yet implemented. Skipping.")
