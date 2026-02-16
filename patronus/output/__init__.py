from __future__ import annotations

from typing import Protocol, runtime_checkable

from patronus.config import Config
from patronus.digest import Digest


@runtime_checkable
class Output(Protocol):
    def send(self, digest: Digest, config: Config) -> None: ...


__all__ = ["Output"]
