from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


ITEM_SNIPPET_MAX_CHARS = 500


@dataclass
class ToolResult:
    items: list[dict] = field(default_factory=list)
    message: str = ""

    def to_text(self) -> str:
        parts: list[str] = []
        if self.message:
            parts.append(self.message)
        for item in self.items:
            lines = []
            if item.get("title"):
                lines.append(f"Title: {item['title']}")
            if item.get("url"):
                lines.append(f"URL: {item['url']}")
            if item.get("source"):
                lines.append(f"Source: {item['source']}")
            if item.get("author"):
                lines.append(f"Author: {item['author']}")
            if item.get("item_type"):
                lines.append(f"Type: {item['item_type']}")
            if item.get("timestamp"):
                lines.append(f"Date: {item['timestamp']}")
            if item.get("id"):
                lines.append(f"ID: {item['id']}")
            if item.get("snippet"):
                lines.append(f"Snippet: {item['snippet']}")
            parts.append("\n".join(lines))
        return "\n\n---\n\n".join(parts) if parts else "No results found."


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @abstractmethod
    def execute(self, **params: object) -> ToolResult: ...

    def to_definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
