from __future__ import annotations

import logging

from patronus.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[dict]:
        return [tool.to_definition() for tool in self._tools.values()]

    def execute(self, name: str, **params: object) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(message=f"Unknown tool: {name}")
        try:
            return tool.execute(**params)
        except Exception:
            logger.exception("Tool %s failed", name)
            return ToolResult(message=f"Tool {name} failed with an internal error.")

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


__all__ = ["Tool", "ToolResult", "ToolRegistry"]
