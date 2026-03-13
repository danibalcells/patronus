from __future__ import annotations

import logging
from datetime import datetime, timezone

from patronus.agent import plan_and_assemble
from patronus.config import Config
from patronus.context import Context, PersonalizationSource, merge_sources
from patronus.db import Database
from patronus.digest import Digest, generate_digest_deterministic
from patronus.interests import InterestsSource
from patronus.observability import pipeline_run
from patronus.output import Output
from patronus.output.terminal import TerminalOutput
from patronus.tools import ToolRegistry
from patronus.tools.arxiv import register_arxiv_tools
from patronus.tools.local import register_local_tools
from patronus.tools.notion import register_notion_tools
from patronus.tools.openalex import register_openalex_tools

logger = logging.getLogger(__name__)


def _build_default_sources(config: Config, db: Database) -> list[PersonalizationSource]:
    if config.notion and config.notion_token:
        try:
            from patronus.notion import NotionSource
            return [NotionSource(db=db)]
        except Exception:
            logger.warning("Failed to initialize NotionSource, falling back to static interests", exc_info=True)

    return [InterestsSource()]


def _build_tool_registry(config: Config, db: Database) -> ToolRegistry:
    registry = ToolRegistry()
    register_local_tools(registry, config, db)
    register_arxiv_tools(registry, db=db)
    register_notion_tools(registry, config)
    register_openalex_tools(registry, config, db=db)
    return registry


class DigestPipeline:
    def __init__(
        self,
        config: Config,
        db: Database,
        sources: list[PersonalizationSource] | None = None,
        outputs: list[Output] | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._sources = sources if sources is not None else _build_default_sources(config, db)
        self._outputs = outputs if outputs is not None else []

    def generate(self) -> Digest:
        if self._config.digest.mode == "agent":
            return self._generate_agent()
        return generate_digest_deterministic(self._config, self._db)

    def run(self, *, skip_penalty: bool = False, notion_force_refresh: bool = False) -> Digest:
        with pipeline_run(
            "digest-pipeline",
            {"mode": self._config.digest.mode, "outputs": [type(o).__name__ for o in self._outputs]},
        ) as run_obs:
            if self._config.digest.mode == "agent":
                digest = self._generate_agent(notion_force_refresh=notion_force_refresh)
            else:
                digest = generate_digest_deterministic(self._config, self._db, skip_penalty=skip_penalty)

            self._save_digest(digest)

            for output in self._outputs:
                try:
                    output.send(digest, self._config)
                except Exception:
                    logger.exception("Output %s failed", type(output).__name__)

            run_obs.update(output={
                "sections": len(digest.sections),
                "items": digest.item_count,
                "section_types": [s.type.value for s in digest.sections],
            })

        return digest

    def _generate_agent(self, notion_force_refresh: bool = False) -> Digest:
        logger.info("Generating agent digest")

        context = merge_sources(self._sources, self._config, notion_force_refresh=notion_force_refresh)
        logger.info("Personalization context: %d chars prose, %d interest vectors",
                     len(context.prose), len(context.vectors))

        if not context.prose.strip():
            logger.warning("No personalization context available, falling back to deterministic")
            return generate_digest_deterministic(self._config, self._db)

        registry = _build_tool_registry(self._config, self._db)
        logger.info("Tool registry: %s", registry.tool_names)

        digest = plan_and_assemble(self._config, context, registry, db=self._db)

        if digest.item_count == 0:
            logger.warning("Agent produced empty digest, falling back to deterministic")
            return generate_digest_deterministic(self._config, self._db)

        return digest

    def _save_digest(self, digest: Digest) -> None:
        try:
            from patronus.output.telegram import format_telegram
            formatted = format_telegram(digest, self._config)
        except Exception:
            formatted = ""

        items_data = []
        for section in digest.sections:
            for item in section.items:
                items_data.append({
                    "item_id": item.item_id or (item.scored_item.item.id if item.scored_item else ""),
                    "title": item.title,
                    "summary": item.summary,
                    "score": item.scored_item.score if item.scored_item else 0.0,
                    "matched_topic": item.scored_item.matched_topic if item.scored_item else "",
                    "section_type": section.type.value,
                })
        for item in digest.items:
            items_data.append({
                "item_id": item.item_id or (item.scored_item.item.id if item.scored_item else ""),
                "title": item.title,
                "summary": item.summary,
                "score": item.scored_item.score if item.scored_item else 0.0,
                "matched_topic": item.scored_item.matched_topic if item.scored_item else "",
                "section_type": None,
            })

        self._db.save_digest(
            generated_at=digest.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            item_count=digest.item_count,
            formatted_text=formatted,
            items=items_data,
        )
