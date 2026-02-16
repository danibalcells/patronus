from __future__ import annotations

import asyncio
import logging
import re

import telegram
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from patronus.config import Config
from patronus.db import Database
from patronus.ingest import ingest_url
from patronus.output.telegram import TelegramOutput, format_telegram_sections, send_message

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096

_URL_PATTERN = re.compile(r"https?://\S+")


def _is_authorized(update: Update, config: Config) -> bool:
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    return chat_id == config.telegram.chat_id


def _pack_sections(sections: list[str], limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    separator = "\n\n"
    messages: list[str] = []
    current = ""
    for section in sections:
        candidate = f"{current}{separator}{section}" if current else section
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                messages.append(current)
            current = section
    if current:
        messages.append(current)
    return messages


async def _send_markdown(bot: telegram.Bot, chat_id: str, text: str) -> None:
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="MarkdownV2",
        disable_web_page_preview=True,
    )


async def _send_sections(bot: telegram.Bot, chat_id: str, sections: list[str]) -> None:
    for message in _pack_sections(sections):
        await _send_markdown(bot, chat_id, message)


async def _handle_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    db: Database = context.bot_data["db"]
    if not _is_authorized(update, config):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /add <url>")
        return

    url = args[0]
    await update.message.reply_text(f"Ingesting {url}...")

    try:
        item_id = await asyncio.to_thread(ingest_url, db, url)
        if item_id:
            item = db.get_item(item_id)
            title = item.title if item and item.title else url
            await update.message.reply_text(f"Added: {title}")
        else:
            await update.message.reply_text("URL already exists in the database.")
    except Exception:
        logger.exception("Failed to ingest URL: %s", url)
        await update.message.reply_text("Failed to ingest URL. Check logs for details.")


async def _handle_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    db: Database = context.bot_data["db"]
    if not _is_authorized(update, config):
        return

    await update.message.reply_text("Generating digest...")

    try:
        from patronus.pipeline import DigestPipeline
        pipeline = DigestPipeline(config, db)
        digest = await asyncio.to_thread(pipeline.run)

        if not digest.item_count:
            await update.message.reply_text("No unread items to build a digest from.")
            return

        sections = format_telegram_sections(digest, config)
        await _send_sections(context.bot, config.telegram.chat_id, sections)
    except Exception:
        logger.exception("Failed to generate digest")
        await update.message.reply_text("Failed to generate digest. Check logs for details.")


async def _handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    db: Database = context.bot_data["db"]
    if not _is_authorized(update, config):
        return

    try:
        total = db.get_item_count()
        unread = db.get_unread_count()
        feeds = db.get_feed_count()
        digests = db.get_latest_digests(1)
        last_digest = digests[0].generated_at if digests else "never"

        lines = [
            f"Items: {total} total, {unread} unread",
            f"Active feeds: {feeds}",
            f"Last digest: {last_digest}",
            f"Mode: {config.digest.mode}",
        ]
        await update.message.reply_text("\n".join(lines))
    except Exception:
        logger.exception("Failed to get status")
        await update.message.reply_text("Failed to get status.")


async def _handle_url_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    db: Database = context.bot_data["db"]
    if not _is_authorized(update, config):
        return

    text = (update.message.text or "").strip()
    match = _URL_PATTERN.match(text)
    if not match or match.group() != text:
        return

    url = text
    await update.message.reply_text(f"Ingesting {url}...")

    try:
        item_id = await asyncio.to_thread(ingest_url, db, url)
        if item_id:
            item = db.get_item(item_id)
            title = item.title if item and item.title else url
            await update.message.reply_text(f"Added: {title}")
        else:
            await update.message.reply_text("URL already exists in the database.")
    except Exception:
        logger.exception("Failed to ingest URL: %s", url)
        await update.message.reply_text("Failed to ingest URL. Check logs for details.")


def run_bot(config: Config, db: Database) -> None:
    if not config.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(config.telegram_bot_token).build()
    app.bot_data["config"] = config
    app.bot_data["db"] = db

    app.add_handler(CommandHandler("add", _handle_add))
    app.add_handler(CommandHandler("digest", _handle_digest))
    app.add_handler(CommandHandler("status", _handle_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_url_message))

    logger.info("Starting Telegram bot")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
