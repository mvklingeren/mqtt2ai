"""Telegram package for MQTT AI Daemon.

This package provides bidirectional Telegram integration
for alerts, notifications, and user commands.
"""
from mqtt2ai.telegram.bot import TelegramBot, TELEGRAM_AVAILABLE
from mqtt2ai.telegram.handler import TelegramHandler

__all__ = [
    "TelegramBot",
    "TelegramHandler",
    "TELEGRAM_AVAILABLE",
]
