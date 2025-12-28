"""MQTT2AI - Smart MQTT to AI integration daemon.

This package provides an intelligent MQTT monitoring daemon that uses
AI to analyze message patterns, detect anomalies, and automate actions.

Subpackages:
- core: Configuration, event bus, utilities, and main daemon
- mqtt: MQTT client, collector, and simulator
- ai: AI agent, orchestrator, providers, and tools
- rules: Rule engine, trigger analyzer, and knowledge base
- telegram: Telegram bot integration
"""

from mqtt2ai.core.config import Config
from mqtt2ai.core.daemon import MqttAiDaemon
from mqtt2ai.core.event_bus import event_bus, EventBus, EventType, Event

__version__ = "0.4"

__all__ = [
    "Config",
    "MqttAiDaemon",
    "event_bus",
    "EventBus",
    "EventType",
    "Event",
    "__version__",
]
