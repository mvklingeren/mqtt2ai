"""AI package for MQTT AI Daemon.

This package contains AI agent, orchestrator, providers, tools,
and prompt building components.
"""
from mqtt2ai.ai.agent import AiAgent
from mqtt2ai.ai.orchestrator import AiOrchestrator
from mqtt2ai.ai.prompt_builder import PromptBuilder
from mqtt2ai.ai.prompt_templates import COMPACT_RULEBOOK
from mqtt2ai.ai.tool_definitions import OPENAI_TOOLS, OPENAI_TOOLS_MINIMAL
from mqtt2ai.ai import tools
from mqtt2ai.ai.tools import ToolHandler
from mqtt2ai.ai.alert_handler import AlertHandler

__all__ = [
    "AiAgent",
    "AiOrchestrator",
    "AlertHandler",
    "PromptBuilder",
    "COMPACT_RULEBOOK",
    "OPENAI_TOOLS",
    "OPENAI_TOOLS_MINIMAL",
    "ToolHandler",
    "tools",
]
