"""Constants used throughout the MQTT2AI application.

This module centralizes magic strings and values to:
- Prevent typos and inconsistencies
- Make refactoring easier
- Provide a single source of truth for configuration values
"""


# AI Provider identifiers
class AiProvider:
    """AI provider name constants."""
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI_COMPATIBLE = "openai-compatible"


# MQTT Topics
class MqttTopics:
    """Standard MQTT topic constants used by the daemon."""
    ACTION_ANNOUNCE = "mqtt2ai/action/announce"
    ALERTS = "mqtt2ai/alerts"


# Trigger reasons
class TriggerReason:
    """Constants for AI trigger reasons."""
    SMART_TRIGGER = "smart_trigger"
    MANUAL = "manual (Enter pressed)"

    @staticmethod
    def message_count(threshold: int) -> str:
        """Generate message count trigger reason."""
        return f"message_count ({threshold})"

    @staticmethod
    def interval(seconds: int) -> str:
        """Generate interval trigger reason."""
        return f"interval ({seconds}s)"


# Terminal colors for output formatting
class TermColors:
    """ANSI color codes for terminal output."""
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"


# Rule engine source identifiers
class RuleSource:
    """Source identifiers for rule-triggered actions."""
    DIRECT_RULE = "direct_rule"
    AI_ANALYSIS = "ai_analysis"


# Message prefixes
class MessagePrefix:
    """Prefixes used in message formatting."""
    RETAINED = "[RETAINED]"
