"""Configuration module for the MQTT AI Daemon.

This module provides the Config dataclass which holds all configuration
settings for the daemon, including MQTT connection details, AI provider
settings, and filtering options.
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Load .env file if present (for API keys etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables


# Default models for round-robin (Groq free tier, excluding guard models)
DEFAULT_OPENAI_MODELS = [
    "llama-3.3-70b-versatile",  # More reliable tool calling than Llama 4
    # "meta-llama/llama-4-maverick-17b-128e-instruct",
    # "meta-llama/llama-4-scout-17b-16e-instruct",  # Unreliable tool calling
]


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Configuration settings for the MQTT AI Daemon."""

    mqtt_host: str = field(default_factory=lambda: os.environ.get("MQTT_HOST", "192.168.1.245"))
    mqtt_port: str = field(default_factory=lambda: os.environ.get("MQTT_PORT", "1883"))
    mqtt_topics: List[str] = field(
        default_factory=lambda: [
            "zigbee2mqtt/#",
            "jokes/#",
            "mqtt2ai/#"  # Causation announcements for automated actions
        ]
    )
    max_messages: int = 1000  # Keep last 500 messages in buffer
    ai_check_interval: int = 600  # 10 minutes
    ai_check_threshold: int = 300  # Trigger AI after 300 messages
    disable_interval_trigger: bool = False  # Disable time-based AI triggers
    disable_threshold_trigger: bool = False  # Disable message count-based AI triggers

    # Device state tracking for alert system
    device_track_pattern: str = "zigbee2mqtt/*"  # Glob pattern for topics to track

    # Files
    filtered_triggers_file: str = "configs/filtered_triggers.json"
    learned_rules_file: str = "configs/learned_rules.json"
    pending_patterns_file: str = "configs/pending_patterns.json"
    rejected_patterns_file: str = "configs/rejected_patterns.json"

    # AI Provider
    ai_provider: str = "openai-compatible"  # "openai-compatible" (Groq), "gemini", "claude"

    # Gemini (Google AI)
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )

    # Claude (Anthropic)
    claude_model: str = "claude-3-5-haiku-latest"
    claude_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # OpenAI-compatible API (Groq, Ollama, LM Studio, vLLM, etc.)
    openai_api_base: str = "https://api.groq.com/openai/v1"  # Groq (blazing fast!)
    openai_api_key: str = ""  # Set via GROQ_API_KEY or --openai-api-key
    openai_models: List[str] = field(default_factory=DEFAULT_OPENAI_MODELS.copy)
    _model_index: int = field(default=0, repr=False)  # Internal counter for round-robin

    def get_next_model(self) -> str:
        """Get the next model in round-robin rotation."""
        if not self.openai_models:
            return "llama-3.3-70b-versatile"  # Fallback
        model = self.openai_models[self._model_index % len(self.openai_models)]
        self._model_index += 1
        return model

    @property
    def openai_model(self) -> str:
        """Return current model without advancing (for display/logging)."""
        if not self.openai_models:
            return "llama-3.3-70b-versatile"
        return self.openai_models[self._model_index % len(self.openai_models)]

    # Filtering & Display
    verbose: bool = False
    compress_output: bool = False  # Compress MQTT payloads in console output
    demo_mode: bool = False
    no_ai: bool = False  # Run without making AI calls (logging only)
    test_ai: bool = False  # Test AI connection before starting daemon
    # If True, new rules are disabled by default. Set via env var or --disable-new-rules
    disable_new_rules: bool = field(
        default_factory=lambda: os.environ.get(
            "DISABLE_NEW_RULES", ""
        ).lower() in ("1", "true", "yes")
    )
    ignore_printing_topics: List[str] = field(
        default_factory=lambda: [
            "zigbee2mqtt/bridge/logging",
            "zigbee2mqtt/bridge/health"
        ]
    )
    ignore_printing_prefixes: List[str] = field(
        default_factory=lambda: [
            "stat/",  # Tasmota logging/status messages
        ]
    )

    # Simulation mode
    simulation_file: Optional[str] = None  # Path to simulation scenario JSON file
    simulation_speed: Optional[float] = None  # Override speed multiplier for simulation

    # Test mode - validate assertions and exit with pass/fail status
    test_mode: bool = False
    test_report_file: Optional[str] = None  # Path to write JSON test report

    # Debug mode - write HTTP call details to files
    debug_output: bool = False
    debug_output_dir: str = "debug-output"

    # Telegram Bot integration
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_ids: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_CHAT_IDS", "")
    )
    telegram_enabled: bool = field(
        default_factory=lambda: bool(os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    )

    @classmethod
    def from_args(cls) -> 'Config':
        """Parse command-line arguments and return a Config instance."""
        parser = argparse.ArgumentParser(
            description="MQTT AI Daemon - Smart home automation with AI"
        )

        parser.add_argument(
            "--mqtt-host",
            default=os.environ.get("MQTT_HOST", "192.168.1.245"),
            help="MQTT Broker Host"
        )
        parser.add_argument(
            "--mqtt-port",
            default=os.environ.get("MQTT_PORT", "1883"),
            help="MQTT Broker Port"
        )

        # AI Provider
        parser.add_argument(
            "--ai-provider",
            choices=["gemini", "claude", "openai-compatible"],
            default=os.environ.get("AI_PROVIDER", "openai-compatible"),
            help="AI provider to use (gemini, claude, or openai-compatible)"
        )

        # Gemini (Google AI)
        parser.add_argument(
            "--gemini-model",
            default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            help="Gemini Model ID"
        )
        parser.add_argument(
            "--gemini-api-key",
            default=os.environ.get("GEMINI_API_KEY", ""),
            help="API key for Google Gemini (uses GEMINI_API_KEY env var)"
        )

        # Claude (Anthropic)
        parser.add_argument(
            "--claude-model",
            default=os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-latest"),
            help="Claude Model ID"
        )
        parser.add_argument(
            "--claude-api-key",
            default=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="API key for Anthropic Claude (uses ANTHROPIC_API_KEY env var)"
        )

        # OpenAI-compatible API (Groq, Ollama, LM Studio, vLLM, etc.)
        parser.add_argument(
            "--openai-api-base",
            default=os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1"),
            help="Base URL for OpenAI-compatible API (e.g., Groq, Ollama, LM Studio)"
        )
        parser.add_argument(
            "--openai-api-key",
            default=os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            help="API key for OpenAI-compatible API (uses GROQ_API_KEY or OPENAI_API_KEY)"
        )
        parser.add_argument(
            "--openai-models",
            default=os.environ.get("OPENAI_MODELS", ""),
            help="Comma-separated list of models for round-robin (e.g., 'model1,model2,model3')"
        )

        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--compress", "-c",
            action="store_true",
            help="Compress MQTT payloads in console output (removes noise fields)"
        )
        parser.add_argument(
            "--demo",
            action="store_true",
            help="Enable demo mode"
        )
        parser.add_argument(
            "--no-ai",
            action="store_true",
            help="Disable AI calls (logging only mode)"
        )
        parser.add_argument(
            "--test-ai",
            action="store_true",
            help="Test AI connection before starting daemon"
        )
        parser.add_argument(
            "--disable-new-rules",
            action="store_true",
            default=os.environ.get("DISABLE_NEW_RULES", "").lower() in ("1", "true", "yes"),
            help="New rules are disabled by default (env: DISABLE_NEW_RULES)"
        )

        # Simulation mode
        parser.add_argument(
            "--simulation",
            metavar="FILE",
            help="Run in simulation mode using the specified scenario JSON file"
        )
        parser.add_argument(
            "--simulation-speed",
            type=float,
            metavar="MULTIPLIER",
            help="Override speed multiplier for simulation (e.g., 10 = 10x faster)"
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Run in test mode: validate assertions from scenario and exit with status"
        )
        parser.add_argument(
            "--report",
            metavar="FILE",
            help="Write JSON test report to specified file (requires --test)"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Write HTTP call details (URL, body, content-length) to debug-output/ directory"
        )
        parser.add_argument(
            "--disable-interval-trigger",
            action="store_true",
            help="Disable time-based AI triggers (interval checks)"
        )
        parser.add_argument(
            "--disable-threshold-trigger",
            action="store_true",
            help="Disable message count-based AI triggers (threshold checks)"
        )

        # Telegram Bot integration
        parser.add_argument(
            "--telegram-token",
            default=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            help="Telegram Bot token from BotFather (env: TELEGRAM_BOT_TOKEN)"
        )
        parser.add_argument(
            "--telegram-chat-ids",
            default=os.environ.get("TELEGRAM_CHAT_IDS", ""),
            help="Comma-separated list of authorized Telegram chat IDs (env: TELEGRAM_CHAT_IDS)"
        )
        parser.add_argument(
            "--no-telegram",
            action="store_true",
            help="Disable Telegram bot even if token is configured"
        )

        args = parser.parse_args()

        c = cls()
        c.mqtt_host = args.mqtt_host
        c.mqtt_port = args.mqtt_port
        c.ai_provider = args.ai_provider
        c.gemini_model = args.gemini_model
        c.gemini_api_key = args.gemini_api_key
        c.claude_model = args.claude_model
        c.claude_api_key = args.claude_api_key
        c.openai_api_base = args.openai_api_base
        c.openai_api_key = args.openai_api_key
        # Parse comma-separated models list, use defaults if empty
        if args.openai_models.strip():
            c.openai_models = [m.strip() for m in args.openai_models.split(",") if m.strip()]
        # else: keep default from dataclass
        c.verbose = args.verbose
        c.compress_output = args.compress
        c.demo_mode = args.demo
        c.no_ai = args.no_ai
        c.test_ai = args.test_ai
        c.disable_new_rules = args.disable_new_rules
        c.simulation_file = args.simulation
        c.simulation_speed = args.simulation_speed
        c.test_mode = args.test
        c.test_report_file = args.report
        c.debug_output = args.debug
        c.disable_interval_trigger = args.disable_interval_trigger
        c.disable_threshold_trigger = args.disable_threshold_trigger

        # Telegram configuration
        c.telegram_bot_token = args.telegram_token
        c.telegram_chat_ids = args.telegram_chat_ids
        c.telegram_enabled = bool(args.telegram_token) and not args.no_telegram
        return c

